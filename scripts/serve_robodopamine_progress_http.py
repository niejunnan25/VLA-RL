#!/usr/bin/env python3
"""Serve Robo-Dopamine progress through VLA-RL pickle HTTP RPC.

The VLA-RL actor sends request dictionaries to ``predict_progress`` and expects
absolute progress values. This adapter keeps the RL repo independent from
Robo-Dopamine transport details while reusing Robo-Dopamine's GRM engine.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
import io
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import types
from typing import Any, Callable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vla_rl.runtime.remote_http import make_pickle_rpc_handler

LOGGER = logging.getLogger("serve_robodopamine_progress_http")


@dataclass(slots=True)
class _ArrayPayload:
    data: bytes


@dataclass(slots=True)
class _TransitionPayload:
    obs: dict[str, _ArrayPayload]
    step_in_episode: int


def _ndarray_to_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def _as_transition(payload: dict[str, Any], *, step_in_episode: int, task_id: int | None) -> _TransitionPayload:
    images = payload.get("images", {}) if isinstance(payload, dict) else {}
    if not isinstance(images, dict):
        raise TypeError(f"trajectory[{step_in_episode}].images must be a dict, got {type(images).__name__}")
    obs = {str(key): _ArrayPayload(_ndarray_to_bytes(value)) for key, value in images.items()}
    if task_id is not None:
        obs["task_id"] = _ArrayPayload(_ndarray_to_bytes(np.asarray([int(task_id)], dtype=np.int32)))
    return _TransitionPayload(obs=obs, step_in_episode=int(step_in_episode))


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [int(item) for item in value]
    return [int(value)]


def _compact_query_indices(request: dict[str, Any], *, trajectory_len: int, trajectory_indices: list[int]) -> list[int]:
    query_indices = _int_list(request.get("query_indices", []))
    if not query_indices:
        query_indices = list(range(trajectory_len))
    if all(0 <= int(idx) < trajectory_len for idx in query_indices):
        return query_indices

    absolute_query_indices = _int_list(request.get("absolute_query_indices", []))
    if absolute_query_indices:
        index_to_local = {int(index): local for local, index in enumerate(trajectory_indices)}
        try:
            return [index_to_local[int(index)] for index in absolute_query_indices]
        except KeyError as exc:
            raise IndexError(
                f"absolute query index {int(exc.args[0])} is not present in trajectory_indices={trajectory_indices}"
            ) from exc
    raise IndexError(f"query_indices={query_indices} out of range for compact trajectory length {trajectory_len}")


def _compact_start_idx(request: dict[str, Any], *, trajectory_len: int, trajectory_indices: list[int]) -> int:
    raw_start_idx = request.get("trajectory_start_idx", 0)
    start_idx = int(raw_start_idx or 0)
    if 0 <= start_idx < trajectory_len:
        return start_idx
    index_to_local = {int(index): local for local, index in enumerate(trajectory_indices)}
    return int(index_to_local.get(start_idx, 0))


GOAL_SCAN_SKIP_DIRS = frozenset({"data", "videos", ".git", "__pycache__"})


def _install_robodopamine_examples_package(root: Path) -> None:
    examples_dir = root / "examples"
    if not examples_dir.is_dir():
        return

    for module_name in list(sys.modules):
        if module_name == "examples" or module_name.startswith("examples."):
            del sys.modules[module_name]

    examples_module = types.ModuleType("examples")
    examples_module.__file__ = str(examples_dir)
    examples_module.__path__ = [str(examples_dir)]
    examples_module.__package__ = "examples"
    sys.modules["examples"] = examples_module


def _load_robodopamine(root: Path) -> tuple[type[Any], type[Any], Callable[[str], str], type[Any]]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Robo-Dopamine root does not exist: {root}")
    sys.path.insert(0, str(root))
    _install_robodopamine_examples_package(root)
    from scripts.serve_grm_reward import (
        GoalLookupResult,
        GrmProgressEngine,
        LiberoGoalProvider,
        normalize_task,
    )

    return GrmProgressEngine, LiberoGoalProvider, normalize_task, GoalLookupResult


def _is_lerobot_goal_dataset(path: Path) -> bool:
    return (
        (path / "meta" / "info.json").is_file()
        and (path / "meta" / "episodes.jsonl").is_file()
    )


def _discover_lerobot_goal_datasets(dataset_path: Path) -> list[Path]:
    dataset_path = dataset_path.expanduser().resolve()
    if _is_lerobot_goal_dataset(dataset_path):
        return [dataset_path]

    roots: list[Path] = []
    for current, dirnames, _filenames in os.walk(dataset_path):
        current_path = Path(current)
        if _is_lerobot_goal_dataset(current_path):
            roots.append(current_path)
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if name not in GOAL_SCAN_SKIP_DIRS]
    return sorted(set(roots))


class PromptRoutingLiberoGoalProvider:
    """Route a LIBERO prompt to the matching LeRobot task dataset.

    Robo-Dopamine's native ``LiberoGoalProvider`` expects one LeRobot dataset
    root. Our LIBERO copy is organized as suite/task subdirectories, so this
    wrapper builds a prompt index across all child datasets and then delegates
    final-frame loading to the native provider.
    """

    def __init__(
        self,
        *,
        dataset_roots: Sequence[Path],
        provider_cls: type[Any],
        normalize_task_fn: Callable[[str], str],
        goal_result_cls: type[Any],
        image_key: str,
        allow_task_id_fallback: bool,
    ) -> None:
        self.normalize_task = normalize_task_fn
        self.goal_result_cls = goal_result_cls
        self.task_to_provider: dict[str, Any] = {}
        self.provider_roots: dict[int, Path] = {}
        self.task_id_to_provider: dict[int, Any] = {}
        ambiguous_task_ids: set[int] = set()

        for root in dataset_roots:
            provider = provider_cls(
                dataset_path=str(root),
                image_key=str(image_key),
                allow_task_id_fallback=bool(allow_task_id_fallback),
            )
            self.provider_roots[id(provider)] = root
            for task_key in getattr(provider, "task_to_episode", {}):
                existing = self.task_to_provider.get(task_key)
                if existing is not None:
                    LOGGER.warning(
                        "duplicate LIBERO goal prompt %r in %s and %s; keeping first",
                        task_key,
                        self.provider_roots[id(existing)],
                        root,
                    )
                    continue
                self.task_to_provider[task_key] = provider

            for task_id in getattr(provider, "task_index_to_episode", {}):
                task_id = int(task_id)
                existing_provider = self.task_id_to_provider.get(task_id)
                if existing_provider is not None and existing_provider is not provider:
                    ambiguous_task_ids.add(task_id)
                    continue
                self.task_id_to_provider[task_id] = provider

        for task_id in ambiguous_task_ids:
            self.task_id_to_provider.pop(task_id, None)

        LOGGER.info(
            "loaded LIBERO goal router: %d dataset roots, %d prompt entries, %d unique task-id entries, %d ambiguous task ids",
            len(dataset_roots),
            len(self.task_to_provider),
            len(self.task_id_to_provider),
            len(ambiguous_task_ids),
        )

    def _empty_result(self) -> Any:
        return self.goal_result_cls(image=None, source=None)

    def get_goal(self, task: str, task_id: int | None) -> Any:
        task_key = self.normalize_task(str(task or ""))
        provider = self.task_to_provider.get(task_key) if task_key else None
        if provider is not None:
            result = provider.get_goal(task, task_id=None)
            if result.image is None:
                return result
            root = self.provider_roots.get(id(provider))
            source = f"{root}:{result.source}" if root is not None else result.source
            return self.goal_result_cls(image=result.image, source=source)

        if task_id is not None:
            provider = self.task_id_to_provider.get(int(task_id))
            if provider is not None:
                LOGGER.warning(
                    "using unique task_id fallback for LIBERO goal lookup: task_id=%s",
                    task_id,
                )
                result = provider.get_goal(task, task_id=int(task_id))
                if result.image is None:
                    return result
                root = self.provider_roots.get(id(provider))
                source = f"{root}:{result.source}" if root is not None else result.source
                return self.goal_result_cls(image=result.image, source=source)

        return self._empty_result()


def _build_libero_goal_provider(
    *,
    dataset_path: str,
    image_key: str,
    allow_task_id_fallback: bool,
    provider_cls: type[Any],
    normalize_task_fn: Callable[[str], str],
    goal_result_cls: type[Any],
) -> Any:
    roots = _discover_lerobot_goal_datasets(Path(dataset_path))
    if not roots:
        expected = "meta/info.json and meta/episodes.jsonl"
        raise FileNotFoundError(
            f"no LeRobot LIBERO goal datasets found under {dataset_path}; expected {expected}"
        )
    if len(roots) == 1:
        LOGGER.info("using single LIBERO goal dataset: %s", roots[0])
        return provider_cls(
            dataset_path=str(roots[0]),
            image_key=str(image_key),
            allow_task_id_fallback=bool(allow_task_id_fallback),
        )
    LOGGER.info("using LIBERO goal dataset router over %d child datasets under %s", len(roots), dataset_path)
    return PromptRoutingLiberoGoalProvider(
        dataset_roots=roots,
        provider_cls=provider_cls,
        normalize_task_fn=normalize_task_fn,
        goal_result_cls=goal_result_cls,
        image_key=str(image_key),
        allow_task_id_fallback=bool(allow_task_id_fallback),
    )


class RoboDopamineProgressHttpService:
    def __init__(self, *, engine: Any, goal_provider: Any | None, require_goal: bool, response_key: str) -> None:
        self.engine = engine
        self.goal_provider = goal_provider
        self.require_goal = bool(require_goal)
        self.response_key = str(response_key)
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "errors": 0}

    def predict_progress(self, request: dict[str, Any]) -> dict[str, Any]:
        trajectory = request.get("trajectory", [])
        if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes, bytearray)):
            raise TypeError("request.trajectory must be a sequence")
        trajectory_indices = _int_list(request.get("trajectory_indices", []))
        if len(trajectory_indices) != len(trajectory):
            trajectory_indices = list(range(len(trajectory)))
        query_indices = _compact_query_indices(
            request,
            trajectory_len=len(trajectory),
            trajectory_indices=trajectory_indices,
        )
        if not trajectory or not query_indices:
            return {"progress": []}

        metadata = dict(request.get("metadata", {}) or {})
        task_id = metadata.get("task_id", None)
        task_id = None if task_id is None else int(task_id)
        task = str(request.get("task") or "")
        if not task:
            for item in trajectory:
                if isinstance(item, dict) and item.get("task"):
                    task = str(item["task"])
                    break

        transitions = [
            _as_transition(dict(item), step_in_episode=trajectory_indices[idx], task_id=task_id)
            for idx, item in enumerate(trajectory)
        ]
        start_idx = _compact_start_idx(
            request,
            trajectory_len=len(trajectory),
            trajectory_indices=trajectory_indices,
        )
        absolute_query_indices = _int_list(request.get("absolute_query_indices", []))
        query_label = absolute_query_indices if absolute_query_indices else query_indices
        episode_id = request.get("episode_id", "episode")
        request_label = f"{episode_id}_{min(query_label)}_{max(query_label)}"

        goal_image = None
        goal_source = None
        if self.goal_provider is not None:
            goal_result = self.goal_provider.get_goal(task, task_id=task_id)
            goal_image = goal_result.image
            goal_source = goal_result.source
        if goal_image is None and self.require_goal:
            raise RuntimeError(f"no expert goal image found for task_id={task_id}, task={task!r}")

        progress = self.engine.predict(
            transitions=transitions,
            query_indices=query_indices,
            trajectory_start_idx=start_idx,
            task=task,
            goal_image=goal_image,
            request_label=request_label,
        )
        progress = [float(value) for value in progress]
        with self.lock:
            self.stats["requests"] += 1
        LOGGER.info(
            "request ok: episode=%s task=%r query=%s progress=%s goal=%s",
            episode_id,
            task,
            query_label,
            [round(value, 4) for value in progress],
            goal_source,
        )
        return {"progress": progress, "progress_predictions_by_key": {self.response_key: {"progress": progress}}}

    def dispatch(self, method: str, kwargs: dict[str, Any]) -> Any:
        try:
            if method != "predict_progress":
                raise ValueError(f"unsupported RPC method: {method}")
            return self.predict_progress(dict(kwargs.get("request", {})))
        except Exception:
            with self.lock:
                self.stats["errors"] += 1
            LOGGER.exception("request failed")
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodopamine-root", default="/vla/users/niejunnan/codebase/Robo-Dopamine")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--image-keys", nargs="+", default=["image_rgb_0", "image_rgb_1"])
    parser.add_argument("--view-mode", choices=["two_view_copy_main", "three_view"], default="two_view_copy_main")
    parser.add_argument("--image-preprocess", choices=["libero", "none"], default="none")
    parser.add_argument("--goal-image-preprocess", choices=["none", "libero"], default="none")
    parser.add_argument("--eval-modes", nargs="+", choices=["forward", "incremental", "backward"], default=["forward"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--goal-dataset", default="/vla/users/niejunnan/datasets/libero_lerobot")
    parser.add_argument("--goal-image-key", default="image")
    parser.add_argument("--allow-task-id-fallback", action="store_true")
    parser.add_argument("--allow-missing-goal", action="store_true")
    parser.add_argument("--out-root", default="/tmp/robodopamine_reward_http")
    parser.add_argument("--keep-runs", action="store_true")
    parser.add_argument("--no-clip-progress", action="store_true")
    parser.add_argument("--response-key", default="robodopamine")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument(
        "--vllm-attention-backend",
        default=os.environ.get("VLLM_ATTENTION_BACKEND", "TORCH_SDPA"),
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level)),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ["VLLM_ATTENTION_BACKEND"] = str(args.vllm_attention_backend)

    GrmProgressEngine, LiberoGoalProvider, normalize_task, GoalLookupResult = _load_robodopamine(
        Path(args.robodopamine_root)
    )
    goal_provider = None
    if args.goal_dataset:
        goal_provider = _build_libero_goal_provider(
            dataset_path=str(args.goal_dataset),
            image_key=str(args.goal_image_key),
            allow_task_id_fallback=bool(args.allow_task_id_fallback),
            provider_cls=LiberoGoalProvider,
            normalize_task_fn=normalize_task,
            goal_result_cls=GoalLookupResult,
        )
    engine = GrmProgressEngine(
        model_path=str(args.model_path),
        eval_modes=list(args.eval_modes),
        batch_size=int(args.batch_size),
        image_keys=list(args.image_keys),
        view_mode=str(args.view_mode),
        image_preprocess=str(args.image_preprocess),
        goal_image_preprocess=str(args.goal_image_preprocess),
        out_root=str(args.out_root),
        keep_runs=bool(args.keep_runs),
        clip=not bool(args.no_clip_progress),
    )
    service = RoboDopamineProgressHttpService(
        engine=engine,
        goal_provider=goal_provider,
        require_goal=not bool(args.allow_missing_goal),
        response_key=str(args.response_key),
    )
    handler = make_pickle_rpc_handler(service.dispatch)
    server = ThreadingHTTPServer((str(args.host), int(args.port)), handler)

    def request_stop(signum: int, _frame: Any) -> None:
        LOGGER.info("received signal %s; stopping server", signum)
        server.shutdown()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    LOGGER.info("Robo-Dopamine HTTP progress server listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        LOGGER.info("server stopped; stats=%s", service.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
