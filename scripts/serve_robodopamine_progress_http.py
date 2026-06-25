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
from typing import Any, Sequence

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


def _load_robodopamine(root: Path) -> tuple[type[Any], type[Any]]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Robo-Dopamine root does not exist: {root}")
    sys.path.insert(0, str(root))
    from scripts.serve_grm_reward import GrmProgressEngine, LiberoGoalProvider

    return GrmProgressEngine, LiberoGoalProvider


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
        query_indices = [int(idx) for idx in request.get("query_indices", [])]
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

        transitions = [_as_transition(dict(item), step_in_episode=idx, task_id=task_id) for idx, item in enumerate(trajectory)]
        start_idx = int(request.get("trajectory_start_idx", 0) or 0)
        episode_id = request.get("episode_id", "episode")
        request_label = f"{episode_id}_{min(query_indices)}_{max(query_indices)}"

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
            query_indices,
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
    parser.add_argument("--goal-dataset", default="/vla/users/yixin/LIBERO/Libero_Lerobot")
    parser.add_argument("--goal-image-key", default="image")
    parser.add_argument("--allow-task-id-fallback", action="store_true")
    parser.add_argument("--allow-missing-goal", action="store_true")
    parser.add_argument("--out-root", default="/tmp/robodopamine_reward_http")
    parser.add_argument("--keep-runs", action="store_true")
    parser.add_argument("--no-clip-progress", action="store_true")
    parser.add_argument("--response-key", default="robodopamine")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--vllm-attention-backend", default=os.environ.get("VLLM_ATTENTION_BACKEND", "TORCH_SDPA"))
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level)), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ["VLLM_ATTENTION_BACKEND"] = str(args.vllm_attention_backend)

    GrmProgressEngine, LiberoGoalProvider = _load_robodopamine(Path(args.robodopamine_root))
    goal_provider = None
    if args.goal_dataset:
        goal_provider = LiberoGoalProvider(
            dataset_path=str(args.goal_dataset),
            image_key=str(args.goal_image_key),
            allow_task_id_fallback=bool(args.allow_task_id_fallback),
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
