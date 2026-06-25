#!/usr/bin/env python3
"""Serve RoboMeter progress through VLA-RL pickle HTTP RPC.

The RLT actor only knows how to call ``predict_progress(request)`` and expects
absolute progress values. This adapter keeps RoboMeter-specific transport and
trajectory-prefix handling outside the training loop:

RLT request -> this RPC adapter -> RoboMeter /evaluate_batch_npy -> progress.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
import io
import json
import logging
import sys
from pathlib import Path
import signal
import threading
import time
from typing import Any, Sequence

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vla_rl.runtime.remote_http import make_pickle_rpc_handler

LOGGER = logging.getLogger("serve_robometer_progress_http")


@dataclass(slots=True)
class _EpisodeCache:
    task: str = ""
    frames_by_index: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class _EpisodeSnapshot:
    task: str
    frames_by_index: dict[int, dict[str, np.ndarray]]


@dataclass(slots=True)
class _ProgressResult:
    progress: list[float]
    progress_by_key: dict[str, list[float]]
    sampled_indices: list[int]


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [int(item) for item in value]
    return [int(value)]


def _normalize_image(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"expected image with 3 dimensions, got shape={image.shape}")

    # Accept either HWC or CHW. LIBERO observations are normally HWC uint8.
    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"expected RGB image, got shape={image.shape}")

    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        if image.size and float(np.nanmax(image)) <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _extract_images(payload: Any, image_keys: Sequence[str]) -> dict[str, np.ndarray]:
    if not isinstance(payload, dict):
        raise TypeError(f"trajectory item must be a dict, got {type(payload).__name__}")
    raw_images = payload.get("images", {})
    if not isinstance(raw_images, dict):
        raise TypeError(f"trajectory item images must be a dict, got {type(raw_images).__name__}")

    selected_keys = list(image_keys) if image_keys else sorted(str(key) for key in raw_images)
    images: dict[str, np.ndarray] = {}
    for key in selected_keys:
        if key in raw_images:
            images[str(key)] = _normalize_image(raw_images[key])
    if not images:
        raise ValueError(f"no configured image keys found in request; available={sorted(raw_images)}")
    return images


def _task_from_request(request: dict[str, Any], trajectory: Sequence[Any]) -> str:
    task = str(request.get("task") or "")
    if task:
        return task
    for item in trajectory:
        if isinstance(item, dict) and item.get("task"):
            return str(item["task"])
    return ""


def _query_absolute_indices(
    request: dict[str, Any],
    *,
    trajectory_indices: Sequence[int],
    trajectory_len: int,
) -> list[int]:
    absolute_query_indices = _int_list(request.get("absolute_query_indices", []))
    if absolute_query_indices:
        return absolute_query_indices

    query_indices = _int_list(request.get("query_indices", []))
    if not query_indices:
        return list(trajectory_indices)
    if all(0 <= idx < trajectory_len for idx in query_indices):
        return [int(trajectory_indices[idx]) for idx in query_indices]
    return query_indices


def _select_sampled_indices(
    available_indices: Sequence[int],
    required_indices: Sequence[int],
    max_frames: int,
) -> list[int]:
    available = sorted({int(idx) for idx in available_indices})
    available_set = set(available)
    required = sorted({int(idx) for idx in required_indices if int(idx) in available_set})
    if not available:
        return []
    if max_frames <= 0 or len(available) <= max_frames:
        return available
    if len(required) >= max_frames:
        return required

    linspace_positions = np.linspace(0, len(available) - 1, max_frames, dtype=int).tolist()
    selected = {available[pos] for pos in linspace_positions}
    selected.update(required)

    while len(selected) > max_frames:
        removable = sorted(idx for idx in selected if idx not in required)
        if not removable:
            break
        selected.remove(removable[len(removable) // 2])

    if len(selected) < max_frames:
        for idx in available:
            selected.add(idx)
            if len(selected) >= max_frames:
                break
    return sorted(selected)


def _view_keys_for_mode(
    frames_by_index: dict[int, dict[str, np.ndarray]],
    sampled_indices: Sequence[int],
    view_mode: str,
) -> list[str]:
    if not sampled_indices:
        return []
    common_keys: set[str] | None = None
    for idx in sampled_indices:
        keys = set(frames_by_index[int(idx)])
        common_keys = keys if common_keys is None else common_keys & keys
    keys = sorted(common_keys or [])
    if not keys:
        return []
    if view_mode in {"first", "main"}:
        return [keys[0]]
    if view_mode == "average_two":
        return keys[:2]
    if view_mode == "average_all":
        return keys
    raise ValueError(f"unsupported view_mode: {view_mode}")


def _npy_file_tuple(array: np.ndarray, filename: str) -> tuple[str, io.BytesIO, str]:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    buffer.seek(0)
    return filename, buffer, "application/octet-stream"


def _make_progress_sample(frames: np.ndarray, task: str, sample_id: str, view_name: str) -> dict[str, Any]:
    return {
        "sample_type": "progress",
        "trajectory": {
            "frames": frames,
            "frames_shape": list(frames.shape),
            "task": task,
            "id": sample_id,
            "metadata": {
                "subsequence_length": int(frames.shape[0]),
                "view_name": view_name,
                "source": "vla_rl_rlt_online",
            },
            "video_embeddings": None,
            "text_embedding": None,
        },
    }


def _build_multipart_payload(samples: Sequence[dict[str, Any]], *, use_frame_steps: bool) -> tuple[dict[str, Any], dict[str, str]]:
    files: dict[str, Any] = {}
    data: dict[str, str] = {"use_frame_steps": "true" if use_frame_steps else "false"}

    for sample_idx, sample in enumerate(samples):
        trajectory = sample["trajectory"]
        frames = trajectory["frames"]
        file_key = f"sample_{sample_idx}_trajectory_frames"
        files[file_key] = _npy_file_tuple(np.asarray(frames), f"{file_key}.npy")
        sample_json = {
            **sample,
            "trajectory": {
                **trajectory,
                "frames": {"__numpy_file__": file_key},
            },
        }
        data[f"sample_{sample_idx}"] = json.dumps(sample_json)
    return files, data


def _as_float_list(value: Any) -> list[float]:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(item) for item in array]


class RoboMeterProgressHttpService:
    def __init__(
        self,
        *,
        robometer_url: str,
        image_keys: Sequence[str],
        view_mode: str,
        max_history_frames: int,
        use_frame_steps: bool,
        timeout: float,
        response_key: str,
        clamp_progress: bool,
    ) -> None:
        self.robometer_url = str(robometer_url).rstrip("/")
        self.image_keys = tuple(str(key) for key in image_keys)
        self.view_mode = str(view_mode)
        self.max_history_frames = int(max_history_frames)
        self.use_frame_steps = bool(use_frame_steps)
        self.timeout = float(timeout)
        self.response_key = str(response_key)
        self.clamp_progress = bool(clamp_progress)
        self.lock = threading.Lock()
        self.episodes: dict[str, _EpisodeCache] = {}
        self.stats: dict[str, Any] = {
            "requests": 0,
            "errors": 0,
            "episodes_cached": 0,
            "last_latency_sec": 0.0,
            "last_progress": None,
        }

    def predict_progress(self, request: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        trajectory = request.get("trajectory", [])
        if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes, bytearray)):
            raise TypeError("request.trajectory must be a sequence")
        if not trajectory:
            return {"progress": []}

        trajectory_indices = _int_list(request.get("trajectory_indices", []))
        if len(trajectory_indices) != len(trajectory):
            trajectory_indices = list(range(len(trajectory)))
        query_abs = _query_absolute_indices(
            request,
            trajectory_indices=trajectory_indices,
            trajectory_len=len(trajectory),
        )
        if not query_abs:
            return {"progress": []}

        episode_id = str(request.get("episode_id", "episode"))
        task = _task_from_request(request, trajectory)
        metadata = dict(request.get("metadata", {}) or {})

        with self.lock:
            cache = self.episodes.setdefault(episode_id, _EpisodeCache(task=task))
            if task:
                cache.task = task
            cache.updated_at = time.time()
            for idx, payload in zip(trajectory_indices, trajectory):
                cache.frames_by_index[int(idx)] = _extract_images(payload, self.image_keys)
            snapshot = _EpisodeSnapshot(
                task=cache.task,
                frames_by_index={
                    int(idx): dict(images)
                    for idx, images in cache.frames_by_index.items()
                },
            )

        try:
            result = self._predict_from_snapshot(
                episode_id=episode_id,
                snapshot=snapshot,
                query_abs=query_abs,
            )
        except Exception:
            if bool(metadata.get("done", False) or metadata.get("truncated", False)):
                with self.lock:
                    self.episodes.pop(episode_id, None)
                    self.stats["episodes_cached"] = len(self.episodes)
            raise
        progress = result.progress
        if self.clamp_progress:
            progress = [max(0.0, min(1.0, value)) for value in progress]

        latency = time.perf_counter() - start
        done = bool(metadata.get("done", False) or metadata.get("truncated", False))
        with self.lock:
            self.stats["requests"] += 1
            self.stats["episodes_cached"] = len(self.episodes)
            self.stats["last_latency_sec"] = float(latency)
            self.stats["last_progress"] = progress[-1] if progress else None
            if done:
                self.episodes.pop(episode_id, None)
                self.stats["episodes_cached"] = len(self.episodes)

        LOGGER.info(
            "request ok: episode=%s task=%r query=%s sampled=%s progress=%s latency=%.3fs",
            episode_id,
            snapshot.task,
            query_abs,
            result.sampled_indices,
            [round(value, 4) for value in progress],
            latency,
        )
        return {
            "progress": progress,
            "progress_predictions_by_key": {
                self.response_key: {"progress": progress},
                **{key: {"progress": values} for key, values in result.progress_by_key.items()},
            },
            "metadata": {
                "sampled_indices": result.sampled_indices,
                "view_mode": self.view_mode,
                "use_frame_steps": self.use_frame_steps,
            },
        }

    def _predict_from_snapshot(
        self,
        *,
        episode_id: str,
        snapshot: _EpisodeSnapshot,
        query_abs: Sequence[int],
    ) -> _ProgressResult:
        available_indices = sorted(snapshot.frames_by_index)
        required_indices = [0, *[int(idx) for idx in query_abs], available_indices[-1]]
        sampled_indices = _select_sampled_indices(
            available_indices,
            required_indices,
            max_frames=self.max_history_frames,
        )
        missing = [int(idx) for idx in query_abs if int(idx) not in sampled_indices]
        if missing:
            raise RuntimeError(
                f"query indices {missing} were not retained in sampled prefix {sampled_indices}"
            )

        view_keys = _view_keys_for_mode(snapshot.frames_by_index, sampled_indices, self.view_mode)
        if not view_keys:
            raise RuntimeError(
                f"no common view keys for sampled prefix; available_indices={sampled_indices}"
            )

        samples = []
        for view_key in view_keys:
            frames = np.stack(
                [snapshot.frames_by_index[int(idx)][view_key] for idx in sampled_indices],
                axis=0,
            )
            sample_id = f"{episode_id}:{view_key}:{sampled_indices[0]}-{sampled_indices[-1]}"
            samples.append(_make_progress_sample(frames, snapshot.task, sample_id, view_key))

        progress_by_view = self._post_progress_batch(samples)
        index_to_position = {int(idx): pos for pos, idx in enumerate(sampled_indices)}
        selected_by_view: dict[str, list[float]] = {}
        full_progress_arrays = []
        for view_key, progress_array in zip(view_keys, progress_by_view):
            progress_values = _as_float_list(progress_array)
            if len(progress_values) != len(sampled_indices):
                raise RuntimeError(
                    f"RoboMeter returned {len(progress_values)} progress values for view {view_key}, "
                    f"expected {len(sampled_indices)}"
                )
            selected = [float(progress_values[index_to_position[int(idx)]]) for idx in query_abs]
            selected_by_view[f"robometer_{view_key}"] = selected
            full_progress_arrays.append(np.asarray(selected, dtype=np.float32))

        fused = np.stack(full_progress_arrays, axis=0).mean(axis=0).astype(np.float32).tolist()
        return _ProgressResult(
            progress=[float(value) for value in fused],
            progress_by_key=selected_by_view,
            sampled_indices=[int(idx) for idx in sampled_indices],
        )

    def _post_progress_batch(self, samples: Sequence[dict[str, Any]]) -> list[list[float]]:
        files, data = _build_multipart_payload(samples, use_frame_steps=self.use_frame_steps)
        try:
            response = requests.post(
                self.robometer_url + "/evaluate_batch_npy",
                files=files,
                data=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        finally:
            for file_tuple in files.values():
                file_tuple[1].close()
        outputs_progress = payload.get("outputs_progress")
        if not isinstance(outputs_progress, dict):
            raise RuntimeError(f"RoboMeter response missing outputs_progress: {payload.keys()}")
        progress_pred = outputs_progress.get("progress_pred", [])
        if len(progress_pred) != len(samples):
            raise RuntimeError(f"expected {len(samples)} progress outputs, got {len(progress_pred)}")
        return [_as_float_list(item) for item in progress_pred]

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
    parser.add_argument("--robometer-url", default="http://127.0.0.1:8401")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50152)
    parser.add_argument("--image-keys", nargs="+", default=["image_rgb_0", "image_rgb_1"])
    parser.add_argument(
        "--view-mode",
        choices=["first", "main", "average_two", "average_all"],
        default="average_two",
    )
    parser.add_argument("--max-history-frames", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--response-key", default="robometer")
    parser.add_argument("--no-frame-steps", dest="use_frame_steps", action="store_false")
    parser.add_argument("--use-frame-steps", dest="use_frame_steps", action="store_true", default=True)
    parser.add_argument("--no-clamp-progress", dest="clamp_progress", action="store_false")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level)),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    service = RoboMeterProgressHttpService(
        robometer_url=str(args.robometer_url),
        image_keys=list(args.image_keys),
        view_mode=str(args.view_mode),
        max_history_frames=int(args.max_history_frames),
        use_frame_steps=bool(args.use_frame_steps),
        timeout=float(args.timeout_s),
        response_key=str(args.response_key),
        clamp_progress=bool(args.clamp_progress),
    )
    handler = make_pickle_rpc_handler(service.dispatch)
    server = ThreadingHTTPServer((str(args.host), int(args.port)), handler)

    def request_stop(signum: int, _frame: Any) -> None:
        LOGGER.info("received signal %s; stopping server", signum)
        server.shutdown()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    LOGGER.info(
        "RoboMeter progress adapter listening on http://%s:%s; robometer_url=%s view_mode=%s use_frame_steps=%s",
        args.host,
        args.port,
        args.robometer_url,
        args.view_mode,
        args.use_frame_steps,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        LOGGER.info("server stopped; stats=%s", service.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
