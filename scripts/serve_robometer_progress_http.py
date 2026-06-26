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
import os
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
    sampled_indices: list[list[int]]


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


def _cache_key_from_request(request: dict[str, Any], metadata: dict[str, Any]) -> str:
    session_id = str(
        request.get("session_id")
        or metadata.get("session_id")
        or request.get("run_id")
        or metadata.get("run_id")
        or "default"
    )
    episode_id = str(request.get("episode_id", "episode"))
    return f"{session_id}:{episode_id}"


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


class RoboMeterNativeBackend:
    """In-process RoboMeter inference backend.

    This follows RoboMeter's policy-learning relabel path: each query gets its
    own prefix sample, then those samples are batched for one model forward. It
    avoids the extra official eval-server process and repeated multipart .npy
    transfer while preserving the prefix-per-query reward semantics.
    """

    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        forward_batch_size: int,
        robometer_root: str | None = None,
    ) -> None:
        if robometer_root:
            root = Path(robometer_root).expanduser().resolve()
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))

        import torch
        from robometer.models.utils import convert_bins_to_continuous
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        self.torch = torch
        self.convert_bins_to_continuous = convert_bins_to_continuous
        self.device = torch.device(device)
        self.forward_batch_size = max(1, int(forward_batch_size))

        LOGGER.info("loading RoboMeter native backend from %s on %s", model_path, self.device)
        exp_config, tokenizer, processor, model = load_model_from_hf(
            model_path=str(model_path),
            device=self.device,
        )
        model = model.to(self.device)
        model.eval()

        self.exp_config = exp_config
        self.tokenizer = tokenizer
        self.model = model
        self.batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
        progress_loss_type = getattr(exp_config.loss, "progress_loss_type", "l2")
        self.is_discrete_mode = str(progress_loss_type).lower() == "discrete"
        self.num_bins = int(
            getattr(
                exp_config.loss,
                "progress_discrete_bins",
                getattr(exp_config.model, "progress_discrete_bins", 10),
            )
        )
        self.model_type = str(getattr(exp_config.model, "model_type", ""))
        self.lock = threading.Lock()

    def predict_progress_samples(self, samples: Sequence[dict[str, Any]]) -> list[list[float]]:
        if not samples:
            return []
        buckets: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for sample_idx, sample in enumerate(samples):
            buckets.setdefault(self._sample_frame_count(sample), []).append((sample_idx, sample))

        outputs: list[list[float] | None] = [None] * len(samples)
        with self.lock:
            with self.torch.inference_mode():
                for indexed_samples in buckets.values():
                    for start in range(0, len(indexed_samples), self.forward_batch_size):
                        indexed_chunk = indexed_samples[start : start + self.forward_batch_size]
                        chunk_indices = [sample_idx for sample_idx, _sample in indexed_chunk]
                        chunk = [sample for _sample_idx, sample in indexed_chunk]
                        chunk_outputs = self._predict_same_length_chunk(chunk)
                        if len(chunk_outputs) != len(chunk_indices):
                            raise RuntimeError(
                                f"expected {len(chunk_indices)} progress outputs, got {len(chunk_outputs)}"
                            )
                        for sample_idx, sample_output in zip(chunk_indices, chunk_outputs):
                            outputs[sample_idx] = sample_output
        if any(output is None for output in outputs):
            raise RuntimeError("RoboMeter native backend did not produce all requested outputs")
        return [list(output) for output in outputs if output is not None]

    @staticmethod
    def _sample_frame_count(sample: dict[str, Any]) -> int:
        trajectory = sample.get("trajectory", {})
        frames = trajectory.get("frames") if isinstance(trajectory, dict) else None
        if hasattr(frames, "shape"):
            return int(frames.shape[0])
        return len(frames) if frames is not None else 0

    def _predict_same_length_chunk(self, chunk: Sequence[dict[str, Any]]) -> list[list[float]]:
        frame_lengths = {self._sample_frame_count(sample) for sample in chunk}
        if len(frame_lengths) > 1:
            raise RuntimeError(f"native RoboMeter batch has mixed frame lengths: {sorted(frame_lengths)}")
        batch_inputs = self.batch_collator(chunk)["progress_inputs"]
        batch_inputs = {
            key: value.to(self.device) if isinstance(value, self.torch.Tensor) else value
            for key, value in batch_inputs.items()
        }
        return self._compute_progress_outputs(batch_inputs)

    def _compute_progress_outputs(self, batch_inputs: dict[str, Any]) -> list[list[float]]:
        if "rewind" in self.model.__class__.__name__.lower():
            model_output, _extra = self.model(
                video_embeddings=batch_inputs.get("video_embeddings"),
                text_embeddings=batch_inputs.get("text_embeddings"),
                sample_type="progress",
                timing_raw=None,
            )
        else:
            model_output, _extra = self.model(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                pixel_values=batch_inputs.get("pixel_values", None),
                pixel_values_videos=batch_inputs.get("pixel_values_videos", None),
                image_grid_thw=batch_inputs.get("image_grid_thw", None),
                video_grid_thw=batch_inputs.get("video_grid_thw", None),
                second_per_grid_ts=batch_inputs.get("second_per_grid_ts", None),
                sample_type="progress",
                timing_raw=None,
            )

        progress_logits = getattr(model_output, "progress_logits", None)
        if not isinstance(progress_logits, dict):
            return [[] for _ in range(int(batch_inputs["input_ids"].shape[0]))]
        seq_a = progress_logits.get("A")
        if seq_a is None:
            return [[] for _ in range(int(batch_inputs["input_ids"].shape[0]))]

        progress_pred: list[list[float]] = []
        for item in [seq_a[idx] for idx in range(seq_a.shape[0])]:
            if self.is_discrete_mode:
                continuous = self.convert_bins_to_continuous(item.detach().cpu().float())
                progress_pred.append(continuous.numpy().flatten().tolist())
            else:
                progress_pred.append(item.detach().cpu().flatten().tolist())
        return progress_pred


class RoboMeterProgressHttpService:
    def __init__(
        self,
        *,
        backend: str,
        robometer_url: str,
        native_backend: RoboMeterNativeBackend | None,
        image_keys: Sequence[str],
        view_mode: str,
        query_mode: str,
        max_history_frames: int,
        use_frame_steps: bool,
        timeout: float,
        response_key: str,
        clamp_progress: bool,
    ) -> None:
        self.backend = str(backend)
        self.robometer_url = str(robometer_url).rstrip("/")
        self.native_backend = native_backend
        self.image_keys = tuple(str(key) for key in image_keys)
        self.view_mode = str(view_mode)
        self.query_mode = str(query_mode)
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
        if self.backend not in {"native", "http"}:
            raise ValueError(f"unsupported backend: {self.backend}")
        if self.backend == "native" and self.native_backend is None:
            raise ValueError("native backend requires native_backend")
        if self.query_mode not in {"prefix_per_query", "sequence_multi_query"}:
            raise ValueError(f"unsupported query_mode: {self.query_mode}")

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
        cache_key = _cache_key_from_request(request, metadata)

        with self.lock:
            cache = self.episodes.setdefault(cache_key, _EpisodeCache(task=task))
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
                    self.episodes.pop(cache_key, None)
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
                self.episodes.pop(cache_key, None)
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
                "query_mode": self.query_mode,
                "backend": self.backend,
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
        if self.query_mode == "sequence_multi_query":
            return self._predict_sequence_multi_query(
                episode_id=episode_id,
                snapshot=snapshot,
                query_abs=query_abs,
            )
        return self._predict_prefix_per_query(
            episode_id=episode_id,
            snapshot=snapshot,
            query_abs=query_abs,
        )

    def _predict_sequence_multi_query(
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

        progress_by_view = self._predict_samples(samples)
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
            sampled_indices=[[int(idx) for idx in sampled_indices]],
        )

    def _predict_prefix_per_query(
        self,
        *,
        episode_id: str,
        snapshot: _EpisodeSnapshot,
        query_abs: Sequence[int],
    ) -> _ProgressResult:
        available_indices = sorted(snapshot.frames_by_index)
        available_set = set(available_indices)
        missing = [int(idx) for idx in query_abs if int(idx) not in available_set]
        if missing:
            raise RuntimeError(f"query indices {missing} are not cached; available={available_indices}")
        if not available_indices:
            return _ProgressResult(progress=[], progress_by_key={}, sampled_indices=[])

        view_keys = _view_keys_for_mode(snapshot.frames_by_index, available_indices, self.view_mode)
        if not view_keys:
            raise RuntimeError(f"no common view keys for available prefix; available_indices={available_indices}")

        samples: list[dict[str, Any]] = []
        sample_meta: list[tuple[str, int, list[int]]] = []
        sampled_by_query: list[list[int]] = []
        for query_index in [int(idx) for idx in query_abs]:
            prefix_available = [int(idx) for idx in available_indices if int(idx) <= query_index]
            sampled_indices = _select_sampled_indices(
                prefix_available,
                [prefix_available[0], query_index],
                max_frames=self.max_history_frames,
            )
            if query_index not in sampled_indices:
                raise RuntimeError(
                    f"query index {query_index} was not retained in prefix sample {sampled_indices}"
                )
            sampled_by_query.append([int(idx) for idx in sampled_indices])
            for view_key in view_keys:
                frames = np.stack(
                    [snapshot.frames_by_index[int(idx)][view_key] for idx in sampled_indices],
                    axis=0,
                )
                sample_id = f"{episode_id}:{view_key}:prefix-{query_index}"
                samples.append(_make_progress_sample(frames, snapshot.task, sample_id, view_key))
                sample_meta.append((view_key, query_index, [int(idx) for idx in sampled_indices]))

        progress_pred = self._predict_samples(samples)
        if len(progress_pred) != len(sample_meta):
            raise RuntimeError(f"expected {len(sample_meta)} progress outputs, got {len(progress_pred)}")

        selected_by_view = {f"robometer_{view_key}": [] for view_key in view_keys}
        for (view_key, _query_index, _sampled_indices), progress_array in zip(sample_meta, progress_pred):
            progress_values = _as_float_list(progress_array)
            selected_by_view[f"robometer_{view_key}"].append(
                float(progress_values[-1]) if progress_values else 0.0
            )

        fused = np.stack(
            [np.asarray(selected_by_view[f"robometer_{view_key}"], dtype=np.float32) for view_key in view_keys],
            axis=0,
        ).mean(axis=0).astype(np.float32).tolist()
        return _ProgressResult(
            progress=[float(value) for value in fused],
            progress_by_key=selected_by_view,
            sampled_indices=sampled_by_query,
        )

    def _predict_samples(self, samples: Sequence[dict[str, Any]]) -> list[list[float]]:
        if self.backend == "native":
            assert self.native_backend is not None
            return self.native_backend.predict_progress_samples(samples)
        return self._post_progress_batch(samples)

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
    parser.add_argument("--backend", choices=["native", "http"], default="native")
    parser.add_argument("--model-path", default="/vla/users/niejunnan/assets/Robometer-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forward-batch-size", type=int, default=4)
    parser.add_argument("--robometer-root", default=os.environ.get("ROBOMETER_ROOT"))
    parser.add_argument("--robometer-url", default="http://127.0.0.1:8401")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50152)
    parser.add_argument("--image-keys", nargs="+", default=["image_rgb_0"])
    parser.add_argument(
        "--view-mode",
        choices=["first", "main", "average_two", "average_all"],
        default="first",
    )
    parser.add_argument(
        "--query-mode",
        choices=["prefix_per_query", "sequence_multi_query"],
        default="prefix_per_query",
    )
    parser.add_argument("--max-history-frames", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--response-key", default="robometer")
    parser.add_argument("--no-frame-steps", dest="use_frame_steps", action="store_false")
    parser.add_argument("--use-frame-steps", dest="use_frame_steps", action="store_true")
    parser.set_defaults(use_frame_steps=False)
    parser.add_argument("--no-clamp-progress", dest="clamp_progress", action="store_false")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level)),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    native_backend = None
    if args.backend == "native":
        native_backend = RoboMeterNativeBackend(
            model_path=str(args.model_path),
            device=str(args.device),
            forward_batch_size=int(args.forward_batch_size),
            robometer_root=args.robometer_root,
        )

    service = RoboMeterProgressHttpService(
        backend=str(args.backend),
        robometer_url=str(args.robometer_url),
        native_backend=native_backend,
        image_keys=list(args.image_keys),
        view_mode=str(args.view_mode),
        query_mode=str(args.query_mode),
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
        "RoboMeter progress adapter listening on http://%s:%s; backend=%s robometer_url=%s "
        "view_mode=%s query_mode=%s use_frame_steps=%s",
        args.host,
        args.port,
        args.backend,
        args.robometer_url,
        args.view_mode,
        args.query_mode,
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
