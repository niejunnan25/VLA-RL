from __future__ import annotations

from dataclasses import dataclass, replace
import queue
import threading
import time
from typing import Any, Callable

import numpy as np
from omegaconf import DictConfig

from vla_rl.data import Observation, Transition
from vla_rl.rewards import RemoteProgressClient, compute_progress_reward
from vla_rl.rewards.progress import SPARSE_REWARD_TYPES, SUPPORTED_REWARD_TYPES


MetricWriter = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class PendingRLTRewardTransition:
    episode_id: int
    chunk_index: int
    start_observation: Observation
    end_observation: Observation
    transition: Transition
    env_reward: float
    executed_steps: int
    task: str | None


class BaseRLTRewardProcessor:
    reward_type = "sparse"
    requires_single_transition_replay = False

    def submit(self, pending: PendingRLTRewardTransition) -> int:
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:
        return {}

    def close(self, *, drain: bool = True, raise_on_error: bool = True) -> dict[str, Any]:
        del drain, raise_on_error
        return self.stats()


class SparseRLTRewardProcessor(BaseRLTRewardProcessor):
    def __init__(self, data_store: Any) -> None:
        self.data_store = data_store
        self._submitted = 0
        self._committed = 0

    def submit(self, pending: PendingRLTRewardTransition) -> int:
        transition = replace(
            pending.transition,
            reward=float(pending.env_reward),
            info={
                **dict(pending.transition.info),
                "reward_type": "sparse",
                "env_reward": float(pending.env_reward),
            },
        )
        self.data_store.insert(transition.to_payload())
        self._submitted += 1
        self._committed += 1
        return 1

    def stats(self) -> dict[str, Any]:
        return {
            "reward_type": "sparse",
            "submitted": int(self._submitted),
            "committed": int(self._committed),
            "pending": 0,
            "failed": 0,
            "fallback_sparse": 0,
        }


class AsyncRemoteProgressRLTRewardProcessor(BaseRLTRewardProcessor):
    requires_single_transition_replay = True

    def __init__(
        self,
        *,
        data_store: Any,
        client: RemoteProgressClient,
        reward_type: str,
        gamma: float,
        scale: float = 1.0,
        image_keys: tuple[str, ...] = (),
        max_pending_chunks: int = 64,
        initial_progress: str = "query_start",
        on_error: str = "fallback_sparse",
        metric_writer: MetricWriter | None = None,
    ) -> None:
        self.data_store = data_store
        self.client = client
        self.reward_type = str(reward_type)
        self.gamma = float(gamma)
        self.scale = float(scale)
        self.image_keys = tuple(str(key) for key in image_keys)
        self.initial_progress = str(initial_progress)
        self.on_error = str(on_error)
        self.metric_writer = metric_writer
        if self.initial_progress not in {"zero", "query_start"}:
            raise ValueError("reward.initial_progress must be zero or query_start")
        if self.on_error not in {"fallback_sparse", "raise"}:
            raise ValueError("reward.on_error must be fallback_sparse or raise")
        self._queue: queue.Queue[PendingRLTRewardTransition | None] = queue.Queue(maxsize=max(1, int(max_pending_chunks)))
        self._lock = threading.Lock()
        self._closed = False
        self._failed: Exception | None = None
        self._stats: dict[str, Any] = {
            "reward_type": self.reward_type,
            "submitted": 0,
            "committed": 0,
            "pending": 0,
            "failed": 0,
            "fallback_sparse": 0,
            "last_progress": None,
            "last_reward": None,
            "last_latency_sec": 0.0,
        }
        self._episodes: dict[int, dict[str, Any]] = {}
        self._thread = threading.Thread(target=self._worker_main, name="rlt-reward-worker", daemon=True)
        self._thread.start()

    def submit(self, pending: PendingRLTRewardTransition) -> int:
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("reward processor is closed")
        self._queue.put(pending)
        with self._lock:
            self._stats["submitted"] += 1
            self._stats["pending"] = self._queue.qsize()
        return 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            stats = dict(self._stats)
        stats["pending"] = self._queue.qsize()
        return stats

    def close(self, *, drain: bool = True, raise_on_error: bool = True) -> dict[str, Any]:
        if not self._closed:
            if drain:
                self._queue.join()
            self._closed = True
            self._queue.put(None)
            self._thread.join(timeout=30.0)
            self.client.close()
        if raise_on_error:
            self._raise_if_failed()
        return self.stats()

    def _worker_main(self) -> None:
        while True:
            pending = self._queue.get()
            if pending is None:
                self._queue.task_done()
                break
            try:
                self._process_one(pending)
            except Exception as exc:  # pragma: no cover - covered through fallback tests where possible.
                self._handle_error(pending, exc)
            finally:
                with self._lock:
                    self._stats["pending"] = self._queue.qsize()
                self._queue.task_done()

    def _process_one(self, pending: PendingRLTRewardTransition) -> None:
        start = time.perf_counter()
        episode = self._episodes.get(pending.episode_id)
        if episode is None:
            start_payload = observation_to_reward_payload(pending.start_observation, image_keys=self.image_keys)
            episode = {
                "last_progress": None,
                "last_progress_index": 0,
                "last_boundary_payload": start_payload,
                "next_index": 1,
            }
            self._episodes[pending.episode_id] = episode
        end_payload = observation_to_reward_payload(pending.end_observation, image_keys=self.image_keys)
        current_index = int(episode.get("next_index", 1))
        episode["next_index"] = current_index + 1
        episode["pending_boundary_payload"] = end_payload
        episode["pending_boundary_index"] = current_index

        trajectory_payload: list[dict[str, Any]]
        trajectory_indices: list[int]
        query_indices: list[int]
        absolute_query_indices: list[int]
        previous_index = int(episode.get("last_progress_index", max(0, current_index - 1)))
        needs_previous_query = episode["last_progress"] is None and (
            self.initial_progress == "query_start" or previous_index > 0
        )
        if needs_previous_query:
            previous_payload = episode.get("last_boundary_payload")
            if previous_payload is not None and previous_index != current_index:
                trajectory_payload = [previous_payload, end_payload]
                trajectory_indices = [previous_index, current_index]
                query_indices = [0, 1]
                absolute_query_indices = [previous_index, current_index]
            else:
                trajectory_payload = [end_payload]
                trajectory_indices = [current_index]
                query_indices = [0]
                absolute_query_indices = [current_index]
        else:
            trajectory_payload = [end_payload]
            trajectory_indices = [current_index]
            query_indices = [0]
            absolute_query_indices = [current_index]

        request = {
            "episode_id": int(pending.episode_id),
            "chunk_index": int(pending.chunk_index),
            "task": pending.task,
            "trajectory": trajectory_payload,
            "trajectory_indices": trajectory_indices,
            "query_indices": query_indices,
            "absolute_query_indices": absolute_query_indices,
            "image_keys": self.image_keys,
            "metadata": {
                "env_reward": float(pending.env_reward),
                "executed_steps": int(pending.executed_steps),
                "done": bool(pending.transition.done),
                "truncated": bool(pending.transition.truncated),
                "env_steps": int(pending.transition.env_steps),
                "trajectory_indices": trajectory_indices,
                "absolute_query_indices": absolute_query_indices,
            },
        }
        progress_values = self.client.predict_progress(request, expected_count=len(query_indices))
        if len(progress_values) == len(query_indices) and len(query_indices) > 1:
            previous_progress = float(progress_values[0])
            progress = float(progress_values[-1])
        else:
            previous_progress = episode["last_progress"]
            progress = float(progress_values[-1])

        reward = compute_progress_reward(
            self.reward_type,
            env_reward=float(pending.env_reward),
            progress=progress,
            previous_progress=previous_progress,
            gamma=self.gamma,
            executed_steps=int(pending.executed_steps),
            scale=self.scale,
            discount=float(pending.transition.discount),
        )
        episode["last_progress"] = progress
        episode["last_progress_index"] = current_index
        episode["last_boundary_payload"] = end_payload
        episode.pop("pending_boundary_payload", None)
        episode.pop("pending_boundary_index", None)
        self._commit(pending, reward, progress=progress, previous_progress=previous_progress, latency=time.perf_counter() - start)
        if pending.transition.done or pending.transition.truncated:
            self._episodes.pop(pending.episode_id, None)

    def _commit(
        self,
        pending: PendingRLTRewardTransition,
        reward: float,
        *,
        progress: float | None,
        previous_progress: float | None,
        latency: float,
        error: str | None = None,
    ) -> None:
        info = {
            **dict(pending.transition.info),
            "reward_type": self.reward_type,
            "env_reward": float(pending.env_reward),
            "reward_model_progress": None if progress is None else float(progress),
            "reward_model_previous_progress": None if previous_progress is None else float(previous_progress),
            "reward_model_latency_sec": float(latency),
        }
        if error is not None:
            info["reward_model_error"] = error
        transition = replace(pending.transition, reward=float(reward), info=info)
        self.data_store.insert(transition.to_payload())
        with self._lock:
            self._stats["committed"] += 1
            self._stats["last_progress"] = None if progress is None else float(progress)
            self._stats["last_reward"] = float(reward)
            self._stats["last_latency_sec"] = float(latency)

    def _handle_error(self, pending: PendingRLTRewardTransition, exc: Exception) -> None:
        with self._lock:
            self._stats["failed"] += 1
        if self.on_error == "fallback_sparse":
            with self._lock:
                self._stats["fallback_sparse"] += 1
            episode = self._episodes.get(pending.episode_id)
            if pending.transition.done or pending.transition.truncated:
                self._episodes.pop(pending.episode_id, None)
            elif episode is not None:
                pending_payload = episode.pop("pending_boundary_payload", None)
                pending_index = episode.pop("pending_boundary_index", None)
                if pending_payload is not None and pending_index is not None:
                    episode["last_boundary_payload"] = pending_payload
                    episode["last_progress_index"] = int(pending_index)
                episode["last_progress"] = None
            self._commit(
                pending,
                float(pending.env_reward),
                progress=None,
                previous_progress=None,
                latency=0.0,
                error=repr(exc),
            )
            return
        with self._lock:
            self._failed = exc

    def _raise_if_failed(self) -> None:
        with self._lock:
            failed = self._failed
        if failed is not None:
            raise RuntimeError(f"reward worker failed: {failed!r}") from failed


def build_rlt_reward_processor(
    cfg: DictConfig,
    *,
    data_store: Any,
    gamma: float,
    metric_writer: MetricWriter | None = None,
) -> BaseRLTRewardProcessor:
    reward_cfg = cfg.get("reward", None)
    if reward_cfg is None:
        return SparseRLTRewardProcessor(data_store)
    reward_type = str(reward_cfg.get("type", "sparse"))
    if reward_type not in SUPPORTED_REWARD_TYPES:
        raise ValueError(f"unsupported reward.type: {reward_type}")
    source = str(reward_cfg.get("source", "env" if reward_type in SPARSE_REWARD_TYPES else "remote_progress"))
    if reward_type in SPARSE_REWARD_TYPES:
        return SparseRLTRewardProcessor(data_store)
    if source in {"env", "sparse", "none"}:
        raise ValueError(f"reward.type={reward_type} requires reward.source=remote_progress, got {source}")
    if source != "remote_progress":
        raise ValueError(f"unsupported reward.source: {source}")

    remote = reward_cfg.get("remote", {})
    url = remote.get("url", None)
    if not url:
        raise ValueError("reward.remote.url is required when reward.source=remote_progress")
    client = RemoteProgressClient(
        str(url),
        method=str(remote.get("method", "predict_progress")),
        timeout=float(remote.get("timeout", 120.0)),
        retries=int(remote.get("retries", 1)),
        retry_sleep=float(remote.get("retry_sleep", 0.5)),
    )
    trajectory_cfg = reward_cfg.get("trajectory", {})
    image_keys = tuple(str(key) for key in trajectory_cfg.get("image_keys", ()))
    async_cfg = reward_cfg.get("async", {})
    return AsyncRemoteProgressRLTRewardProcessor(
        data_store=data_store,
        client=client,
        reward_type=reward_type,
        gamma=float(gamma),
        scale=float(reward_cfg.get("scale", 1.0)),
        image_keys=image_keys,
        max_pending_chunks=int(async_cfg.get("max_pending_chunks", 64)),
        initial_progress=str(reward_cfg.get("initial_progress", "query_start")),
        on_error=str(reward_cfg.get("on_error", "fallback_sparse")),
        metric_writer=metric_writer,
    )


def observation_to_reward_payload(obs: Observation, *, image_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    selected_keys = image_keys or tuple(obs.images.keys())
    images = {
        key: np.asarray(obs.images[key]).copy()
        for key in selected_keys
        if key in obs.images
    }
    return {
        "images": images,
        "proprio": None if obs.proprio is None else np.asarray(obs.proprio, dtype=np.float32).copy(),
        "task": obs.task,
    }
