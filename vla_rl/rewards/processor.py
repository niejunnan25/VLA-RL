from __future__ import annotations

from dataclasses import dataclass, replace
import queue
import threading
import time
import uuid
from typing import Any, Callable

import numpy as np
from omegaconf import DictConfig

from vla_rl.data import Observation, Transition
from vla_rl.rewards.progress import (
    RemoteProgressClient,
    SPARSE_REWARD_TYPES,
    SUPPORTED_REWARD_TYPES,
    compute_potential_discount,
    compute_progress_reward,
)


MetricWriter = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class PendingRewardTransition:
    episode_id: int
    chunk_index: int
    start_observation: Observation
    end_observation: Observation
    transition: Transition
    env_reward: float
    executed_steps: int
    task: str | None


class BaseRewardProcessor:
    reward_type = "sparse"
    requires_single_transition_replay = False

    def submit(self, pending: PendingRewardTransition) -> int:
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:
        return {}

    def close(self, *, drain: bool = True) -> dict[str, Any]:
        del drain
        return self.stats()


class SparseRewardProcessor(BaseRewardProcessor):
    def __init__(self, data_store: Any) -> None:
        self.data_store = data_store
        self._submitted = 0
        self._committed = 0

    def submit(self, pending: PendingRewardTransition) -> int:
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
        }


class AsyncRemoteProgressRewardProcessor(BaseRewardProcessor):
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
        batch_size: int = 1,
        max_wait_ms: float = 0.0,
        initial_progress: str = "query_start",
        progress_event_writer: MetricWriter | None = None,
        reward_remote_url: str | None = None,
    ) -> None:
        self.data_store = data_store
        self.client = client
        self.reward_type = str(reward_type)
        self.gamma = float(gamma)
        self.scale = float(scale)
        self.image_keys = tuple(str(key) for key in image_keys)
        self.batch_size = max(1, int(batch_size))
        self.max_wait_sec = max(0.0, float(max_wait_ms) / 1000.0)
        self.initial_progress = str(initial_progress)
        self.progress_event_writer = progress_event_writer
        self.reward_remote_url = None if reward_remote_url is None else str(reward_remote_url)
        self.session_id = uuid.uuid4().hex
        if self.initial_progress not in {"zero", "query_start"}:
            raise ValueError("reward.initial_progress must be zero or query_start")
        self._queue: queue.Queue[PendingRewardTransition | None] = queue.Queue(maxsize=max(1, int(max_pending_chunks)))
        self._lock = threading.Lock()
        self._closed = False
        self._failed: Exception | None = None
        self._stats: dict[str, Any] = {
            "reward_type": self.reward_type,
            "submitted": 0,
            "committed": 0,
            "pending": 0,
            "failed": 0,
            "last_progress": None,
            "last_reward": None,
            "last_latency_sec": 0.0,
            "last_batch_size": 0,
            "batch_size": int(self.batch_size),
            "max_wait_ms": float(max_wait_ms),
            "progress_event_write_failed": 0,
        }
        self._episodes: dict[int, dict[str, Any]] = {}
        self._lookahead: PendingRewardTransition | None = None
        self._thread = threading.Thread(target=self._worker_main, name="reward-worker", daemon=True)
        self._thread.start()

    def submit(self, pending: PendingRewardTransition) -> int:
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

    def close(self, *, drain: bool = True) -> dict[str, Any]:
        if not self._closed:
            if drain:
                self._queue.join()
            self._closed = True
            self._queue.put(None)
            self._thread.join(timeout=30.0)
            self.client.close()
        self._raise_if_failed()
        return self.stats()

    def _worker_main(self) -> None:
        while True:
            pending = self._pop_next_pending()
            if pending is None:
                self._queue.task_done()
                break
            with self._lock:
                failed = self._failed
            if failed is not None:
                self._queue.task_done()
                continue
            batch, stop_after_batch = self._collect_batch(pending)
            try:
                self._process_batch(batch)
            except Exception as exc:
                self._handle_batch_error(batch, exc)
            finally:
                with self._lock:
                    self._stats["pending"] = self._queue.qsize()
                for _item in batch:
                    self._queue.task_done()
                if stop_after_batch:
                    break

    def _pop_next_pending(self) -> PendingRewardTransition | None:
        if self._lookahead is not None:
            pending = self._lookahead
            self._lookahead = None
            return pending
        return self._queue.get()

    def _collect_batch(self, first: PendingRewardTransition) -> tuple[list[PendingRewardTransition], bool]:
        batch = [first]
        stop_after_batch = False
        if self.batch_size <= 1 or first.transition.done or first.transition.truncated:
            return batch, stop_after_batch

        deadline = time.perf_counter() + self.max_wait_sec
        while len(batch) < self.batch_size:
            timeout = 0.0
            if self.max_wait_sec > 0.0:
                timeout = max(0.0, deadline - time.perf_counter())
                if timeout <= 0.0:
                    break
            try:
                if self.max_wait_sec > 0.0:
                    pending = self._queue.get(timeout=timeout)
                else:
                    pending = self._queue.get_nowait()
            except queue.Empty:
                break

            if pending is None:
                self._queue.task_done()
                stop_after_batch = True
                break
            if pending.episode_id != first.episode_id:
                self._lookahead = pending
                break
            batch.append(pending)
            if pending.transition.done or pending.transition.truncated:
                break
        return batch, stop_after_batch

    def _process_batch(self, batch: list[PendingRewardTransition]) -> None:
        if not batch:
            return
        start = time.perf_counter()
        first = batch[0]
        episode = self._episodes.get(first.episode_id)
        if episode is None:
            start_payload = observation_to_reward_payload(first.start_observation, image_keys=self.image_keys)
            episode = {
                "start_boundary_payload": start_payload,
                "start_boundary_sent": False,
                "last_progress": None,
                "last_progress_index": 0,
                "last_boundary_payload": start_payload,
                "next_index": 1,
            }
            self._episodes[first.episode_id] = episode
        end_payloads = [
            observation_to_reward_payload(pending.end_observation, image_keys=self.image_keys)
            for pending in batch
        ]
        first_index = int(episode.get("next_index", 1))
        boundary_indices = list(range(first_index, first_index + len(batch)))
        episode["next_index"] = first_index + len(batch)
        episode["pending_batch_payloads"] = list(zip(boundary_indices, end_payloads))

        previous_index = int(episode.get("last_progress_index", max(0, first_index - 1)))
        start_payload = episode.get("start_boundary_payload")
        previous_payload = episode.get("last_boundary_payload")
        needs_start_query = episode["last_progress"] is None and self.initial_progress == "query_start"
        needs_previous_recovery = (
            episode["last_progress"] is None
            and previous_payload is not None
            and previous_index > 0
            and previous_index != first_index
        )

        payload_by_index: dict[int, dict[str, Any]] = {}

        def add_payload(index: int, payload: dict[str, Any] | None) -> None:
            if payload is not None and int(index) not in payload_by_index:
                payload_by_index[int(index)] = payload

        if start_payload is not None and (not bool(episode.get("start_boundary_sent", False)) or needs_start_query):
            add_payload(0, start_payload)
        if needs_previous_recovery:
            add_payload(previous_index, previous_payload)
        for boundary_index, payload in zip(boundary_indices, end_payloads, strict=True):
            add_payload(boundary_index, payload)

        trajectory_indices = sorted(payload_by_index)
        trajectory_payload = [payload_by_index[index] for index in trajectory_indices]

        absolute_query_indices: list[int] = []
        if needs_previous_recovery:
            absolute_query_indices.append(previous_index)
        elif needs_start_query:
            absolute_query_indices.append(0)
        absolute_query_indices.extend(boundary_indices)

        local_index = {int(index): pos for pos, index in enumerate(trajectory_indices)}
        query_indices = [local_index[int(index)] for index in absolute_query_indices if int(index) in local_index]
        if len(query_indices) != len(absolute_query_indices):
            missing = [int(index) for index in absolute_query_indices if int(index) not in local_index]
            raise RuntimeError(f"reward request missing payloads for query indices: {missing}")

        request = {
            "episode_id": int(first.episode_id),
            "chunk_index": int(first.chunk_index),
            "task": first.task,
            "trajectory": trajectory_payload,
            "trajectory_indices": trajectory_indices,
            "query_indices": query_indices,
            "absolute_query_indices": absolute_query_indices,
            "image_keys": self.image_keys,
            "session_id": self.session_id,
            "metadata": {
                "session_id": self.session_id,
                "batch_size": int(len(batch)),
                "env_reward": float(batch[-1].env_reward),
                "executed_steps": int(sum(int(pending.executed_steps) for pending in batch)),
                "done": bool(batch[-1].transition.done),
                "truncated": bool(batch[-1].transition.truncated),
                "env_steps": int(batch[-1].transition.env_steps),
                "trajectory_indices": trajectory_indices,
                "absolute_query_indices": absolute_query_indices,
            },
        }
        progress_values = self.client.predict_progress(request, expected_count=len(query_indices))

        if needs_start_query or (
            start_payload is not None
            and first_index != 0
            and episode["last_progress"] is None
            and previous_payload is not None
            and previous_index > 0
            and previous_index != first_index
        ):
            previous_progress = float(progress_values[0])
            end_progress_values = [float(value) for value in progress_values[1:]]
        else:
            previous_progress = episode["last_progress"]
            end_progress_values = [float(value) for value in progress_values]
        if len(end_progress_values) != len(batch):
            raise RuntimeError(f"expected {len(batch)} end progress values, got {len(end_progress_values)}")

        latency = time.perf_counter() - start
        previous_boundary_index = previous_index
        for index, (pending, progress, boundary_index) in enumerate(
            zip(batch, end_progress_values, boundary_indices, strict=True)
        ):
            terminal = bool(pending.transition.done or pending.transition.truncated)
            potential_discount = compute_potential_discount(
                gamma=self.gamma,
                executed_steps=int(pending.executed_steps),
                discount=float(pending.transition.discount),
                terminal=terminal,
            )
            reward = compute_progress_reward(
                self.reward_type,
                env_reward=float(pending.env_reward),
                progress=float(progress),
                previous_progress=previous_progress,
                gamma=self.gamma,
                executed_steps=int(pending.executed_steps),
                scale=self.scale,
                discount=float(pending.transition.discount),
                terminal=terminal,
            )
            if index == 0 and (needs_start_query or previous_boundary_index != boundary_index - 1):
                event_progress_values = [] if previous_progress is None else [float(previous_progress), float(progress)]
            else:
                event_progress_values = [float(progress)]
            self._commit(
                pending,
                reward,
                progress=float(progress),
                previous_progress=previous_progress,
                latency=latency,
                batch_size=len(batch),
                boundary_index=int(boundary_index),
                previous_boundary_index=previous_boundary_index,
                trajectory_indices=trajectory_indices,
                query_indices=query_indices,
                absolute_query_indices=absolute_query_indices,
                progress_values=event_progress_values,
                potential_discount=potential_discount,
            )
            previous_progress = float(progress)
            previous_boundary_index = int(boundary_index)

        episode["last_progress"] = float(end_progress_values[-1])
        episode["last_progress_index"] = int(boundary_indices[-1])
        episode["last_boundary_payload"] = end_payloads[-1]
        episode["start_boundary_sent"] = True
        episode.pop("pending_batch_payloads", None)
        if batch[-1].transition.done or batch[-1].transition.truncated:
            self._episodes.pop(first.episode_id, None)

    def _handle_batch_error(self, batch: list[PendingRewardTransition], exc: Exception) -> None:
        with self._lock:
            self._stats["failed"] += len(batch)
            if self._failed is None:
                self._failed = exc

    def _commit(
        self,
        pending: PendingRewardTransition,
        reward: float,
        *,
        progress: float | None,
        previous_progress: float | None,
        latency: float,
        batch_size: int = 1,
        boundary_index: int | None = None,
        previous_boundary_index: int | None = None,
        trajectory_indices: list[int] | None = None,
        query_indices: list[int] | None = None,
        absolute_query_indices: list[int] | None = None,
        progress_values: list[float] | None = None,
        potential_discount: float | None = None,
    ) -> None:
        info = {
            **dict(pending.transition.info),
            "reward_type": self.reward_type,
            "env_reward": float(pending.env_reward),
            "reward_model_progress": None if progress is None else float(progress),
            "reward_model_previous_progress": None if previous_progress is None else float(previous_progress),
            "reward_model_latency_sec": float(latency),
            "reward_model_batch_size": int(batch_size),
            "reward_potential_discount": None if potential_discount is None else float(potential_discount),
        }
        transition = replace(pending.transition, reward=float(reward), info=info)
        self._write_progress_event(
            pending,
            reward=reward,
            progress=progress,
            previous_progress=previous_progress,
            latency=latency,
            batch_size=batch_size,
            boundary_index=boundary_index,
            previous_boundary_index=previous_boundary_index,
            trajectory_indices=trajectory_indices,
            query_indices=query_indices,
            absolute_query_indices=absolute_query_indices,
            progress_values=progress_values,
            potential_discount=potential_discount,
        )
        self.data_store.insert(transition.to_payload())
        with self._lock:
            self._stats["committed"] += 1
            self._stats["last_progress"] = None if progress is None else float(progress)
            self._stats["last_reward"] = float(reward)
            self._stats["last_latency_sec"] = float(latency)
            self._stats["last_batch_size"] = int(batch_size)

    def _write_progress_event(
        self,
        pending: PendingRewardTransition,
        *,
        reward: float,
        progress: float | None,
        previous_progress: float | None,
        latency: float,
        batch_size: int,
        boundary_index: int | None,
        previous_boundary_index: int | None,
        trajectory_indices: list[int] | None,
        query_indices: list[int] | None,
        absolute_query_indices: list[int] | None,
        progress_values: list[float] | None,
        potential_discount: float | None,
    ) -> None:
        if self.progress_event_writer is None:
            return
        transition_info = dict(pending.transition.info)
        event = {
            "role": "reward_progress",
            "source": "remote_progress",
            "status": "ok",
            "episode_id": int(pending.episode_id),
            "chunk_index": int(pending.chunk_index),
            "task": pending.task,
            "boundary_index": None if boundary_index is None else int(boundary_index),
            "previous_boundary_index": None
            if previous_boundary_index is None
            else int(previous_boundary_index),
            "trajectory_indices": [] if trajectory_indices is None else [int(idx) for idx in trajectory_indices],
            "query_indices": [] if query_indices is None else [int(idx) for idx in query_indices],
            "absolute_query_indices": []
            if absolute_query_indices is None
            else [int(idx) for idx in absolute_query_indices],
            "progress_values": [] if progress_values is None else [float(value) for value in progress_values],
            "progress": None if progress is None else float(progress),
            "previous_progress": None if previous_progress is None else float(previous_progress),
            "env_reward": float(pending.env_reward),
            "computed_reward": float(reward),
            "reward_type": self.reward_type,
            "scale": float(self.scale),
            "gamma": float(self.gamma),
            "discount": float(pending.transition.discount),
            "potential_discount": None if potential_discount is None else float(potential_discount),
            "executed_steps": int(pending.executed_steps),
            "env_steps": int(pending.transition.env_steps),
            "chunk_start_env_steps": int(
                transition_info.get("chunk_start_env_steps", pending.transition.env_steps)
            ),
            "done": bool(pending.transition.done),
            "truncated": bool(pending.transition.truncated),
            "latency_sec": float(latency),
            "batch_size": int(batch_size),
            "reward_remote_url": self.reward_remote_url,
        }
        try:
            self.progress_event_writer(event)
        except Exception:
            with self._lock:
                self._stats["progress_event_write_failed"] += 1

    def _raise_if_failed(self) -> None:
        with self._lock:
            failed = self._failed
        if failed is not None:
            raise RuntimeError(f"reward worker failed: {failed!r}") from failed


def build_reward_processor(
    cfg: DictConfig,
    *,
    data_store: Any,
    gamma: float,
    metric_writer: MetricWriter | None = None,
    progress_event_writer: MetricWriter | None = None,
) -> BaseRewardProcessor:
    reward_cfg = cfg.get("reward", None)
    if reward_cfg is None:
        return SparseRewardProcessor(data_store)
    if "terminal_potential" in reward_cfg:
        raise ValueError(
            "reward.terminal_potential is no longer supported; terminal potential is always zeroed"
        )
    reward_type = str(reward_cfg.get("type", "sparse"))
    if reward_type not in SUPPORTED_REWARD_TYPES:
        raise ValueError(f"unsupported reward.type: {reward_type}")
    source = str(reward_cfg.get("source", "env" if reward_type in SPARSE_REWARD_TYPES else "remote_progress"))
    if reward_type in SPARSE_REWARD_TYPES:
        return SparseRewardProcessor(data_store)
    if source in {"env", "sparse", "none"}:
        raise ValueError(f"reward.type={reward_type} requires reward.source=remote_progress, got {source}")
    if source != "remote_progress":
        raise ValueError(f"unsupported reward.source: {source}")
    if "on_error" in reward_cfg:
        raise ValueError("reward.on_error is not supported; reward model failures always raise")

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
    if not image_keys:
        raise ValueError("reward.trajectory.image_keys is required when reward.source=remote_progress")
    async_cfg = reward_cfg.get("async", {})
    return AsyncRemoteProgressRewardProcessor(
        data_store=data_store,
        client=client,
        reward_type=reward_type,
        gamma=float(gamma),
        scale=float(reward_cfg.get("scale", 1.0)),
        image_keys=image_keys,
        max_pending_chunks=int(async_cfg.get("max_pending_chunks", 64)),
        batch_size=int(async_cfg.get("batch_size", async_cfg.get("max_batch_chunks", 1))),
        max_wait_ms=float(async_cfg.get("max_wait_ms", 0.0)),
        initial_progress=str(reward_cfg.get("initial_progress", "query_start")),
        progress_event_writer=progress_event_writer,
        reward_remote_url=str(url),
    )


def observation_to_reward_payload(obs: Observation, *, image_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    selected_keys = image_keys or tuple(obs.images.keys())
    missing = [key for key in selected_keys if key not in obs.images]
    if missing:
        raise KeyError(f"observation missing reward image keys: {missing}")
    images = {
        key: np.asarray(obs.images[key]).copy()
        for key in selected_keys
    }
    return {
        "images": images,
        "proprio": None if obs.proprio is None else np.asarray(obs.proprio, dtype=np.float32).copy(),
        "task": obs.task,
    }
