"""Memory-efficient replay buffer for image-based Transition records.

Image observations are stored as uint8, and next-state images are reconstructed
from the next stored observation whenever transitions are contiguous. This keeps
the generic step-level Transition API while avoiding duplicate float32 image
storage in obs and next_obs.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import threading
from typing import Any

import numpy as np

from vla_rl.data.schema import RolloutBatch, Transition


def _is_image_key(key: str) -> bool:
    return key.startswith("image_")


def _as_obs_dict(obs: Any, *, name: str) -> dict[str, np.ndarray]:
    if not isinstance(obs, Mapping):
        raise TypeError(f"{name} must be a mapping for replay")
    return {str(key): np.asarray(value) for key, value in obs.items()}


def _image_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size and np.nanmin(arr) >= -1e-6 and np.nanmax(arr) <= 1.0 + 1e-6:
            arr = arr * 255.0
        return np.ascontiguousarray(np.rint(np.clip(arr, 0.0, 255.0)).astype(np.uint8))
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def _copy_obs_array(key: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if _is_image_key(key):
        return _image_to_uint8(arr)
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


class MemoryEfficientReplayBuffer:
    """Thread-safe FIFO replay specialized for image-based transitions."""

    def __init__(self, capacity: int = 100_000, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self._size = 0
        self._insert_count = 0
        self._latest_env_steps = 0
        self._obs_images: dict[str, np.ndarray] = {}
        self._obs_non_images: dict[str, np.ndarray] = {}
        self._next_obs_non_images: dict[str, np.ndarray] = {}
        self._explicit_next_images: list[dict[str, np.ndarray] | None] = [None] * self.capacity
        self._has_next_obs = np.zeros((self.capacity,), dtype=np.bool_)
        self._actions: np.ndarray | None = None
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.bool_)
        self._truncated = np.zeros((self.capacity,), dtype=np.bool_)
        self._discounts = np.ones((self.capacity,), dtype=np.float32)
        self._executed_steps = np.ones((self.capacity,), dtype=np.int32)
        self._env_steps: list[int | None] = [None] * self.capacity
        self._infos: list[dict[str, Any]] = [{} for _ in range(self.capacity)]
        self._initialized = False

    def __len__(self) -> int:
        with self._lock:
            return self._size

    @property
    def latest_env_steps(self) -> int:
        with self._lock:
            return int(self._latest_env_steps)

    def add(self, transition: Transition | dict[str, Any]) -> None:
        item = Transition.from_payload(transition)
        item.validate()
        obs = _as_obs_dict(item.obs, name="obs")
        next_obs = _as_obs_dict(item.next_obs, name="next_obs") if item.next_obs is not None else None
        with self._lock:
            if not self._initialized:
                self._initialize(obs, next_obs, item.action)
            idx = self._insert_count % self.capacity
            self._write_transition(idx, item, obs, next_obs)
            self._drop_previous_explicit_next_image_if_contiguous(idx, item)
            self._insert_count += 1
            self._size = min(self._size + 1, self.capacity)
            self._latest_env_steps = max(self._latest_env_steps, int(item.env_steps))

    def extend(self, transitions: Iterable[Transition | dict[str, Any]]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int) -> RolloutBatch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        with self._lock:
            if self._size == 0:
                raise ValueError("cannot sample from an empty replay buffer")
            offsets = self._rng.integers(0, self._size, size=int(batch_size))
            first_step_id = self._insert_count - self._size
            transitions = [self._read_transition_unlocked((first_step_id + int(offset)) % self.capacity) for offset in offsets]
        batch = RolloutBatch(transitions=transitions)
        batch.validate()
        return batch

    def iter_transitions(self) -> Iterator[Transition]:
        with self._lock:
            size = self._size
            first_step_id = self._insert_count - self._size
        for offset in range(size):
            idx = (first_step_id + offset) % self.capacity
            with self._lock:
                yield self._read_transition_unlocked(idx)

    def _initialize(self, obs: Mapping[str, np.ndarray], next_obs: Mapping[str, np.ndarray] | None, action: np.ndarray) -> None:
        image_keys = [key for key in obs if _is_image_key(key)]
        if not image_keys:
            raise ValueError("MemoryEfficientReplayBuffer expects at least one image_* observation key")
        obs_non_image_keys = [key for key in obs if not _is_image_key(key)]
        next_source = next_obs if next_obs is not None else obs
        for key in image_keys:
            image = _copy_obs_array(key, obs[key])
            self._obs_images[key] = np.empty((self.capacity, *image.shape), dtype=np.uint8)
        for key in obs_non_image_keys:
            value = _copy_obs_array(key, obs[key])
            self._obs_non_images[key] = np.empty((self.capacity, *value.shape), dtype=np.float32)
            if key not in next_source:
                raise KeyError(f"next_obs missing non-image key {key!r}")
            next_value = _copy_obs_array(key, next_source[key])
            self._next_obs_non_images[key] = np.empty((self.capacity, *next_value.shape), dtype=np.float32)
        self._actions = np.empty((self.capacity, *np.asarray(action).shape), dtype=np.float32)
        self._initialized = True

    def _write_transition(
        self,
        idx: int,
        transition: Transition,
        obs: Mapping[str, np.ndarray],
        next_obs: Mapping[str, np.ndarray] | None,
    ) -> None:
        assert self._actions is not None
        for key, storage in self._obs_images.items():
            storage[idx] = _copy_obs_array(key, obs[key])
        for key, storage in self._obs_non_images.items():
            storage[idx] = _copy_obs_array(key, obs[key])
        self._has_next_obs[idx] = next_obs is not None
        if next_obs is not None:
            self._explicit_next_images[idx] = {key: _copy_obs_array(key, next_obs[key]) for key in self._obs_images}
            for key, storage in self._next_obs_non_images.items():
                storage[idx] = _copy_obs_array(key, next_obs[key])
        else:
            self._explicit_next_images[idx] = None
            for storage in self._next_obs_non_images.values():
                storage[idx] = 0.0
        self._actions[idx] = np.asarray(transition.action, dtype=np.float32)
        self._rewards[idx] = np.float32(transition.reward)
        self._dones[idx] = bool(transition.done)
        self._truncated[idx] = bool(transition.truncated)
        self._discounts[idx] = np.float32(transition.discount)
        self._executed_steps[idx] = np.int32(transition.executed_steps)
        self._env_steps[idx] = int(transition.env_steps)
        self._infos[idx] = dict(transition.info or {})

    def _drop_previous_explicit_next_image_if_contiguous(self, idx: int, transition: Transition) -> None:
        if self._size == 0:
            return
        prev_idx = (self._insert_count - 1) % self.capacity
        if prev_idx == idx:
            return
        if self._dones[prev_idx] or self._truncated[prev_idx] or not self._has_next_obs[prev_idx]:
            return
        prev_env_steps = self._env_steps[prev_idx]
        current_env_steps = transition.env_steps
        expected = int(prev_env_steps) + int(self._executed_steps[prev_idx]) if prev_env_steps is not None else None
        if expected is not None and int(current_env_steps) != expected:
            return
        self._explicit_next_images[prev_idx] = None

    def _read_transition_unlocked(self, idx: int) -> Transition:
        assert self._actions is not None
        obs: dict[str, np.ndarray] = {}
        for key, storage in self._obs_images.items():
            obs[key] = storage[idx].copy()
        for key, storage in self._obs_non_images.items():
            obs[key] = storage[idx].copy()

        next_obs: dict[str, np.ndarray] | None = None
        if bool(self._has_next_obs[idx]):
            next_obs = {}
            next_idx = (idx + 1) % self.capacity
            explicit_next_images = self._explicit_next_images[idx]
            for key, storage in self._obs_images.items():
                if explicit_next_images is not None:
                    next_obs[key] = explicit_next_images[key].copy()
                else:
                    next_obs[key] = storage[next_idx].copy()
            for key, storage in self._next_obs_non_images.items():
                next_obs[key] = storage[idx].copy()

        transition = Transition(
            obs=obs,
            next_obs=next_obs,
            action=self._actions[idx].copy(),
            reward=float(self._rewards[idx]),
            done=bool(self._dones[idx]),
            truncated=bool(self._truncated[idx]),
            discount=float(self._discounts[idx]),
            executed_steps=int(self._executed_steps[idx]),
            env_steps=int(self._env_steps[idx] or 0),
            info=dict(self._infos[idx]),
        )
        transition.validate()
        return transition
