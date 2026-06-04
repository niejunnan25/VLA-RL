from __future__ import annotations

from collections import deque
import threading
from typing import Iterable

import numpy as np

from vla_rl.data.schema import RolloutBatch, Transition


class ReplayBuffer:
    """Thread-safe FIFO replay for learner-facing Transition records."""

    def __init__(self, capacity: int = 100_000, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self._items: deque[Transition] = deque(maxlen=self.capacity)
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self._latest_env_steps = 0

    def add(self, transition: Transition | dict) -> None:
        item = Transition.from_payload(transition)
        item.validate()
        with self._lock:
            self._items.append(item)
            self._latest_env_steps = max(self._latest_env_steps, int(item.env_steps))

    def extend(self, transitions: Iterable[Transition | dict]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int) -> RolloutBatch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        with self._lock:
            if not self._items:
                raise ValueError("cannot sample from an empty replay buffer")
            indices = self._rng.integers(0, len(self._items), size=int(batch_size))
            transitions = [self._items[int(index)] for index in indices]
        batch = RolloutBatch(transitions=transitions)
        batch.validate()
        return batch

    @property
    def latest_env_steps(self) -> int:
        with self._lock:
            return int(self._latest_env_steps)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
