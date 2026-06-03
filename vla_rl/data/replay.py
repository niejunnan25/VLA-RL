from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from vla_rl.data.schema import RolloutBatch, Transition


class ReplayBuffer:
    """Small in-memory FIFO replay buffer for local smoke tests."""

    def __init__(self, capacity: int = 100_000, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self._items: deque[Transition] = deque(maxlen=self.capacity)
        self._rng = np.random.default_rng(seed)

    def add(self, transition: Transition) -> None:
        transition.validate()
        self._items.append(transition)

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int) -> RolloutBatch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not self._items:
            raise ValueError("cannot sample from an empty replay buffer")
        indices = self._rng.integers(0, len(self._items), size=int(batch_size))
        batch = RolloutBatch(transitions=[self._items[int(index)] for index in indices])
        batch.validate()
        return batch

    def __len__(self) -> int:
        return len(self._items)
