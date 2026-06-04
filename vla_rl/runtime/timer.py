from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from typing import Iterator


class Timer:
    """Small SERL-style wall-clock timer for rollout and learner diagnostics."""

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def tick(self, key: str) -> None:
        if key in self._starts:
            raise RuntimeError(f"timer key already active: {key}")
        self._starts[key] = time.perf_counter()

    def tock(self, key: str) -> float:
        start = self._starts.pop(key)
        elapsed = time.perf_counter() - start
        self._totals[key] += elapsed
        self._counts[key] += 1
        return elapsed

    @contextlib.contextmanager
    def context(self, key: str) -> Iterator[None]:
        self.tick(key)
        try:
            yield
        finally:
            self.tock(key)

    def get_average_times(
        self,
        *,
        reset: bool = True,
        prefix: str = "",
        suffix: str = "",
    ) -> dict[str, float]:
        metrics = {
            f"{prefix}{key}{suffix}": total / self._counts[key]
            for key, total in self._totals.items()
            if self._counts[key] > 0
        }
        if reset:
            self.reset()
        return metrics

    def get_total_times(
        self,
        *,
        reset: bool = True,
        prefix: str = "",
        suffix: str = "",
    ) -> dict[str, float]:
        metrics = {f"{prefix}{key}{suffix}": total for key, total in self._totals.items()}
        if reset:
            self.reset()
        return metrics

    def reset(self) -> None:
        self._starts.clear()
        self._totals.clear()
        self._counts.clear()
