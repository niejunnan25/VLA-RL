from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
from typing import Any, Iterable

import numpy as np

from vla_rl.data.schema import Observation, RolloutBatch, Transition


@dataclass(slots=True)
class CompactTransition:
    agent_obs: dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    done: bool
    discount: float
    next_agent_obs: dict[str, np.ndarray] | None = None
    truncated: bool = False
    executed_steps: int = 1
    env_steps: int = 0
    info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _validate_agent_obs("agent_obs", self.agent_obs)
        if self.next_agent_obs is not None:
            _validate_agent_obs("next_agent_obs", self.next_agent_obs)
        action = np.asarray(self.action)
        if action.ndim != 1:
            raise ValueError(f"compact action must be flat, got shape={action.shape}")
        if not np.issubdtype(action.dtype, np.floating):
            raise ValueError(f"compact action must be floating, got {action.dtype}")
        if not np.isfinite(float(self.reward)):
            raise ValueError(f"reward must be finite, got {self.reward}")
        if not 0.0 <= float(self.discount) <= 1.0:
            raise ValueError(f"discount must be in [0, 1], got {self.discount}")
        if int(self.executed_steps) <= 0:
            raise ValueError(f"executed_steps must be positive, got {self.executed_steps}")

    def to_transition(self) -> Transition:
        empty_obs = Observation()
        return Transition(
            obs=empty_obs,
            action=np.asarray(self.action, dtype=np.float32),
            reward=float(self.reward),
            next_obs=empty_obs,
            done=bool(self.done),
            truncated=bool(self.truncated),
            discount=float(self.discount),
            agent_obs=_copy_agent_obs(self.agent_obs),
            next_agent_obs=None if self.next_agent_obs is None else _copy_agent_obs(self.next_agent_obs),
            info={**self.info, "executed_steps": int(self.executed_steps), "env_steps": int(self.env_steps)},
        )

    @classmethod
    def from_payload(cls, payload: "CompactTransition | dict[str, Any]") -> "CompactTransition":
        if isinstance(payload, CompactTransition):
            return payload
        return cls(
            agent_obs=_copy_agent_obs(payload["agent_obs"]),
            next_agent_obs=None
            if payload.get("next_agent_obs") is None
            else _copy_agent_obs(payload.get("next_agent_obs", {})),
            action=np.asarray(payload["action"], dtype=np.float32).reshape(-1),
            reward=float(payload["reward"]),
            done=bool(payload["done"]),
            truncated=bool(payload.get("truncated", False)),
            discount=float(payload["discount"]),
            executed_steps=int(payload.get("executed_steps", 1)),
            env_steps=int(payload.get("env_steps", 0)),
            info=dict(payload.get("info", {})),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "agent_obs": _copy_agent_obs(self.agent_obs),
            "next_agent_obs": None if self.next_agent_obs is None else _copy_agent_obs(self.next_agent_obs),
            "action": np.asarray(self.action, dtype=np.float32),
            "reward": float(self.reward),
            "done": bool(self.done),
            "truncated": bool(self.truncated),
            "discount": float(self.discount),
            "executed_steps": int(self.executed_steps),
            "env_steps": int(self.env_steps),
            "info": dict(self.info),
        }


class CompactReplayBuffer:
    """Thread-safe FIFO replay for compact actor-to-learner transitions."""

    def __init__(self, capacity: int = 100_000, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self._items: deque[CompactTransition] = deque(maxlen=self.capacity)
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self._latest_env_steps = 0

    def add(self, transition: CompactTransition | dict[str, Any]) -> None:
        compact = CompactTransition.from_payload(transition)
        with self._lock:
            self._items.append(compact)
            self._latest_env_steps = max(self._latest_env_steps, int(compact.env_steps))

    def extend(self, transitions: Iterable[CompactTransition | dict[str, Any]]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int) -> RolloutBatch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        with self._lock:
            if not self._items:
                raise ValueError("cannot sample from an empty replay buffer")
            indices = self._rng.integers(0, len(self._items), size=int(batch_size))
            compact_batch = [self._items[int(index)] for index in indices]
        return RolloutBatch(transitions=[transition.to_transition() for transition in compact_batch])

    @property
    def latest_env_steps(self) -> int:
        with self._lock:
            return int(self._latest_env_steps)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def _copy_agent_obs(value: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {str(key): np.asarray(array, dtype=np.float32).copy() for key, array in value.items()}


def _validate_agent_obs(name: str, value: dict[str, np.ndarray]) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict, got {type(value).__name__}")
    for key, array in value.items():
        arr = np.asarray(array)
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(f"{name}[{key}] must be floating, got {arr.dtype}")
        if arr.size == 0 and key != "proprio":
            raise ValueError(f"{name}[{key}] must not be empty")
