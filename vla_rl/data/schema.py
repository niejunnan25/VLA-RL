from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _as_float_array(name: str, value: np.ndarray, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be a floating array, got {array.dtype}")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape={array.shape}")
    return array.astype(np.float32, copy=False)


@dataclass(slots=True)
class ActionSpec:
    shape: tuple[int, ...]
    minimum: np.ndarray | float
    maximum: np.ndarray | float

    def validate(self) -> None:
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError(f"action shape must be positive, got {self.shape}")
        minimum = np.asarray(self.minimum, dtype=np.float32)
        maximum = np.asarray(self.maximum, dtype=np.float32)
        if minimum.shape not in [(), self.shape]:
            raise ValueError(f"minimum must be scalar or shape {self.shape}, got {minimum.shape}")
        if maximum.shape not in [(), self.shape]:
            raise ValueError(f"maximum must be scalar or shape {self.shape}, got {maximum.shape}")
        if np.any(maximum <= minimum):
            raise ValueError("maximum must be greater than minimum")


@dataclass(slots=True)
class ObservationSpec:
    image_keys: tuple[str, ...] = ()
    proprio_shape: tuple[int, ...] | None = None

    def validate(self) -> None:
        if len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError(f"duplicate image keys: {self.image_keys}")
        if self.proprio_shape is not None and any(dim <= 0 for dim in self.proprio_shape):
            raise ValueError(f"invalid proprio shape: {self.proprio_shape}")


@dataclass(slots=True)
class Observation:
    images: dict[str, np.ndarray] = field(default_factory=dict)
    proprio: np.ndarray | None = None
    task: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for key, image in self.images.items():
            array = np.asarray(image)
            if array.ndim != 3:
                raise ValueError(f"image {key!r} must be HWC, got shape={array.shape}")
            if array.shape[-1] not in (1, 3, 4):
                raise ValueError(f"image {key!r} must have 1, 3, or 4 channels, got {array.shape}")
        if self.proprio is not None:
            _as_float_array("proprio", self.proprio, ndim=1)


@dataclass(slots=True)
class PolicyFeatures:
    reference_actions: np.ndarray | None = None
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    proprio: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.reference_actions is not None:
            _as_float_array("reference_actions", self.reference_actions, ndim=2)
        for key, value in self.embeddings.items():
            _as_float_array(f"embeddings[{key}]", value)
        if self.proprio is not None:
            _as_float_array("features.proprio", self.proprio, ndim=1)


@dataclass(slots=True)
class Transition:
    obs: Observation
    action: np.ndarray
    reward: float
    next_obs: Observation
    done: bool
    discount: float
    truncated: bool = False
    agent_obs: dict[str, np.ndarray] | None = None
    next_agent_obs: dict[str, np.ndarray] | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.obs.validate()
        self.next_obs.validate()
        _as_float_array("transition.action", self.action, ndim=1)
        if self.agent_obs is not None:
            _validate_agent_obs("agent_obs", self.agent_obs)
        if self.next_agent_obs is not None:
            _validate_agent_obs("next_agent_obs", self.next_agent_obs)
        if not np.isfinite(self.reward):
            raise ValueError(f"reward must be finite, got {self.reward}")
        if not 0.0 <= float(self.discount) <= 1.0:
            raise ValueError(f"discount must be in [0, 1], got {self.discount}")


@dataclass(slots=True)
class RolloutBatch:
    transitions: list[Transition]

    def validate(self) -> None:
        if not self.transitions:
            raise ValueError("rollout batch must contain at least one transition")
        for transition in self.transitions:
            transition.validate()


def _validate_agent_obs(name: str, value: dict[str, np.ndarray]) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict, got {type(value).__name__}")
    for key, array in value.items():
        _as_float_array(f"{name}[{key}]", array)
