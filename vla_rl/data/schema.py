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


ObsValue = dict[str, np.ndarray] | Observation


@dataclass(slots=True)
class Transition:
    obs: ObsValue
    action: np.ndarray
    reward: float
    next_obs: ObsValue | None
    done: bool
    discount: float
    truncated: bool = False
    executed_steps: int = 1
    env_steps: int = 0
    info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _validate_obs_value("obs", self.obs)
        if self.next_obs is None:
            if not (self.done or self.truncated):
                raise ValueError("non-terminal transition requires next_obs")
        else:
            _validate_obs_value("next_obs", self.next_obs)
        _as_float_array("transition.action", self.action, ndim=1)
        if not np.isfinite(self.reward):
            raise ValueError(f"reward must be finite, got {self.reward}")
        if not 0.0 <= float(self.discount) <= 1.0:
            raise ValueError(f"discount must be in [0, 1], got {self.discount}")
        if int(self.executed_steps) <= 0:
            raise ValueError(f"executed_steps must be positive, got {self.executed_steps}")

    @classmethod
    def from_payload(cls, payload: "Transition | dict[str, Any]") -> "Transition":
        if isinstance(payload, Transition):
            return payload
        return cls(
            obs=_copy_obs_value(payload["obs"]),
            next_obs=None if payload.get("next_obs") is None else _copy_obs_value(payload["next_obs"]),
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
            "obs": _copy_obs_value(self.obs),
            "next_obs": None if self.next_obs is None else _copy_obs_value(self.next_obs),
            "action": np.asarray(self.action, dtype=np.float32).reshape(-1),
            "reward": float(self.reward),
            "done": bool(self.done),
            "truncated": bool(self.truncated),
            "discount": float(self.discount),
            "executed_steps": int(self.executed_steps),
            "env_steps": int(self.env_steps),
            "info": dict(self.info),
        }


@dataclass(slots=True)
class RolloutBatch:
    transitions: list[Transition]

    def validate(self) -> None:
        if not self.transitions:
            raise ValueError("rollout batch must contain at least one transition")
        for transition in self.transitions:
            transition.validate()


def _validate_obs_value(name: str, value: ObsValue) -> None:
    if isinstance(value, Observation):
        value.validate()
        return
    _validate_obs_dict(name, value)


def _validate_obs_dict(name: str, value: dict[str, np.ndarray]) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an Observation or dict, got {type(value).__name__}")
    for key, array in value.items():
        if str(key).startswith("image_"):
            image = np.asarray(array)
            if image.dtype != np.uint8 and not np.issubdtype(image.dtype, np.floating):
                raise ValueError(f"{name}[{key}] image must be uint8 or floating, got {image.dtype}")
            if image.ndim != 3:
                raise ValueError(f"{name}[{key}] image must have ndim=3, got shape={image.shape}")
            continue
        _as_float_array(f"{name}[{key}]", array)


def _copy_obs_array(key: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if key.startswith("image_") and array.dtype == np.uint8:
        return np.ascontiguousarray(array.copy())
    return np.ascontiguousarray(array.astype(np.float32, copy=True))


def _copy_obs_value(value: ObsValue) -> ObsValue:
    if isinstance(value, Observation):
        return Observation(
            images={str(key): np.asarray(array).copy() for key, array in value.images.items()},
            proprio=None if value.proprio is None else np.asarray(value.proprio, dtype=np.float32).copy(),
            task=value.task,
            raw=dict(value.raw),
        )
    return {str(key): _copy_obs_array(str(key), array) for key, array in value.items()}
