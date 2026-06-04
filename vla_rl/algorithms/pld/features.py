from __future__ import annotations

from typing import Iterable

import numpy as np

from vla_rl.data import Observation


class PLDObservationBuilder:
    """Build residual-RL observations from env observations and frozen VLA base actions."""

    def __init__(
        self,
        image_keys: Iterable[str] = ("image_rgb_0", "image_rgb_1"),
        action_dim: int = 7,
        chunk_horizon: int = 1,
        alpha: float = 0.5,
    ) -> None:
        self.image_keys = tuple(str(key) for key in image_keys)
        if not self.image_keys:
            raise ValueError("PLDObservationBuilder requires at least one image key")
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.alpha = float(alpha)
        if self.action_dim <= 0 or self.chunk_horizon <= 0:
            raise ValueError("action_dim and chunk_horizon must be positive")

    def build_observation(self, obs: Observation, base_actions: np.ndarray) -> dict[str, np.ndarray]:
        base = _base_action_prefix(base_actions, horizon=self.chunk_horizon, action_dim=self.action_dim)
        proprio = np.asarray(obs.proprio, dtype=np.float32).reshape(-1)
        result: dict[str, np.ndarray] = {
            "proprio": proprio,
            "base_action_chunk": base,
            "alpha": np.asarray([self.alpha], dtype=np.float32),
        }
        for key in self.image_keys:
            result[f"image_{key}"] = _image_to_chw_float(obs.images[key])
        return result


def build_pld_obs(obs: Observation, base_actions: np.ndarray, *, builder: PLDObservationBuilder) -> dict[str, np.ndarray]:
    return builder.build_observation(obs, base_actions)


def pld_base_action_prefix(base_actions: np.ndarray, *, horizon: int, action_dim: int) -> np.ndarray:
    return _base_action_prefix(base_actions, horizon=horizon, action_dim=action_dim)


def _base_action_prefix(base_actions: np.ndarray, *, horizon: int, action_dim: int) -> np.ndarray:
    base = np.asarray(base_actions, dtype=np.float32)
    if base.ndim != 2:
        raise ValueError(f"base_actions must have shape (T, A), got {base.shape}")
    if base.shape[0] < horizon or base.shape[1] < action_dim:
        raise ValueError(f"base_actions shape {base.shape} is shorter than required {(horizon, action_dim)}")
    return base[:horizon, :action_dim].astype(np.float32, copy=False)


def _image_to_chw_float(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return np.transpose(arr, (2, 0, 1)).astype(np.float32) / 255.0
    return np.transpose(arr.astype(np.float32, copy=False), (2, 0, 1))
