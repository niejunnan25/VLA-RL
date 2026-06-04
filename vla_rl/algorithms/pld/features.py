from __future__ import annotations

from typing import Iterable

import numpy as np

from vla_rl.data import Observation, PolicyFeatures
from vla_rl.features import FeatureProcessor


class PLDFeatureProcessor(FeatureProcessor):
    """Build PLD residual observations from env observations and base actions."""

    def __init__(
        self,
        image_keys: Iterable[str] = ("image_rgb_0", "image_rgb_1"),
        action_dim: int = 7,
        chunk_horizon: int = 1,
        alpha: float = 0.5,
    ) -> None:
        self.image_keys = tuple(str(key) for key in image_keys)
        if not self.image_keys:
            raise ValueError("PLDFeatureProcessor requires at least one image key")
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.alpha = float(alpha)
        if self.action_dim <= 0 or self.chunk_horizon <= 0:
            raise ValueError("action_dim and chunk_horizon must be positive")

    def process(self, obs: Observation, features: PolicyFeatures) -> dict[str, np.ndarray]:
        base = np.asarray(features.reference_actions, dtype=np.float32)
        proprio = np.asarray(obs.proprio, dtype=np.float32).reshape(-1)
        result: dict[str, np.ndarray] = {
            "proprio": proprio,
            "base_action_chunk": base[: self.chunk_horizon, : self.action_dim].astype(np.float32, copy=False),
            "alpha": np.asarray([self.alpha], dtype=np.float32),
        }
        for key in self.image_keys:
            result[f"image_{key}"] = _image_to_chw_float(obs.images[key])
        return result


def _image_to_chw_float(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return np.transpose(arr, (2, 0, 1)).astype(np.float32) / 255.0
    return np.transpose(arr.astype(np.float32, copy=False), (2, 0, 1))
