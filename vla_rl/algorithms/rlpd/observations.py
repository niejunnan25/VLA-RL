from __future__ import annotations

from typing import Iterable

import numpy as np

from vla_rl.data import Observation


class ObservationBuilder:
    """Build standard RLPD observations from env observations.

    The SAC policy observes robot state/images and directly outputs the
    environment action. No base-policy action is included.
    """

    def __init__(self, image_keys: Iterable[str] = ("image_rgb_0", "image_rgb_1")) -> None:
        self.image_keys = tuple(str(key) for key in image_keys)
        if not self.image_keys:
            raise ValueError("ObservationBuilder requires at least one image key")

    def build_observation(self, obs: Observation) -> dict[str, np.ndarray]:
        proprio = np.asarray(obs.proprio, dtype=np.float32).reshape(-1)
        result: dict[str, np.ndarray] = {"proprio": proprio}
        for key in self.image_keys:
            if key not in obs.images:
                raise KeyError(f"observation missing image key {key!r}; available={sorted(obs.images)}")
            result[f"image_{key}"] = _image_to_chw_uint8(obs.images[key])
        return result


def build_rlpd_obs(obs: Observation, *, builder: ObservationBuilder) -> dict[str, np.ndarray]:
    return builder.build_observation(obs)


def _image_to_chw_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"image must be HWC or CHW, got shape={arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        chw = arr
    else:
        chw = np.transpose(arr, (2, 0, 1))
    if chw.dtype == np.uint8:
        return np.ascontiguousarray(chw)
    if np.issubdtype(chw.dtype, np.floating):
        if chw.size and np.nanmin(chw) >= -1e-6 and np.nanmax(chw) <= 1.0 + 1e-6:
            chw = chw * 255.0
        return np.ascontiguousarray(np.rint(np.clip(chw, 0.0, 255.0)).astype(np.uint8))
    return np.ascontiguousarray(np.clip(chw, 0, 255).astype(np.uint8))
