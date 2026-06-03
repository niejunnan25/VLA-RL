from __future__ import annotations

from typing import Any

import numpy as np

from vla_rl.data import Observation

LIBERO_IMAGE_KEYS = ("image_rgb_0", "image_rgb_1", "image_rgb_2")
LIBERO_STATE_DIM = 8


def build_libero_observation(
    raw_obs: dict[str, Any],
    task: str | None = None,
    image_size: int = 224,
) -> Observation:
    state = build_libero_state(raw_obs)
    images = extract_libero_images(raw_obs, image_size=image_size)
    openpi_observation = {
        "prompt": task,
        "observation/image": images["image_rgb_0"],
        "observation/wrist_image": images["image_rgb_1"],
        "observation/state": state,
        "state": state,
        "images": images,
        "image_mask": {
            "image_rgb_0": True,
            "image_rgb_1": True,
            "image_rgb_2": False,
        },
        # Also expose flat image keys for OpenPI forks that consume dict inputs
        # directly instead of a small typed PolicyInput wrapper.
        **{f"image/{key}": value for key, value in images.items()},
    }
    obs = Observation(
        images=images,
        proprio=state,
        task=task,
        raw={
            "libero": raw_obs,
            "openpi_observation": openpi_observation,
        },
    )
    obs.validate()
    return obs


def build_libero_state(raw_obs: dict[str, Any]) -> np.ndarray:
    pos = _first_present(raw_obs, ("robot0_eef_pos", "eef_pos", "ee_pos"), default=np.zeros(3, dtype=np.float32))
    quat = _first_present(raw_obs, ("robot0_eef_quat", "eef_quat", "ee_quat"), default=np.array([0, 0, 0, 1], dtype=np.float32))
    gripper = _first_present(raw_obs, ("robot0_gripper_qpos", "gripper_qpos"), default=np.zeros(1, dtype=np.float32))
    axis_angle = quat_to_axis_angle(np.asarray(quat, dtype=np.float32).reshape(-1)[:4])
    gripper_values = np.asarray(gripper, dtype=np.float32).reshape(-1)
    if gripper_values.size < 2:
        gripper_values = np.pad(gripper_values, (0, 2 - gripper_values.size))
    gripper_values = gripper_values[:2]
    state = np.concatenate(
        [
            np.asarray(pos, dtype=np.float32).reshape(-1)[:3],
            axis_angle.astype(np.float32),
            gripper_values.astype(np.float32),
        ],
        axis=0,
    )
    if state.shape != (LIBERO_STATE_DIM,):
        raise ValueError(f"expected LIBERO state dim {LIBERO_STATE_DIM}, got {state.shape}")
    return state.astype(np.float32, copy=False)


def extract_libero_images(raw_obs: dict[str, Any], image_size: int = 224) -> dict[str, np.ndarray]:
    front = _first_present(raw_obs, ("agentview_image", "front_image", "image", "rgb"), default=None)
    wrist = _first_present(raw_obs, ("robot0_eye_in_hand_image", "wrist_image", "eye_in_hand_image"), default=None)
    if front is None:
        front = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    if wrist is None:
        wrist = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    images = {
        "image_rgb_0": normalize_image(front, image_size=image_size),
        "image_rgb_1": normalize_image(wrist, image_size=image_size),
        "image_rgb_2": np.zeros((image_size, image_size, 3), dtype=np.uint8),
    }
    return images


def normalize_image(image: Any, image_size: int = 224) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"image must be HWC, got shape={array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    # LIBERO camera frames are commonly upside down relative to policy input.
    array = array[::-1, ::-1]
    if array.shape[:2] != (image_size, image_size):
        array = resize_nearest(array, (image_size, image_size))
    return np.ascontiguousarray(array)


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        return np.zeros(3, dtype=np.float32)
    quat = quat / norm
    xyz = quat[:3]
    w = float(np.clip(quat[3], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = xyz / sin_half
    return (axis * angle).astype(np.float32)


def resize_nearest(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    out_h, out_w = size
    in_h, in_w = image.shape[:2]
    y = np.linspace(0, in_h - 1, out_h).astype(np.int64)
    x = np.linspace(0, in_w - 1, out_w).astype(np.int64)
    return image[y][:, x]


def _first_present(raw_obs: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for key in keys:
        if key in raw_obs:
            return raw_obs[key]
    return default
