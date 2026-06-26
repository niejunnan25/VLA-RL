from __future__ import annotations

from typing import Any

import numpy as np


def assert_single_step_actions(actions: np.ndarray, *, context: str) -> np.ndarray:
    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim == 1:
        action_array = action_array.reshape(1, -1)
    if action_array.ndim != 2:
        raise AssertionError(f"{context}: single-step RLPD requires actions with shape [1, action_dim], got {action_array.shape}")
    if int(action_array.shape[0]) != 1:
        raise AssertionError(f"{context}: single-step RLPD requires exactly one action, got {action_array.shape[0]}")
    return action_array.astype(np.float32, copy=False)


def assert_single_executed_step(info: dict[str, Any], *, context: str) -> int:
    executed_steps = int(info.get("executed_steps", info.get("num_steps", 0)))
    if executed_steps != 1:
        raise AssertionError(f"{context}: single-step RLPD requires executed_steps=1, got {executed_steps}")
    return executed_steps


def single_step_discount(*, gamma: float, terminal: bool) -> float:
    return 0.0 if bool(terminal) else float(gamma)


def step_info_success(info: dict[str, Any]) -> bool:
    return bool(
        info.get("success", False)
        or info.get("env_done", False)
        or info.get("is_success", False)
    )


def single_step_action_summary(actions: np.ndarray) -> dict[str, float]:
    arr = np.asarray(actions, dtype=np.float32)
    return {"action_abs_mean": float(np.mean(np.abs(arr))), "action_abs_max": float(np.max(np.abs(arr)))}
