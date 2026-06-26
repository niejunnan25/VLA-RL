from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ResidualActionSpec:
    full_action_dim: int = 7
    action_mask: tuple[bool, ...] | None = None
    action_limits: tuple[float, ...] | None = None
    alpha: float = 0.5
    clip_gripper: bool = True
    chunk_horizon: int = 1

    def __post_init__(self) -> None:
        full_dim = int(self.full_action_dim)
        horizon = int(self.chunk_horizon)
        if full_dim <= 0:
            raise ValueError(f"full_action_dim must be positive, got {full_dim}")
        if horizon <= 0:
            raise ValueError(f"chunk_horizon must be positive, got {horizon}")
        mask = _resolve_mask(full_dim, self.action_mask)
        limits = _resolve_limits(full_dim, self.action_limits)
        if not np.any(mask):
            raise ValueError("action_mask must enable at least one residual dimension")
        alpha = float(self.alpha)
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError(f"alpha must be finite and >= 0, got {self.alpha}")
        object.__setattr__(self, "full_action_dim", full_dim)
        object.__setattr__(self, "chunk_horizon", horizon)
        object.__setattr__(self, "action_mask", tuple(bool(x) for x in mask.tolist()))
        object.__setattr__(self, "action_limits", tuple(float(x) for x in limits.tolist()))
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "clip_gripper", bool(self.clip_gripper))

    @property
    def control_indices(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.action_mask, dtype=bool)).astype(np.int64)

    @property
    def residual_limits(self) -> np.ndarray:
        limits = np.asarray(self.action_limits, dtype=np.float32)
        return limits[self.control_indices]

    @property
    def residual_action_dim(self) -> int:
        return int(self.control_indices.size)

    @property
    def policy_action_dim(self) -> int:
        return int(self.chunk_horizon * self.residual_action_dim)

    @property
    def critic_action_dim(self) -> int:
        return int(self.chunk_horizon * self.full_action_dim)

    def compose_chunk(self, base_action_chunk: np.ndarray, residual_action: np.ndarray) -> np.ndarray:
        base = _as_chunk(base_action_chunk, self.chunk_horizon, self.full_action_dim, "base_action_chunk")
        residual = np.asarray(residual_action, dtype=np.float32)
        if residual.ndim == 1:
            residual = residual.reshape(self.chunk_horizon, self.residual_action_dim)
        if residual.shape != (self.chunk_horizon, self.residual_action_dim):
            raise ValueError(
                "residual_action shape mismatch: "
                f"got {residual.shape}, expected {(self.chunk_horizon, self.residual_action_dim)}"
            )
        clipped = np.clip(residual, -1.0, 1.0)
        delta_selected = clipped * float(self.alpha) * self.residual_limits.reshape(1, -1)
        delta = np.zeros_like(base, dtype=np.float32)
        delta[:, self.control_indices] = delta_selected
        final = base + delta
        # The composed action is executed by the environment, whose action space
        # is bounded to [-1, 1] on every dimension. Clip all dims (not only the
        # gripper) so the action stored for the critic matches what the env runs.
        final = np.clip(final, -1.0, 1.0)
        return final.astype(np.float32)

    def compose_chunk_torch(self, base_action_chunk: torch.Tensor, residual_action: torch.Tensor) -> torch.Tensor:
        if base_action_chunk.ndim != 3:
            raise ValueError(f"base_action_chunk must be [B,H,A], got {tuple(base_action_chunk.shape)}")
        if residual_action.ndim == 2:
            residual_action = residual_action.reshape(
                residual_action.shape[0], self.chunk_horizon, self.residual_action_dim
            )
        if residual_action.ndim != 3:
            raise ValueError(f"residual_action must be [B,H,R] or [B,H*R], got {tuple(residual_action.shape)}")
        indices = torch.as_tensor(self.control_indices, dtype=torch.long, device=base_action_chunk.device)
        limits = torch.as_tensor(self.residual_limits, dtype=base_action_chunk.dtype, device=base_action_chunk.device)
        clipped = torch.clamp(residual_action, -1.0, 1.0)
        delta_selected = clipped * float(self.alpha) * limits.view(1, 1, -1)
        delta = torch.zeros_like(base_action_chunk)
        delta.index_copy_(2, indices, delta_selected)
        final = base_action_chunk + delta
        # Match the executed (env-clipped) action on every dimension, not just the gripper.
        final = torch.clamp(final, -1.0, 1.0)
        return final


def _resolve_mask(full_dim: int, mask: Sequence[bool] | None) -> np.ndarray:
    if mask is None:
        return np.ones((full_dim,), dtype=bool)
    arr = np.asarray(list(mask), dtype=bool).reshape(-1)
    if arr.size != full_dim:
        raise ValueError(f"action_mask length {arr.size} does not match full_action_dim={full_dim}")
    return arr


def _resolve_limits(full_dim: int, limits: Sequence[float] | None) -> np.ndarray:
    if limits is None:
        return np.ones((full_dim,), dtype=np.float32)
    arr = np.asarray(list(limits), dtype=np.float32).reshape(-1)
    if arr.size != full_dim:
        raise ValueError(f"action_limits length {arr.size} does not match full_action_dim={full_dim}")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"action_limits must be finite and non-negative, got {arr.tolist()}")
    return arr


def _as_chunk(value: np.ndarray, horizon: int, action_dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(int(horizon), int(action_dim))
    if arr.shape != (int(horizon), int(action_dim)):
        raise ValueError(f"{name} shape mismatch: got {arr.shape}, expected {(int(horizon), int(action_dim))}")
    return arr
