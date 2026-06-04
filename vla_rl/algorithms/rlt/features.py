from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from vla_rl.algorithms.rlt.modeling import RLTokenEncoder
from vla_rl.data import Observation, PolicyFeatures


class RLTStateBuilder:
    def __init__(
        self,
        encoder_path: str | None = None,
        device: str = "cpu",
        input_dim: int = 2048,
        rl_token_dim: int = 2048,
        num_encoder_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.0,
        max_tokens: int | None = 512,
        chunk_size: int = 10,
        action_dim: int = 7,
        encoder: RLTokenEncoder | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        self.max_tokens = _normalize_max_tokens(max_tokens)
        if encoder is None:
            if encoder_path is None:
                raise ValueError("encoder_path is required when encoder is not injected")
            encoder = load_frozen_rlt_encoder(
                encoder_path,
                device=str(self.device),
                input_dim=input_dim,
                rl_token_dim=rl_token_dim,
                num_encoder_layers=num_encoder_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                max_tokens=self.max_tokens,
            )
            self.max_tokens = getattr(encoder, "max_tokens", self.max_tokens)
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        if not hasattr(self.encoder, "max_tokens"):
            self.encoder.max_tokens = self.max_tokens

    @torch.no_grad()
    def build(self, obs: Observation, features: PolicyFeatures) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        base_actions = self._base_actions(features)
        return base_actions, self.process(obs, features)

    @torch.no_grad()
    def process(self, obs: Observation, features: PolicyFeatures) -> dict[str, np.ndarray]:
        del obs
        if "prefix" not in features.embeddings:
            raise ValueError("RLTStateBuilder requires features.embeddings['prefix']")
        base_actions = self._base_actions(features)
        z_vla = torch.as_tensor(features.embeddings["prefix"], dtype=torch.float32, device=self.device)
        if z_vla.dim() == 2:
            z_vla = z_vla.unsqueeze(0)
        if z_vla.dim() != 3:
            raise ValueError(f"prefix embedding must be [B, T, D] or [T, D], got {tuple(z_vla.shape)}")
        max_tokens = getattr(self.encoder, "max_tokens", None)
        if max_tokens is not None:
            z_vla = z_vla[:, : int(max_tokens), :]
        z_rl = self.encoder(z_vla).squeeze(0).detach().cpu().numpy().astype(np.float32)
        if base_actions.ndim != 2:
            raise ValueError(f"reference actions must be [T, A], got shape={base_actions.shape}")
        if base_actions.shape[0] < self.chunk_size:
            raise ValueError(
                f"reference policy returned {base_actions.shape[0]} actions, need chunk_size={self.chunk_size}"
            )
        if base_actions.shape[1] < self.action_dim:
            raise ValueError(f"reference action dim {base_actions.shape[1]} is smaller than action_dim={self.action_dim}")
        reference = base_actions[: self.chunk_size, : self.action_dim].reshape(-1).astype(np.float32)
        proprio = features.proprio
        if proprio is None:
            proprio = np.zeros((0,), dtype=np.float32)
        return {
            "z_rl": z_rl,
            "reference_action": reference,
            "proprio": np.asarray(proprio, dtype=np.float32).reshape(-1),
        }

    def _base_actions(self, features: PolicyFeatures) -> np.ndarray:
        if features.reference_actions is None:
            raise ValueError("RLTStateBuilder requires PolicyFeatures.reference_actions")
        return np.asarray(features.reference_actions, dtype=np.float32)


def load_frozen_rlt_encoder(
    encoder_path: str,
    device: str = "cpu",
    *,
    input_dim: int = 2048,
    rl_token_dim: int = 2048,
    num_encoder_layers: int = 4,
    num_heads: int = 8,
    ff_dim: int = 2048,
    dropout: float = 0.0,
    max_tokens: int | None = None,
) -> RLTokenEncoder:
    checkpoint = torch.load(encoder_path, map_location="cpu")
    rlt_cfg: dict[str, Any] = {}
    if isinstance(checkpoint, dict):
        cfg = checkpoint.get("config", {})
        rlt_cfg = cfg.get("rlt", {}) if isinstance(cfg, dict) else {}
        state_dict = checkpoint.get("encoder_state_dict", checkpoint)
    else:
        state_dict = checkpoint

    if "e_rl" in state_dict:
        input_dim = int(state_dict["e_rl"].shape[-1])
    in_proj_key = next((key for key in state_dict if key.endswith("self_attn.in_proj_weight")), None)
    if in_proj_key is not None:
        rl_token_dim = int(state_dict[in_proj_key].shape[1])
    ff_key = next((key for key in state_dict if key.endswith("linear1.weight")), None)
    if ff_key is not None:
        ff_dim = int(state_dict[ff_key].shape[0])
    layer_ids = sorted(
        {
            int(match.group(1))
            for key in state_dict
            for match in [re.search(r"transformer\.layers\.(\d+)\.", key)]
            if match is not None
        }
    )
    if layer_ids:
        num_encoder_layers = max(layer_ids) + 1

    input_dim = int(rlt_cfg.get("input_dim", input_dim))
    rl_token_dim = int(rlt_cfg.get("rl_token_dim", rl_token_dim))
    num_encoder_layers = int(rlt_cfg.get("num_encoder_layers", num_encoder_layers))
    num_heads = int(rlt_cfg.get("num_heads", num_heads))
    ff_dim = int(rlt_cfg.get("ff_dim", ff_dim))
    dropout = float(rlt_cfg.get("dropout", dropout))
    if max_tokens is None:
        max_tokens = _normalize_max_tokens(rlt_cfg.get("max_tokens", None))
    else:
        max_tokens = _normalize_max_tokens(max_tokens)

    encoder = RLTokenEncoder(
        input_dim=input_dim,
        rl_token_dim=rl_token_dim,
        num_layers=num_encoder_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
    )
    encoder.load_state_dict(state_dict)
    encoder.eval()
    encoder.requires_grad_(False)
    encoder.max_tokens = max_tokens
    encoder.to(torch.device(device))
    return encoder


def _normalize_max_tokens(value: Any) -> int | None:
    if value is None:
        return None
    max_tokens = int(value)
    if max_tokens <= 0:
        return None
    return max_tokens
