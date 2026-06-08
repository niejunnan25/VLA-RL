from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from vla_rl.algorithms.rlt.modeling import RLTokenEncoder


def encode_rlt_obs(
    prefix_tokens: np.ndarray,
    base_actions: np.ndarray,
    proprio: np.ndarray,
    *,
    rl_token_encoder: RLTokenEncoder,
) -> dict[str, np.ndarray]:
    """Encode frozen VLA prefix tokens and reference actions into RLT learner obs."""

    z_vla = torch.as_tensor(
        prefix_tokens,
        dtype=torch.float32,
        device=next(rl_token_encoder.parameters()).device,
    )
    if z_vla.dim() == 2:
        z_vla = z_vla.unsqueeze(0)
    if z_vla.dim() != 3:
        raise ValueError(f"prefix embeddings must be [B, T, D] or [T, D], got {tuple(z_vla.shape)}")

    max_tokens = getattr(rl_token_encoder, "max_tokens", None)
    if max_tokens is not None:
        z_vla = z_vla[:, : int(max_tokens), :]

    z_rl = rl_token_encoder(z_vla).squeeze(0).detach().cpu().numpy().astype(np.float32)
    reference_action = np.asarray(base_actions, dtype=np.float32).reshape(-1)
    return {
        "z_rl": z_rl,
        "reference_action": reference_action,
        "proprio": np.asarray(proprio, dtype=np.float32).reshape(-1),
        "action_mask": np.ones_like(reference_action, dtype=np.float32),
    }


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
    fallback_max_tokens: int | None = None,
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
    checkpoint_max_tokens = _normalize_max_tokens(rlt_cfg.get("max_tokens", None))
    if max_tokens is None:
        max_tokens = checkpoint_max_tokens
        if max_tokens is None:
            max_tokens = _normalize_max_tokens(fallback_max_tokens)
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
