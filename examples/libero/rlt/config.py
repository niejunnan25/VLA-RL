from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vla_rl.algorithms.rlt import RLTAgent, RLTokenEncoder
from vla_rl.algorithms.rlt.features import load_frozen_rlt_encoder
from vla_rl.envs.libero import LiberoRemoteEnvBackend
from vla_rl.policies import ReferencePolicyClient


def load_config(path: str, overrides: list[str]) -> DictConfig:
    dotlist = list(overrides)
    if dotlist and dotlist[0] == "--":
        dotlist = dotlist[1:]
    cfg = OmegaConf.load(Path(path))
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


def validate_rlt_cfg(cfg: DictConfig) -> None:
    rlt = rlt_cfg(cfg)
    agent_cfg = cfg_section(cfg, "algorithm")
    runtime = cfg_section(cfg, "runtime")
    if "execute_horizon" in agent_cfg or "execute_horizon" in runtime:
        raise ValueError("RLT v0 uses chunk_size only; remove execute_horizon from config")
    chunk_size = int(rlt.chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"rlt.chunk_size must be positive, got {chunk_size}")
    if int(agent_cfg.chunk_size) != chunk_size:
        raise ValueError("algorithm.chunk_size must match rlt.chunk_size")


def create_env(cfg: DictConfig) -> LiberoRemoteEnvBackend:
    return LiberoRemoteEnvBackend(
        **section_kwargs(
            cfg_section(cfg, "env"),
            expected_target="vla_rl.envs.libero.LiberoRemoteEnvBackend",
        )
    )


def build_reference_policy(cfg: DictConfig) -> ReferencePolicyClient:
    return ReferencePolicyClient(
        **section_kwargs(
            cfg_section(cfg, "policy"),
            expected_target="vla_rl.policies.ReferencePolicyClient",
        )
    )


def load_rl_token_encoder(cfg: DictConfig) -> RLTokenEncoder:
    feature = feature_cfg(cfg)
    encoder = load_frozen_rlt_encoder(
        str(feature.encoder_path),
        device=str(feature.get("device", "cpu")),
        input_dim=int(feature.get("input_dim", 2048)),
        rl_token_dim=int(feature.get("rl_token_dim", 2048)),
        num_encoder_layers=int(feature.get("num_encoder_layers", 4)),
        num_heads=int(feature.get("num_heads", 8)),
        ff_dim=int(feature.get("ff_dim", 2048)),
        dropout=float(feature.get("dropout", 0.0)),
        max_tokens=normalize_max_tokens(feature.get("max_tokens", 512)),
    )
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def create_rlt_agent(cfg: DictConfig) -> RLTAgent:
    return RLTAgent(
        **section_kwargs(
            cfg_section(cfg, "algorithm"),
            expected_target="vla_rl.algorithms.rlt.RLTAgent",
        )
    )


def rlt_cfg(cfg: DictConfig) -> DictConfig:
    return cfg_section(cfg, "rlt")


def feature_cfg(cfg: DictConfig) -> DictConfig:
    return cfg_section(cfg, "feature")


def cfg_section(cfg: DictConfig, *names: str) -> DictConfig:
    for name in names:
        if name in cfg:
            return cfg[name]
    raise ValueError(f"config is missing one of: {', '.join(names)}")


def section_kwargs(section: DictConfig, *, expected_target: str) -> dict[str, Any]:
    payload = OmegaConf.to_container(section, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping config section, got {type(payload).__name__}")
    target = payload.pop("_target_", None)
    if target is not None and str(target) != expected_target:
        raise ValueError(f"expected {expected_target}, got {target}")
    return payload


def normalize_max_tokens(value: Any) -> int | None:
    if value is None:
        return None
    max_tokens = int(value)
    return max_tokens if max_tokens > 0 else None
