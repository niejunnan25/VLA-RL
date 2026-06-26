from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vla_rl.algorithms.rlpd import ObservationBuilder, SACAgent
from vla_rl.envs.libero import LiberoLocalEnvBackend, LiberoRemoteEnvBackend


def load_config(path: str, overrides: list[str]) -> DictConfig:
    dotlist = list(overrides)
    if dotlist and dotlist[0] == "--":
        dotlist = dotlist[1:]
    cfg = OmegaConf.load(Path(path))
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


def validate_rlpd_cfg(cfg: DictConfig) -> None:
    _reject_removed_fields(cfg)
    if "rlpd_observation" not in cfg:
        raise ValueError("RLPD config requires rlpd_observation section")
    if int(cfg.algorithm.action_dim) <= 0:
        raise ValueError("algorithm.action_dim must be positive")
    offline_ratio = float(cfg.runtime.get("offline_ratio", 0.0))
    if not 0.0 <= offline_ratio <= 1.0:
        raise ValueError(f"runtime.offline_ratio must be in [0, 1], got {offline_ratio}")
    if int(cfg.runtime.get("batch_size", 0)) <= 0:
        raise ValueError("runtime.batch_size must be positive")
    if int(cfg.runtime.get("max_env_steps", 0)) <= 0:
        raise ValueError("runtime.max_env_steps must be positive")


def _reject_removed_fields(cfg: DictConfig) -> None:
    removed = []
    if "rlpd" in cfg:
        removed.append("rlpd.mode")
    runtime = cfg.get("runtime", {})
    algorithm = cfg.get("algorithm", {})
    observation = cfg.get("rlpd_observation", {})
    reward = cfg.get("reward", {})
    if "role" in runtime:
        removed.append("runtime.role")
    if "execute_horizon" in runtime:
        removed.append("runtime.execute_horizon")
    if "random_steps" in runtime:
        removed.append("runtime.random_steps")
    if "require_offline" in runtime:
        removed.append("runtime.require_offline")
    if "warmup_steps" in runtime:
        removed.append("runtime.warmup_steps; use runtime.training_starts for learner replay warmup")
    if "max_update_steps" in runtime:
        removed.append("runtime.max_update_steps; learner lifetime follows actor runtime.max_env_steps")
    if "chunk_horizon" in algorithm:
        removed.append("algorithm.chunk_horizon")
    if "chunk_horizon" in observation:
        removed.append("rlpd_observation.chunk_horizon")
    if "action_dim" in observation:
        removed.append("rlpd_observation.action_dim")
    if "on_error" in reward:
        removed.append("reward.on_error")
    if removed:
        joined = ", ".join(removed)
        raise ValueError(
            f"remove deprecated RLPD config fields: {joined}. "
            "This package implements standard single-step RLPD only; these are implementation invariants, not experiment knobs."
        )


def create_env(cfg: DictConfig):
    return _create_from_section(
        cfg.env,
        {
            "vla_rl.envs.libero.LiberoLocalEnvBackend": LiberoLocalEnvBackend,
            "vla_rl.envs.libero.LiberoRemoteEnvBackend": LiberoRemoteEnvBackend,
        },
    )


def create_rlpd_obs_builder(cfg: DictConfig) -> ObservationBuilder:
    return ObservationBuilder(
        **_section_kwargs(
            cfg.rlpd_observation,
            expected_target="vla_rl.algorithms.rlpd.ObservationBuilder",
        )
    )


def create_rlpd_agent(cfg: DictConfig) -> SACAgent:
    return SACAgent(
        **_section_kwargs(
            cfg.algorithm,
            expected_target="vla_rl.algorithms.rlpd.SACAgent",
        )
    )


def _create_from_section(section: DictConfig, target_map: dict[str, Any]):
    payload = OmegaConf.to_container(section, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping config section, got {type(payload).__name__}")
    target = payload.pop("_target_", None)
    if target not in target_map:
        supported = ", ".join(sorted(target_map))
        raise ValueError(f"unsupported target {target!r}; supported targets: {supported}")
    return target_map[str(target)](**payload)


def _section_kwargs(section: DictConfig, *, expected_target: str) -> dict[str, Any]:
    payload = OmegaConf.to_container(section, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping config section, got {type(payload).__name__}")
    target = payload.pop("_target_", None)
    if target is not None and str(target) != expected_target:
        raise ValueError(f"expected {expected_target}, got {target}")
    return payload
