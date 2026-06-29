from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from vla_rl.algorithms.pld import PLDObservationBuilder, PLDSACAgent, pld_base_action_prefix
from vla_rl.data import Observation
from vla_rl.envs.libero import LiberoLocalEnvBackend, LiberoRemoteEnvBackend
from vla_rl.policies import OpenPIWebsocketPolicyClient, ReferencePolicyClient


def load_config(path: str, overrides: list[str]) -> DictConfig:
    dotlist = list(overrides)
    if dotlist and dotlist[0] == "--":
        dotlist = dotlist[1:]
    cfg = OmegaConf.load(path)
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


def create_env(cfg: DictConfig):
    return _create_from_section(
        cfg.env,
        {
            "vla_rl.envs.libero.LiberoLocalEnvBackend": LiberoLocalEnvBackend,
            "vla_rl.envs.libero.LiberoRemoteEnvBackend": LiberoRemoteEnvBackend,
        },
    )


def create_reference_policy(cfg: DictConfig):
    return _create_from_section(
        cfg.policy,
        {
            "vla_rl.policies.OpenPIWebsocketPolicyClient": OpenPIWebsocketPolicyClient,
            "vla_rl.policies.ReferencePolicyClient": ReferencePolicyClient,
        },
    )


def create_pld_obs_builder(cfg: DictConfig) -> PLDObservationBuilder:
    return PLDObservationBuilder(
        **_section_kwargs(
            cfg.pld_observation,
            expected_target="vla_rl.algorithms.pld.PLDObservationBuilder",
        )
    )


def create_pld_agent(cfg: DictConfig) -> PLDSACAgent:
    return PLDSACAgent(
        **_section_kwargs(
            cfg.algorithm,
            expected_target="vla_rl.algorithms.pld.PLDSACAgent",
        )
    )


def reference_action_policy_horizon(cfg: DictConfig) -> int:
    section = cfg.get("reference_action", {})
    policy_horizon = int(section.get("policy_horizon", cfg.runtime.execute_horizon))
    if policy_horizon <= 0:
        raise ValueError("reference_action.policy_horizon must be positive")
    return policy_horizon


def predict_base_actions(
    reference_policy,
    obs: Observation,
    *,
    horizon: int,
    action_dim: int,
    policy_horizon: int | None = None,
) -> np.ndarray:
    kwargs: dict[str, Any] = {}
    if policy_horizon is not None:
        kwargs["num_steps"] = int(policy_horizon)
    base_actions = reference_policy.sample_actions(obs, **kwargs)
    return pld_base_action_prefix(base_actions, horizon=horizon, action_dim=action_dim)


def validate_pld_cfg(cfg: DictConfig) -> None:
    runtime = cfg.get("runtime", {})
    if "max_update_steps" in runtime:
        raise ValueError("runtime.max_update_steps has been removed; learner lifetime follows actor runtime.max_env_steps")
    if "pld_observation" not in cfg:
        raise ValueError("PLD config requires pld_observation section")
    if int(cfg.algorithm.chunk_horizon) != int(cfg.runtime.execute_horizon):
        raise ValueError("algorithm.chunk_horizon must match runtime.execute_horizon")
    if int(cfg.pld_observation.chunk_horizon) != int(cfg.runtime.execute_horizon):
        raise ValueError("pld_observation.chunk_horizon must match runtime.execute_horizon")
    if int(cfg.pld_observation.action_dim) != int(cfg.algorithm.action_dim):
        raise ValueError("pld_observation.action_dim must match algorithm.action_dim")
    if int(cfg.runtime.execute_horizon) <= 0:
        raise ValueError("runtime.execute_horizon must be positive")
    if reference_action_policy_horizon(cfg) < int(cfg.runtime.execute_horizon):
        raise ValueError("reference_action.policy_horizon must be >= runtime.execute_horizon")


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
