#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import DictConfig, OmegaConf

from vla_rl.config import instantiate
from vla_rl.runtime.agentlace import AgentlaceActorRuntime, AgentlaceLearnerRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agentlace actor or learner for VLA-RL.")
    parser.add_argument("--config", required=True, help="Path to a YAML recipe.")
    parser.add_argument("--role", required=True, choices=("actor", "learner"), help="Agentlace role to run.")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="OmegaConf dotlist overrides.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    cfg = OmegaConf.load(Path(args.config))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    cfg.runtime.role = args.role

    if args.role == "learner":
        algorithm = instantiate(cfg.algorithm)
        kwargs = _runtime_kwargs(cfg.runtime, AgentlaceLearnerRuntime)
        runner = AgentlaceLearnerRuntime(
            algorithm=algorithm,
            config_snapshot=OmegaConf.to_container(cfg, resolve=True),
            **kwargs,
        )
    else:
        env = instantiate(cfg.env)
        policy = instantiate(cfg.policy)
        feature_processor = instantiate(cfg.feature)
        algorithm = instantiate(cfg.algorithm)
        kwargs = _runtime_kwargs(cfg.runtime, AgentlaceActorRuntime)
        runner = AgentlaceActorRuntime(
            env=env,
            policy=policy,
            algorithm=algorithm,
            feature_processor=feature_processor,
            **kwargs,
        )

    summary = runner.run()
    print(json.dumps(summary, sort_keys=True))


def _runtime_kwargs(cfg: DictConfig, cls: type) -> dict[str, Any]:
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"runtime config must be a mapping, got {type(data).__name__}")
    ignored = {"backend", "role", "_target_"}
    params = set(inspect.signature(cls.__init__).parameters)
    return {key: value for key, value in data.items() if key in params and key not in ignored}


if __name__ == "__main__":
    main()
