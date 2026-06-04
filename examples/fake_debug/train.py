#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from vla_rl.config import instantiate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fake-debug VLA-RL recipe.")
    parser.add_argument("--config", required=True, help="Path to a YAML recipe.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = OmegaConf.load(config_path)

    env = instantiate(cfg.env)
    policy = instantiate(cfg.policy)
    algorithm = instantiate(cfg.algorithm)
    feature_processor = instantiate(cfg.feature) if "feature" in cfg else None
    eval_cfg = cfg.get("eval", {})
    eval_enabled = bool(eval_cfg.get("enabled", False)) if eval_cfg is not None else False
    eval_env = instantiate(cfg.eval_env) if eval_enabled and "eval_env" in cfg else None
    eval_policy = instantiate(cfg.eval_policy) if eval_enabled and "eval_policy" in cfg else None
    eval_feature_processor = instantiate(cfg.eval_feature) if eval_enabled and "eval_feature" in cfg else None
    runner_kwargs = {"env": env, "policy": policy, "algorithm": algorithm}
    if feature_processor is not None:
        runner_kwargs["feature_processor"] = feature_processor
    if eval_enabled:
        runner_kwargs.update(
            {
                "eval_enabled": True,
                "eval_env": eval_env,
                "eval_policy": eval_policy,
                "eval_feature_processor": eval_feature_processor,
                "eval_interval_env_steps": int(eval_cfg.get("interval_env_steps", 0)),
                "eval_episodes": int(eval_cfg.get("episodes", 1)),
                "eval_deterministic": bool(eval_cfg.get("deterministic", True)),
                "eval_max_episode_steps": eval_cfg.get("max_episode_steps", None),
                "eval_metrics_key_prefix": str(eval_cfg.get("metrics_key_prefix", "eval")),
            }
        )
    runtime_target = str(cfg.runtime.get("_target_", ""))
    if runtime_target.endswith("LocalActorLearnerRunner"):
        runner_kwargs["config_snapshot"] = OmegaConf.to_container(cfg, resolve=True)
    runner = instantiate(cfg.runtime, **runner_kwargs)
    summary = runner.run()
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
