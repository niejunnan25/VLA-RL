#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.libero.rlt.config import (
    build_reference_policy,
    create_env,
    feature_cfg,
    load_config,
    reference_action_policy_horizon,
    reference_action_stride,
    resolve_online_feature_source,
    rlt_cfg,
    validate_rlt_cfg,
)
from examples.libero.rlt.rollout import predict_reference_actions_for_chunk
from vla_rl.runtime.agentlace import json_sanitize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the LIBERO reference/base policy used by RLT.")
    parser.add_argument("--config", required=True, help="Path to an RLT YAML config.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-env-steps-per-episode", type=int, default=0)
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="OmegaConf dotlist overrides after --.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, list(args.overrides))
    validate_rlt_cfg(cfg)
    summary = run_eval_base_policy(
        cfg,
        episodes=int(args.episodes),
        output_dir=Path(args.output_dir),
        max_env_steps_per_episode=int(args.max_env_steps_per_episode),
    )
    print(json.dumps(json_sanitize(summary), sort_keys=True))


def run_eval_base_policy(
    cfg,
    *,
    episodes: int,
    output_dir: Path,
    max_env_steps_per_episode: int = 0,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = OmegaConf.to_container(cfg, resolve=True)
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(config_snapshot)))

    env = None
    reference_policy = None
    try:
        env = create_env(cfg)
        reference_policy = build_reference_policy(cfg)
        feature = feature_cfg(cfg)
        online_feature_source = resolve_online_feature_source(feature)
        reference_policy_horizon = reference_action_policy_horizon(cfg)
        reference_action_stride_value = reference_action_stride(cfg)
        rlt = rlt_cfg(cfg)
        chunk_size = int(rlt.chunk_size)
        replan_steps = int(rlt.get("replan_steps", chunk_size))

        episodes_path = output_dir / "eval_episodes.jsonl"
        episodes_path.write_text("")
        records: list[dict[str, Any]] = []

        progress = tqdm(range(int(episodes)), desc="base-policy-eval", dynamic_ncols=True)
        for episode_idx in progress:
            obs = env.reset()
            episode_return = 0.0
            episode_steps = 0
            success = False
            done = False
            truncated = False

            while not (done or truncated):
                base_actions, _, _ = predict_reference_actions_for_chunk(
                    reference_policy,
                    obs,
                    feature_source=online_feature_source,
                    num_steps=reference_policy_horizon,
                    chunk_size=chunk_size,
                    action_stride=reference_action_stride_value,
                )
                eval_actions = np.asarray(base_actions[:replan_steps], dtype=np.float32)
                obs, reward, done, truncated, info = env.step_chunk(eval_actions)
                info = dict(info)
                executed_steps = int(info.get("executed_steps", len(eval_actions)))
                episode_return += float(reward)
                episode_steps += executed_steps
                success = bool(
                    success
                    or info.get("success", False)
                    or info.get("env_done", False)
                    or info.get("is_success", False)
                )
                if int(max_env_steps_per_episode) > 0 and episode_steps >= int(max_env_steps_per_episode):
                    break

            record = {
                "episode": int(episode_idx),
                "return": float(episode_return),
                "length": int(episode_steps),
                "success": bool(success),
            }
            records.append(record)
            with episodes_path.open("a") as f:
                f.write(json.dumps(json_sanitize(record), sort_keys=True) + "\n")
            success_rate = float(np.mean([r["success"] for r in records]))
            progress.set_postfix(
                success=f"{success_rate:.3f}",
                last=int(success),
                steps=episode_steps,
            )

        summary = {
            "algorithm": "base_policy",
            "episodes": len(records),
            "success_rate": float(np.mean([r["success"] for r in records])) if records else 0.0,
            "avg_return": float(np.mean([r["return"] for r in records])) if records else 0.0,
            "avg_length": float(np.mean([r["length"] for r in records])) if records else 0.0,
            "feature_source": str(online_feature_source),
            "reference_action_policy_horizon": int(reference_policy_horizon),
            "reference_action_stride": int(reference_action_stride_value),
            "chunk_size": int(chunk_size),
            "replan_steps": int(replan_steps),
            "executed_action_steps": int(min(replan_steps, chunk_size)),
            "task": {
                "suite": cfg.env.get("task_suite_name", None),
                "task_id": cfg.env.get("task_id", None),
            },
        }
        (output_dir / "eval_summary.json").write_text(
            json.dumps(json_sanitize(summary), indent=2, sort_keys=True) + "\n"
        )
        return summary
    finally:
        if env is not None:
            env.close()
        if reference_policy is not None:
            reference_policy.close()


if __name__ == "__main__":
    main()
