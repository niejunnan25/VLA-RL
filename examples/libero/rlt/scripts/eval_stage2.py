#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from omegaconf import OmegaConf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vla_rl.runtime.wandb import make_wandb_logger

from examples.libero.rlt.config import (
    build_reference_policy,
    create_env,
    create_rlt_agent,
    feature_cfg,
    load_config,
    load_rl_token_encoder,
    resolve_online_feature_source,
    validate_rlt_cfg,
)
from examples.libero.rlt.scripts.train_stage2 import encode_rlt_obs, json_sanitize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LIBERO RLT checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-env-steps-per-episode", type=int, default=0)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, list(args.overrides))
    validate_rlt_cfg(cfg)
    summary = run_eval(
        cfg,
        checkpoint_path=Path(args.checkpoint),
        episodes=int(args.episodes),
        output_dir=Path(args.output_dir),
        max_env_steps_per_episode=int(args.max_env_steps_per_episode),
        save_videos=bool(args.save_videos),
        log_wandb=True,
    )
    print(json.dumps(json_sanitize(summary), sort_keys=True))


def run_eval(
    cfg,
    *,
    checkpoint_path: Path,
    episodes: int,
    output_dir: Path,
    max_env_steps_per_episode: int = 0,
    save_videos: bool = False,
    log_wandb: bool = True,
) -> dict[str, Any]:
    validate_rlt_cfg(cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = make_wandb_logger(cfg.get("wandb", None), variant=config_snapshot, run_dir=output_dir) if log_wandb else None

    env = None
    reference_policy = None
    try:
        env = create_env(cfg)
        reference_policy = build_reference_policy(cfg)
        feature = feature_cfg(cfg)
        online_feature_source = resolve_online_feature_source(feature)
        feature_num_steps = int(feature.get("num_steps", 10))
        replan_steps = int(cfg.rlt.get("replan_steps", cfg.rlt.chunk_size))
        encoder = load_rl_token_encoder(cfg)
        agent = create_rlt_agent(cfg)
        checkpoint = torch.load(Path(checkpoint_path), map_location=agent.device)
        agent.load_state_dict(checkpoint["algorithm_state"])

        episodes_path = output_dir / "eval_episodes.jsonl"
        episodes_path.write_text("")
        video_dir = output_dir / "videos"
        if save_videos:
            video_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        for episode_idx in range(int(episodes)):
            obs = env.reset()
            episode_return = 0.0
            episode_steps = 0
            success = False
            frames = []
            done = False
            truncated = False
            while not (done or truncated):
                if save_videos:
                    frames.append(_frame_from_obs(obs))
                base_actions, prefix_tokens, proprio = reference_policy.predict_actions_and_prefix(
                    obs,
                    feature_source=online_feature_source,
                    num_steps=feature_num_steps,
                )
                base_actions = np.asarray(base_actions, dtype=np.float32)[: int(cfg.rlt.chunk_size)]
                rlt_obs = encode_rlt_obs(prefix_tokens, base_actions, proprio, rl_token_encoder=encoder)
                actions = agent.sample_action(rlt_obs, deterministic=True)
                eval_actions = actions[:replan_steps]
                obs, reward, done, truncated, info = env.step_chunk(eval_actions)
                info = dict(info)
                executed_steps = int(info.get("executed_steps", len(eval_actions)))
                episode_return += float(reward)
                episode_steps += executed_steps
                success = bool(success or info.get("success", False) or info.get("env_done", False) or info.get("is_success", False))
                if int(max_env_steps_per_episode) > 0 and episode_steps >= int(max_env_steps_per_episode):
                    break
            if save_videos:
                _write_video(video_dir / f"episode_{episode_idx:04d}.mp4", frames)
            record = {"episode": episode_idx, "return": episode_return, "length": episode_steps, "success": bool(success)}
            records.append(record)
            with episodes_path.open("a") as f:
                f.write(json.dumps(json_sanitize(record), sort_keys=True) + "\n")

        summary = {
            "algorithm": "rlt",
            "episodes": len(records),
            "success_rate": float(np.mean([r["success"] for r in records])) if records else 0.0,
            "avg_return": float(np.mean([r["return"] for r in records])) if records else 0.0,
            "avg_length": float(np.mean([r["length"] for r in records])) if records else 0.0,
            "checkpoint_path": str(Path(checkpoint_path)),
            "feature_source": online_feature_source,
            "task": {"suite": cfg.env.get("task_suite_name", None), "task_id": cfg.env.get("task_id", None)},
        }
        (output_dir / "eval_summary.json").write_text(json.dumps(json_sanitize(summary), indent=2, sort_keys=True) + "\n")
        if wandb_logger is not None:
            wandb_logger.log(
                {
                    "eval/success_rate": summary["success_rate"],
                    "eval/mean_return": summary["avg_return"],
                    "eval/mean_steps": summary["avg_length"],
                    "eval/episodes_run": summary["episodes"],
                }
            )
        return summary
    finally:
        if env is not None:
            env.close()
        if reference_policy is not None:
            reference_policy.close()
        if wandb_logger is not None:
            wandb_logger.finish()


def _frame_from_obs(obs):
    key = sorted(obs.images)[0]
    return np.asarray(obs.images[key])


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    import imageio.v2 as imageio

    if frames:
        imageio.mimsave(path, frames, fps=10)


if __name__ == "__main__":
    main()
