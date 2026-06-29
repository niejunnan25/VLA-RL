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

from examples.libero.pld.config import (  # noqa: E402
    create_env,
    create_pld_agent,
    create_pld_obs_builder,
    load_config,
    validate_pld_cfg,
)
from vla_rl.algorithms.pld import build_pld_obs, pld_base_action_prefix  # noqa: E402
from vla_rl.policies.openpi.backend import official_openpi_libero_payload  # noqa: E402
from vla_rl.runtime.agentlace import json_sanitize  # noqa: E402


class OfficialOpenPIWebsocketPolicy:
    def __init__(self, host: str, port: int) -> None:
        from openpi_client import websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(host, int(port))

    def sample_actions(self, obs, *, horizon: int, action_dim: int) -> np.ndarray:
        if "openpi_observation" not in obs.raw:
            raise RuntimeError("LIBERO observation does not contain raw['openpi_observation']")
        result = self.client.infer(official_openpi_libero_payload(dict(obs.raw["openpi_observation"])))
        if not isinstance(result, dict) or "actions" not in result:
            raise RuntimeError(f"official OpenPI server returned invalid payload: {type(result).__name__}")
        return pld_base_action_prefix(result["actions"], horizon=horizon, action_dim=action_dim)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LIBERO PLD/residual SAC checkpoints with the official OpenPI websocket inference path.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--max-env-steps-per-episode", type=int, default=0)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--force-zero-residual", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, list(args.overrides))
    validate_pld_cfg(cfg)
    summary = run_eval(
        cfg,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        episodes=int(args.episodes),
        output_dir=Path(args.output_dir),
        policy_host=str(args.policy_host),
        policy_port=int(args.policy_port),
        max_env_steps_per_episode=int(args.max_env_steps_per_episode),
        save_videos=bool(args.save_videos),
        force_zero_residual=bool(args.force_zero_residual),
    )
    print(json.dumps(json_sanitize(summary), sort_keys=True))


def run_eval(
    cfg,
    *,
    checkpoint_path: Path | None,
    episodes: int,
    output_dir: Path,
    policy_host: str,
    policy_port: int,
    max_env_steps_per_episode: int = 0,
    save_videos: bool = False,
    force_zero_residual: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    env = None
    policy = None
    agent = None
    train_episode = 0
    try:
        env = create_env(cfg)
        policy = OfficialOpenPIWebsocketPolicy(policy_host, policy_port)
        obs_builder = create_pld_obs_builder(cfg)

        if not force_zero_residual:
            if checkpoint_path is None:
                raise ValueError("--checkpoint is required unless --force-zero-residual is set")
            agent = create_pld_agent(cfg)
            checkpoint = torch.load(Path(checkpoint_path), map_location=agent.device)
            agent.load_state_dict(checkpoint["algorithm_state"])
            train_episode = int(checkpoint.get("episodes", 0) or 0)
        elif checkpoint_path is not None and Path(checkpoint_path).exists():
            checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
            train_episode = int(checkpoint.get("episodes", 0) or 0)

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
            residual_l1: list[float] = []
            residual_l2: list[float] = []
            frames: list[np.ndarray] = []
            done = False
            truncated = False
            init_state_idx = None

            while not (done or truncated):
                if save_videos:
                    frames.append(_frame_from_obs(obs))
                base_actions = policy.sample_actions(
                    obs,
                    horizon=int(cfg.runtime.execute_horizon),
                    action_dim=int(cfg.algorithm.action_dim),
                )
                pld_obs = build_pld_obs(obs, base_actions, builder=obs_builder)
                if force_zero_residual:
                    final_actions = np.asarray(base_actions, dtype=np.float32).copy()
                else:
                    assert agent is not None
                    final_actions = agent.sample_action(pld_obs, deterministic=True)
                delta = np.asarray(final_actions, dtype=np.float32) - np.asarray(base_actions, dtype=np.float32)
                residual_l1.append(float(np.mean(np.abs(delta))))
                residual_l2.append(float(np.sqrt(np.mean(delta * delta))))

                obs, reward, done, truncated, info = env.step_chunk(final_actions)
                info = dict(info)
                init_state_idx = info.get("init_state_idx", init_state_idx)
                executed_steps = int(info.get("executed_steps", len(final_actions)))
                episode_return += float(reward)
                episode_steps += executed_steps
                success = bool(success or info.get("success", False) or info.get("env_done", False) or info.get("is_success", False))
                if int(max_env_steps_per_episode) > 0 and episode_steps >= int(max_env_steps_per_episode):
                    break

            if save_videos:
                _write_video(video_dir / f"episode_{episode_idx:04d}.mp4", frames)
            record = {
                "episode": episode_idx,
                "init_state_idx": init_state_idx,
                "return": episode_return,
                "length": episode_steps,
                "success": bool(success),
                "residual_l1_mean": float(np.mean(residual_l1)) if residual_l1 else 0.0,
                "residual_l2_mean": float(np.mean(residual_l2)) if residual_l2 else 0.0,
            }
            records.append(record)
            with episodes_path.open("a") as f:
                f.write(json.dumps(json_sanitize(record), sort_keys=True) + "\n")

        summary = {
            "algorithm": "residual_sac",
            "base_policy_path": "official_openpi_websocket_infer",
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "episodes": len(records),
            "force_zero_residual": bool(force_zero_residual),
            "avg_length": float(np.mean([r["length"] for r in records])) if records else 0.0,
            "avg_residual_l1": float(np.mean([r["residual_l1_mean"] for r in records])) if records else 0.0,
            "avg_residual_l2": float(np.mean([r["residual_l2_mean"] for r in records])) if records else 0.0,
            "avg_return": float(np.mean([r["return"] for r in records])) if records else 0.0,
            "success_rate": float(np.mean([r["success"] for r in records])) if records else 0.0,
            "task": {"suite": cfg.env.get("task_suite_name", None), "task_id": cfg.env.get("task_id", None)},
            "train_episode": train_episode,
        }
        (output_dir / "eval_summary.json").write_text(json.dumps(json_sanitize(summary), indent=2, sort_keys=True) + "\n")
        return summary
    finally:
        if env is not None:
            env.close()
        if policy is not None:
            policy.close()


def _frame_from_obs(obs):
    key = sorted(obs.images)[0]
    return np.asarray(obs.images[key])


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    import imageio.v2 as imageio

    if frames:
        imageio.mimsave(path, frames, fps=10)


if __name__ == "__main__":
    main()
