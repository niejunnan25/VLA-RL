#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.libero.pld.config import (
    create_env,
    create_pld_obs_builder,
    create_reference_policy,
    load_config,
    predict_base_actions,
    validate_pld_cfg,
)
from vla_rl.algorithms.pld import build_pld_obs, write_pld_offline_episode
from vla_rl.data import Transition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect successful base-policy episodes for PLD offline replay.")
    parser.add_argument("--config", required=True, help="VLA-RL recipe path.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--target-successes", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, list(args.overrides))
    validate_pld_cfg(cfg)

    collect_cfg = cfg.get("collect", {})
    target_successes = int(args.target_successes or collect_cfg.get("target_successes", 50))
    max_attempts = int(args.max_attempts or collect_cfg.get("max_attempts", 1000))
    output_dir = Path(args.output_dir or collect_cfg.get("output_dir", "outputs/pld_base_success"))
    output_dir.mkdir(parents=True, exist_ok=True)

    env = create_env(cfg)
    reference_policy = create_reference_policy(cfg)
    pld_obs_builder = create_pld_obs_builder(cfg)
    gamma = float(cfg.runtime.get("gamma", 0.99))
    execute_horizon = int(cfg.runtime.execute_horizon)
    action_dim = int(cfg.algorithm.action_dim)
    successes = 0
    attempts = 0
    steps_written = 0
    start = time.time()
    try:
        while successes < target_successes and attempts < max_attempts:
            obs = env.reset()
            episode: list[Transition] = []
            episode_success = False
            while True:
                base_actions = predict_base_actions(reference_policy, obs, horizon=execute_horizon, action_dim=action_dim)
                pld_obs = build_pld_obs(obs, base_actions, builder=pld_obs_builder)
                execute_actions = base_actions
                next_obs, reward, done, truncated, info = env.step_chunk(execute_actions)
                executed_steps = int(info.get("executed_steps", min(execute_horizon, len(base_actions))))
                terminal = bool(done or truncated)
                step_success = bool(
                    info.get("success", False) or info.get("env_done", False) or info.get("is_success", False)
                )
                # Success-only critic terminal (truncation still bootstraps).
                critic_terminal = step_success
                episode_success = bool(episode_success or step_success)
                next_pld_obs = None
                if not critic_terminal:
                    next_base_actions = predict_base_actions(reference_policy, next_obs, horizon=execute_horizon, action_dim=action_dim)
                    next_pld_obs = build_pld_obs(next_obs, next_base_actions, builder=pld_obs_builder)
                episode.append(
                    Transition(
                        obs=pld_obs,
                        next_obs=next_pld_obs,
                        action=np.asarray(execute_actions, dtype=np.float32).reshape(-1),
                        reward=float(reward),
                        done=bool(done),
                        truncated=bool(truncated),
                        discount=0.0 if critic_terminal else gamma**executed_steps,
                        executed_steps=executed_steps,
                        env_steps=steps_written + len(episode) + 1,
                        info={**dict(info), "critic_terminal": bool(critic_terminal)},
                    )
                )
                obs = next_obs
                if terminal:
                    break
            attempts += 1
            if episode_success and episode:
                _attach_mc_returns(episode, gamma=gamma)
                write_pld_offline_episode(
                    output_dir,
                    successes,
                    episode,
                    manifest_stats={
                        "target_successes": target_successes,
                        "attempts": attempts,
                        "successes": successes + 1,
                        "steps_written": steps_written + len(episode),
                    },
                )
                successes += 1
                steps_written += len(episode)
            print(json.dumps({"attempt": attempts, "success": episode_success, "successes": successes, "steps_written": steps_written}))
    finally:
        env.close()
        reference_policy.close()
    summary = {
        "output_dir": str(output_dir),
        "target_successes": target_successes,
        "attempts": attempts,
        "successes": successes,
        "steps_written": steps_written,
        "elapsed_sec": time.time() - start,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


def _attach_mc_returns(transitions: list[Transition], gamma: float) -> None:
    running = 0.0
    for transition in reversed(transitions):
        running = float(transition.reward) + float(gamma) * float(transition.discount > 0.0) * running
        transition.info["mc_returns"] = float(running)
        transition.info["mc_returns_valid"] = True


if __name__ == "__main__":
    main()
