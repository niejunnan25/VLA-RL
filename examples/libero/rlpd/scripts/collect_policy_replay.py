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

from examples.libero.rlpd.config import create_env, create_rlpd_obs_builder, load_config, validate_rlpd_cfg
from examples.libero.rlpd.rollout import assert_single_executed_step, assert_single_step_actions, step_info_success
from vla_rl.algorithms.rlpd import build_rlpd_obs, write_offline_episode
from vla_rl.data import Transition
from vla_rl.policies import ReferencePolicyClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RLPD offline replay with a behavior policy.")
    parser.add_argument("--config", required=True, help="RLPD config path.")
    parser.add_argument("--output-dir", default=None, help="Output replay directory.")
    parser.add_argument("--policy-url", required=True, help="Reference/behavior policy server URL.")
    parser.add_argument("--target-successes", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, list(args.overrides))
    validate_rlpd_cfg(cfg)
    output_dir = Path(args.output_dir or cfg.runtime.offline_replay_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = create_env(cfg)
    obs_builder = create_rlpd_obs_builder(cfg)
    behavior_policy = ReferencePolicyClient(
        url=str(args.policy_url),
        action_dim=int(cfg.algorithm.action_dim),
        timeout=float(args.timeout),
    )
    gamma = float(cfg.runtime.gamma)
    successes = 0
    attempts = 0
    steps_written = 0
    start_time = time.perf_counter()
    try:
        while successes < int(args.target_successes) and attempts < int(args.max_attempts):
            obs = env.reset()
            episode: list[Transition] = []
            episode_success = False
            while True:
                direct_obs = build_rlpd_obs(obs, builder=obs_builder)
                actions = np.asarray(behavior_policy.sample_actions(obs), dtype=np.float32)
                action_input = actions if actions.ndim == 1 else actions[:1]
                action = assert_single_step_actions(action_input, context="behavior policy action")
                next_obs, reward, done, truncated, info = env.step_chunk(action)
                executed_steps = assert_single_executed_step(info, context="env.step_chunk")
                terminal = bool(done or truncated)
                step_success = step_info_success(info)
                critic_terminal = bool(terminal or step_success)
                episode_success = bool(episode_success or step_success)
                next_direct_obs = None
                if not terminal:
                    next_direct_obs = build_rlpd_obs(next_obs, builder=obs_builder)
                episode.append(
                    Transition(
                        obs=direct_obs,
                        next_obs=next_direct_obs,
                        action=np.asarray(action, dtype=np.float32).reshape(-1),
                        reward=float(reward),
                        done=bool(done),
                        truncated=bool(truncated),
                        discount=0.0 if critic_terminal else gamma,
                        executed_steps=executed_steps,
                        env_steps=steps_written + len(episode) + 1,
                        info={**dict(info), "critic_terminal": bool(critic_terminal), "behavior_policy": "reference_policy"},
                    )
                )
                obs = next_obs
                if terminal:
                    break
            attempts += 1
            if episode_success and episode:
                _attach_mc_returns(episode, gamma=gamma)
                write_offline_episode(
                    output_dir,
                    successes,
                    episode,
                    manifest_stats={
                        "target_successes": int(args.target_successes),
                        "attempts": attempts,
                        "successes": successes + 1,
                        "steps_written": steps_written + len(episode),
                        "behavior_policy_url": str(args.policy_url),
                    },
                )
                successes += 1
                steps_written += len(episode)
            print(json.dumps({"attempt": attempts, "success": episode_success, "successes": successes, "steps_written": steps_written}), flush=True)
    finally:
        env.close()
        behavior_policy.close()
    summary = {
        "output_dir": str(output_dir),
        "target_successes": int(args.target_successes),
        "attempts": attempts,
        "successes": successes,
        "steps_written": steps_written,
        "elapsed_sec": time.perf_counter() - start_time,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


def _attach_mc_returns(transitions: list[Transition], gamma: float) -> None:
    running = 0.0
    for transition in reversed(transitions):
        running = float(transition.reward) + float(gamma) * float(transition.discount > 0.0) * running
        transition.info["mc_returns"] = float(running)
        transition.info["mc_returns_valid"] = True


if __name__ == "__main__":
    main()
