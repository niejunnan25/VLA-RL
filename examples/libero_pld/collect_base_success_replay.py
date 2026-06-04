#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vla_rl.algorithms.pld import write_pld_offline_episode
from vla_rl.config import instantiate
from vla_rl.data import CompactTransition


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
    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    collect_cfg = cfg.get("collect", {})
    target_successes = int(args.target_successes or collect_cfg.get("target_successes", 50))
    max_attempts = int(args.max_attempts or collect_cfg.get("max_attempts", 1000))
    output_dir = Path(args.output_dir or collect_cfg.get("output_dir", "outputs/pld_base_success"))
    output_dir.mkdir(parents=True, exist_ok=True)

    env = instantiate(cfg.env)
    policy = instantiate(cfg.policy)
    feature_processor = instantiate(cfg.feature)
    gamma = float(cfg.runtime.get("gamma", 0.99))
    execute_horizon = int(cfg.runtime.get("execute_horizon", cfg.algorithm.get("chunk_horizon", 1)))
    successes = 0
    attempts = 0
    steps_written = 0
    start = time.time()
    try:
        while successes < target_successes and attempts < max_attempts:
            obs = env.reset()
            episode: list[CompactTransition] = []
            episode_success = False
            while True:
                features = policy.extract_features(obs)
                action_chunk = _reference_chunk(features.reference_actions)
                agent_obs = feature_processor.process(obs, features)
                next_obs, reward, done, truncated, info = env.step_chunk(action_chunk.actions[:execute_horizon])
                executed_steps = int(info.get("executed_steps", min(execute_horizon, len(action_chunk.actions))))
                terminal = bool(done or truncated)
                episode_success = bool(episode_success or info.get("success", False) or info.get("env_done", False))
                next_agent_obs = None
                if not terminal:
                    next_features = policy.extract_features(next_obs)
                    next_agent_obs = feature_processor.process(next_obs, next_features)
                episode.append(
                    CompactTransition(
                        agent_obs=agent_obs,
                        next_agent_obs=next_agent_obs,
                        action=np.asarray(action_chunk.actions[:execute_horizon], dtype=np.float32).reshape(-1),
                        reward=float(reward),
                        done=bool(done),
                        truncated=bool(truncated),
                        discount=0.0 if terminal else gamma**executed_steps,
                        executed_steps=executed_steps,
                        env_steps=steps_written + len(episode) + 1,
                        info=dict(info),
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
        close = getattr(env, "close", None)
        if callable(close):
            close()
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


def _reference_chunk(reference_actions: np.ndarray | None):
    from vla_rl.data import ActionChunk

    if reference_actions is None:
        raise ValueError("base success collection requires PolicyFeatures.reference_actions")
    actions = np.asarray(reference_actions, dtype=np.float32)
    chunk = ActionChunk(actions=actions, horizon=actions.shape[0], metadata={"source": "base_policy"})
    chunk.validate()
    return chunk


def _attach_mc_returns(transitions: list[CompactTransition], gamma: float) -> None:
    running = 0.0
    for transition in reversed(transitions):
        running = float(transition.reward) + float(gamma) * float(transition.discount > 0.0) * running
        transition.info["mc_returns"] = float(running)
        transition.info["mc_returns_valid"] = True


if __name__ == "__main__":
    main()
