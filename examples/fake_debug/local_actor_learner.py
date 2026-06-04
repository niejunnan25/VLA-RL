from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from vla_rl.algorithms import Algorithm
from vla_rl.data import Transition
from vla_rl.data.replay import ReplayBuffer
from vla_rl.envs import EnvBackend
from vla_rl.features import FeatureProcessor
from vla_rl.policies import PolicyBackend
from vla_rl.runtime.checkpoint import CheckpointManager


class LocalActorLearnerRunner:
    """Single-process actor-learner loop for interface integration tests."""

    def __init__(
        self,
        env: EnvBackend,
        policy: PolicyBackend,
        algorithm: Algorithm,
        max_env_steps: int = 1000,
        max_update_steps: int = 1000,
        warmup_steps: int = 0,
        execute_horizon: int = 1,
        batch_size: int = 1,
        replay_capacity: int = 100_000,
        gamma: float = 0.99,
        metrics_path: str | None = None,
        replay_seed: int = 0,
        feature_processor: FeatureProcessor | None = None,
        run_dir: str | None = None,
        checkpoint_interval_env_steps: int = 0,
        resume_from: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
        eval_env: EnvBackend | None = None,
        eval_policy: PolicyBackend | None = None,
        eval_feature_processor: FeatureProcessor | None = None,
        eval_enabled: bool = False,
        eval_interval_env_steps: int = 0,
        eval_episodes: int = 0,
        eval_deterministic: bool = True,
        eval_max_episode_steps: int | None = None,
        eval_metrics_key_prefix: str = "eval",
    ) -> None:
        self.env = env
        self.policy = policy
        self.algorithm = algorithm
        self.max_env_steps = int(max_env_steps)
        self.max_update_steps = int(max_update_steps)
        self.warmup_steps = int(warmup_steps)
        self.execute_horizon = int(execute_horizon)
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.replay = ReplayBuffer(capacity=replay_capacity, seed=replay_seed)
        self.run_dir = Path(run_dir) if run_dir else None
        self.metrics_path = Path(metrics_path) if metrics_path else (self.run_dir / "metrics.jsonl" if self.run_dir else None)
        self.feature_processor = feature_processor
        self.checkpoint_interval_env_steps = int(checkpoint_interval_env_steps)
        self.resume_from = resume_from
        self.config_snapshot = config_snapshot or {}
        self.checkpoints = CheckpointManager(self.run_dir) if self.run_dir else None
        self.eval_env = eval_env
        self.eval_policy = eval_policy or policy
        self.eval_feature_processor = eval_feature_processor if eval_feature_processor is not None else feature_processor
        self.eval_enabled = bool(eval_enabled)
        self.eval_interval_env_steps = int(eval_interval_env_steps)
        self.eval_episodes = int(eval_episodes)
        self.eval_deterministic = bool(eval_deterministic)
        self.eval_max_episode_steps = eval_max_episode_steps
        self.eval_metrics_key_prefix = str(eval_metrics_key_prefix)
        if self.execute_horizon <= 0:
            raise ValueError(f"execute_horizon must be positive, got {execute_horizon}")
        if self.eval_enabled and self.eval_env is None:
            raise ValueError("eval_enabled requires an independent eval_env")

    def run(self) -> dict:
        self._prepare_outputs()
        env_steps = 0
        update_steps = 0
        episodes = 0
        total_reward = 0.0
        if self.resume_from:
            payload = self._load_checkpoint(self.resume_from)
            env_steps = int(payload.get("env_steps", 0))
            update_steps = int(payload.get("update_steps", 0))
            episodes = int(payload.get("episodes", 0))
            total_reward = float(payload.get("total_reward", 0.0))

        if self.metrics_path is not None:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.resume_from:
                self.metrics_path.write_text("")

        obs = self.env.reset()
        last_update: dict = {}
        start_time = time.perf_counter()
        next_checkpoint_at = self._next_interval(env_steps, self.checkpoint_interval_env_steps)
        next_eval_at = self._next_interval(env_steps, self.eval_interval_env_steps) if self.eval_enabled else None

        while env_steps < self.max_env_steps:
            chunk_start = time.perf_counter()
            if self.feature_processor is None:
                reference = self.policy.sample_actions(obs, task=obs.task)
                features = self.policy.extract_features(obs, actions=reference)
            else:
                features = self.policy.extract_features(obs)
                reference = self._reference_chunk_from_features(features)
            agent_obs = self._process_features(obs, features)
            if env_steps < self.warmup_steps:
                action_chunk = reference
            else:
                action_chunk = self.algorithm.act(obs, features=features, agent_obs=agent_obs)

            execute_actions = action_chunk.actions[: self.execute_horizon]
            next_obs, reward, done, truncated, info = self.env.step_chunk(execute_actions)
            executed_steps = int(info.get("executed_steps", len(execute_actions)))
            env_steps += executed_steps
            total_reward += float(reward)
            terminal = bool(done or truncated)
            next_agent_obs = None
            if not terminal and self.feature_processor is not None:
                next_features = self.policy.extract_features(next_obs)
                next_agent_obs = self._process_features(next_obs, next_features)
            transition = Transition(
                obs=obs,
                action=np.asarray(action_chunk.actions, dtype=np.float32).reshape(-1),
                reward=float(reward),
                next_obs=next_obs,
                done=bool(done),
                truncated=bool(truncated),
                discount=0.0 if terminal else self.gamma**executed_steps,
                agent_obs=agent_obs,
                next_agent_obs=next_agent_obs,
                info={**info, "executed_steps": executed_steps},
            )
            self.replay.add(transition)

            while update_steps < self.max_update_steps and len(self.replay) >= self.batch_size:
                last_update = self.algorithm.update(self.replay.sample(self.batch_size))
                update_steps += 1
                if update_steps >= env_steps:
                    break

            metric = {
                "env_steps": env_steps,
                "update_steps": update_steps,
                "episode": episodes,
                "replay_size": len(self.replay),
                "reward": float(reward),
                "done": bool(done),
                "truncated": bool(truncated),
                "executed_steps": executed_steps,
                "algorithm_updates": last_update.get("updates", 0),
                "wall_time_sec": time.perf_counter() - start_time,
                "chunk_time_sec": time.perf_counter() - chunk_start,
                **{f"train/{key}": value for key, value in last_update.items()},
            }
            self._write_metric(metric)
            if terminal:
                episodes += 1
                obs = self.env.reset()
            else:
                obs = next_obs

            if next_eval_at is not None and env_steps >= next_eval_at:
                self._write_metric(self._run_eval(env_steps=env_steps, update_steps=update_steps))
                next_eval_at += self.eval_interval_env_steps

            if self.checkpoints is not None and self.checkpoint_interval_env_steps > 0 and env_steps >= next_checkpoint_at:
                self._save_checkpoint(env_steps, update_steps, episodes, total_reward)
                next_checkpoint_at += self.checkpoint_interval_env_steps

            if update_steps >= self.max_update_steps and env_steps >= self.max_env_steps:
                break

        summary = {
            "env_steps": env_steps,
            "update_steps": update_steps,
            "episodes": episodes,
            "replay_size": len(self.replay),
            "total_reward": total_reward,
        }
        if self.checkpoints is not None:
            self._save_checkpoint(env_steps, update_steps, episodes, total_reward, tag="final.pt")
        if self.run_dir is not None:
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        self._write_metric({"summary": summary})
        return summary

    def _write_metric(self, metric: dict) -> None:
        if self.metrics_path is None:
            return
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(json_sanitize(metric), sort_keys=True) + "\n")

    def _process_features(self, obs, features):
        if self.feature_processor is None:
            return None
        return self.feature_processor.process(obs, features)

    def _reference_chunk_from_features(self, features):
        if features.reference_actions is None:
            raise ValueError("feature_processor path requires PolicyFeatures.reference_actions for warmup/reference actions")
        from vla_rl.data import ActionChunk

        reference = ActionChunk(
            actions=np.asarray(features.reference_actions, dtype=np.float32),
            horizon=int(features.reference_actions.shape[0]),
            metadata={"source": "policy_features.reference_actions"},
        )
        reference.validate()
        return reference

    def _prepare_outputs(self) -> None:
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(self.config_snapshot), self.run_dir / "config.yaml")

    def _save_checkpoint(
        self,
        env_steps: int,
        update_steps: int,
        episodes: int,
        total_reward: float,
        tag: str | None = None,
    ) -> None:
        if self.checkpoints is None:
            return
        self.checkpoints.save(
            self.algorithm,
            env_steps=env_steps,
            update_steps=update_steps,
            episodes=episodes,
            total_reward=total_reward,
            config=self.config_snapshot,
            tag=tag,
        )

    def _load_checkpoint(self, path: str) -> dict:
        if self.checkpoints is None:
            self.checkpoints = CheckpointManager(Path(path).parent.parent)
        return self.checkpoints.load(path, self.algorithm)

    @staticmethod
    def _next_interval(current: int, interval: int) -> int:
        if interval <= 0:
            return 0
        return ((int(current) // int(interval)) + 1) * int(interval)

    def _run_eval(self, *, env_steps: int, update_steps: int) -> dict:
        if self.eval_env is None:
            raise ValueError("eval_env is required for synchronous eval")
        returns = []
        lengths = []
        successes = []
        for _ in range(max(1, self.eval_episodes)):
            obs = self.eval_env.reset()
            episode_return = 0.0
            episode_steps = 0
            done = False
            truncated = False
            info: dict[str, Any] = {}
            while not (done or truncated):
                if self.eval_feature_processor is None:
                    reference = self.eval_policy.sample_actions(obs, task=obs.task)
                    features = self.eval_policy.extract_features(obs, actions=reference)
                    agent_obs = None
                else:
                    features = self.eval_policy.extract_features(obs)
                    agent_obs = self.eval_feature_processor.process(obs, features)
                action_chunk = self.algorithm.act(
                    obs,
                    features=features,
                    agent_obs=agent_obs,
                    deterministic=self.eval_deterministic,
                )
                execute_actions = action_chunk.actions[: self.execute_horizon]
                obs, reward, done, truncated, info = self.eval_env.step_chunk(execute_actions)
                executed_steps = int(info.get("executed_steps", len(execute_actions)))
                episode_return += float(reward)
                episode_steps += executed_steps
                if self.eval_max_episode_steps is not None and episode_steps >= int(self.eval_max_episode_steps):
                    truncated = True
            returns.append(episode_return)
            lengths.append(episode_steps)
            successes.append(float(info.get("success", done and episode_return > 0.0)))
        prefix = self.eval_metrics_key_prefix
        return {
            "env_steps": env_steps,
            "update_steps": update_steps,
            f"{prefix}/success_rate": float(np.mean(successes)),
            f"{prefix}/mean_return": float(np.mean(returns)),
            f"{prefix}/mean_length": float(np.mean(lengths)),
            f"{prefix}/episodes": int(max(1, self.eval_episodes)),
        }


def json_sanitize(value):
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
