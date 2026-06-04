from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vla_rl.algorithms import Algorithm
from vla_rl.data import RolloutBatch, Transition
from vla_rl.envs import EnvBackend
from vla_rl.policies import PolicyBackend
from vla_rl.runtime.base import Runner


class LocalRunner(Runner):
    def __init__(
        self,
        env: EnvBackend,
        policy: PolicyBackend,
        algorithm: Algorithm,
        max_steps: int = 10,
        gamma: float = 0.99,
        metrics_path: str | None = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.algorithm = algorithm
        self.max_steps = int(max_steps)
        self.gamma = float(gamma)
        self.metrics_path = Path(metrics_path) if metrics_path else None

    def run(self) -> dict:
        transitions: list[Transition] = []
        obs = self.env.reset()
        total_reward = 0.0
        update_metrics: dict = {}

        if self.metrics_path is not None:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.write_text("")

        for step in range(self.max_steps):
            reference = self.policy.sample_actions(obs, task=obs.task)
            features = self.policy.extract_features(obs, actions=reference)
            actions = np.asarray(self.algorithm.sample_action(obs, features=features), dtype=np.float32)
            action = actions[0]
            next_obs, reward, done, truncated, info = self.env.step(action)
            transition = Transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                discount=0.0 if done else self.gamma,
                truncated=truncated,
                info=info,
            )
            transition.validate()
            transitions.append(transition)
            total_reward += reward
            update_metrics = self.algorithm.update(RolloutBatch(transitions=[transition]))
            self._write_metric(
                {
                    "step": step + 1,
                    "reward": reward,
                    "done": done,
                    "truncated": truncated,
                    "total_reward": total_reward,
                    "updates": update_metrics.get("updates", 0),
                }
            )
            obs = next_obs
            if done or truncated:
                break

        summary = {
            "steps": len(transitions),
            "total_reward": total_reward,
            "updates": update_metrics.get("updates", 0),
        }
        self._write_metric({"summary": summary})
        return summary

    def _write_metric(self, metric: dict) -> None:
        if self.metrics_path is None:
            return
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(metric, sort_keys=True) + "\n")
