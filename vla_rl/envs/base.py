from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from vla_rl.data import ActionSpec, Observation, ObservationSpec


class EnvBackend(ABC):
    @abstractmethod
    def observation_spec(self) -> ObservationSpec:
        raise NotImplementedError

    @abstractmethod
    def action_spec(self) -> ActionSpec:
        raise NotImplementedError

    @abstractmethod
    def reset(self, task: str | None = None) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, bool, dict]:
        raise NotImplementedError

    def step_chunk(self, actions: np.ndarray, *, return_steps: bool = False) -> tuple[Observation, float, bool, bool, dict]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        total_reward = 0.0
        next_obs = None
        done = False
        truncated = False
        info: dict = {}
        observations: list[Observation] = []
        rewards: list[float] = []
        dones: list[bool] = []
        truncateds: list[bool] = []
        infos: list[dict] = []
        executed_steps = 0
        for action in actions:
            next_obs, reward, done, truncated, info = self.step(action)
            total_reward += float(reward)
            observations.append(next_obs)
            rewards.append(float(reward))
            dones.append(bool(done))
            truncateds.append(bool(truncated))
            infos.append(dict(info))
            executed_steps += 1
            if done or truncated:
                break
        if next_obs is None:
            raise ValueError("step_chunk requires at least one action")
        info = dict(info)
        info["executed_steps"] = executed_steps
        if return_steps:
            info["observations"] = observations
            info["rewards"] = rewards
            info["dones"] = dones
            info["truncateds"] = truncateds
            info["infos"] = infos
            info["num_steps"] = executed_steps
        return next_obs, float(total_reward), bool(done), bool(truncated), info

    def close(self) -> None:
        pass
