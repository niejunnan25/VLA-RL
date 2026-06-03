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

    def step_chunk(self, actions: np.ndarray) -> tuple[Observation, float, bool, bool, dict]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        total_reward = 0.0
        last_obs: Observation | None = None
        last_info: dict = {}
        done = False
        truncated = False
        executed_steps = 0
        for action in actions:
            result = self.step(action)
            if len(result) == 4:
                last_obs, reward, done, last_info = result
                truncated = bool(last_info.get("truncated", False))
            else:
                last_obs, reward, done, truncated, last_info = result
            total_reward += float(reward)
            executed_steps += 1
            if done or truncated:
                break
        if last_obs is None:
            raise ValueError("step_chunk requires at least one action")
        info = dict(last_info)
        info["executed_steps"] = executed_steps
        return last_obs, total_reward, bool(done), bool(truncated), info
