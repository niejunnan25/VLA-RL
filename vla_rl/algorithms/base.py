from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vla_rl.data import ActionChunk, Observation, PolicyFeatures, RolloutBatch


class Algorithm(ABC):
    @abstractmethod
    def act(
        self,
        obs: Observation,
        features: PolicyFeatures | None = None,
        agent_obs: dict[str, Any] | None = None,
        deterministic: bool = False,
    ) -> ActionChunk:
        raise NotImplementedError

    @abstractmethod
    def update(self, batch: RolloutBatch) -> dict:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        if state:
            raise ValueError(f"{type(self).__name__} does not accept checkpoint state")

    def policy_state_dict(self) -> dict:
        return self.state_dict()

    def load_policy_state_dict(self, state: dict) -> None:
        self.load_state_dict(state)
