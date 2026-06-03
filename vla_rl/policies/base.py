from __future__ import annotations

from abc import ABC, abstractmethod

from vla_rl.data import ActionChunk, ActionSpec, Observation, PolicyFeatures


class PolicyBackend(ABC):
    @abstractmethod
    def action_spec(self) -> ActionSpec:
        raise NotImplementedError

    @abstractmethod
    def sample_actions(self, obs: Observation, task: str | None = None, **kwargs) -> ActionChunk:
        raise NotImplementedError

    @abstractmethod
    def extract_features(
        self,
        obs: Observation,
        actions: ActionChunk | None = None,
        **kwargs,
    ) -> PolicyFeatures:
        raise NotImplementedError
