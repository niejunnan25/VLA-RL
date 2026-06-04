from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from vla_rl.data import ActionSpec, Observation, PolicyFeatures


class PolicyBackend(ABC):
    @abstractmethod
    def action_spec(self) -> ActionSpec:
        raise NotImplementedError

    @abstractmethod
    def sample_actions(self, obs: Observation, task: str | None = None, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def extract_features(
        self,
        obs: Observation,
        actions: np.ndarray | None = None,
        **kwargs,
    ) -> PolicyFeatures:
        raise NotImplementedError
