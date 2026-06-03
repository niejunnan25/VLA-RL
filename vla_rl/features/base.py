from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from vla_rl.data import Observation, PolicyFeatures


class FeatureProcessor(ABC):
    @abstractmethod
    def process(
        self,
        obs: Observation,
        features: PolicyFeatures,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError
