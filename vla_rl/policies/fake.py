from __future__ import annotations

import numpy as np

from vla_rl.data import ActionSpec, Observation, PolicyFeatures
from vla_rl.policies.base import PolicyBackend


class FakePolicyBackend(PolicyBackend):
    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 4,
        embedding_dim: int = 16,
        seed: int = 1,
    ) -> None:
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.embedding_dim = int(embedding_dim)
        self._rng = np.random.default_rng(seed)

    def action_spec(self) -> ActionSpec:
        spec = ActionSpec(shape=(self.action_dim,), minimum=-1.0, maximum=1.0)
        spec.validate()
        return spec

    def sample_actions(self, obs: Observation, task: str | None = None, **kwargs) -> np.ndarray:
        del obs, task, kwargs
        return self._rng.uniform(-0.5, 0.5, size=(self.chunk_size, self.action_dim)).astype(np.float32)

    def extract_features(
        self,
        obs: Observation,
        actions: np.ndarray | None = None,
        **kwargs,
    ) -> PolicyFeatures:
        del kwargs
        if actions is None:
            actions = self.sample_actions(obs)
        proprio = obs.proprio.copy() if obs.proprio is not None else None
        features = PolicyFeatures(
            reference_actions=np.asarray(actions, dtype=np.float32).copy(),
            embeddings={
                "fake_embedding": self._rng.normal(size=(self.embedding_dim,)).astype(np.float32),
                "prefix": self._rng.normal(size=(1, self.chunk_size, self.embedding_dim)).astype(np.float32),
            },
            proprio=proprio,
            metadata={"source": "fake_policy"},
        )
        features.validate()
        return features
