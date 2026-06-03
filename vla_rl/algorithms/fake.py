from __future__ import annotations

import numpy as np

from vla_rl.algorithms.base import Algorithm
from vla_rl.data import ActionChunk, Observation, PolicyFeatures, RolloutBatch


class FakeAlgorithm(Algorithm):
    def __init__(self, action_dim: int = 7, chunk_size: int = 4, seed: int = 2) -> None:
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self._rng = np.random.default_rng(seed)
        self.update_count = 0

    def act(
        self,
        obs: Observation,
        features: PolicyFeatures | None = None,
        agent_obs: dict | None = None,
        deterministic: bool = False,
    ) -> ActionChunk:
        del obs, agent_obs
        if features is not None and features.reference_actions is not None:
            actions = features.reference_actions[: self.chunk_size, : self.action_dim].astype(np.float32, copy=True)
        elif deterministic:
            actions = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        else:
            actions = self._rng.uniform(-0.25, 0.25, size=(self.chunk_size, self.action_dim)).astype(np.float32)
        chunk = ActionChunk(actions=actions, horizon=min(self.chunk_size, actions.shape[0]), metadata={"source": "fake_algorithm"})
        chunk.validate()
        return chunk

    def update(self, batch: RolloutBatch) -> dict:
        batch.validate()
        self.update_count += 1
        rewards = np.array([transition.reward for transition in batch.transitions], dtype=np.float32)
        return {
            "updates": self.update_count,
            "batch_size": len(batch.transitions),
            "mean_reward": float(rewards.mean()),
        }

    def state_dict(self) -> dict:
        return {
            "update_count": self.update_count,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        self.update_count = int(state.get("update_count", 0))
        if "rng_state" in state:
            self._rng.bit_generator.state = state["rng_state"]

    def policy_state_dict(self) -> dict:
        return {}

    def load_policy_state_dict(self, state: dict) -> None:
        if state:
            raise ValueError("FakeAlgorithm policy state must be empty")
