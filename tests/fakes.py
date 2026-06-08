from __future__ import annotations

import numpy as np

from vla_rl.algorithms.base import Algorithm
from vla_rl.data import ActionSpec, Observation, ObservationSpec, PolicyFeatures, RolloutBatch
from vla_rl.envs.base import EnvBackend
from vla_rl.policies.base import PolicyBackend


class FakeAlgorithm(Algorithm):
    def __init__(self, action_dim: int = 7, chunk_size: int = 4, seed: int = 2) -> None:
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self._rng = np.random.default_rng(seed)
        self.update_count = 0

    def sample_action(
        self,
        obs: Observation,
        features: PolicyFeatures | None = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        del obs
        if features is not None and features.reference_actions is not None:
            return features.reference_actions[: self.chunk_size, : self.action_dim].astype(np.float32, copy=True)
        if deterministic:
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        return self._rng.uniform(-0.25, 0.25, size=(self.chunk_size, self.action_dim)).astype(np.float32)

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


class FakeEnvBackend(EnvBackend):
    def __init__(
        self,
        action_dim: int = 7,
        proprio_dim: int = 8,
        max_steps: int = 10,
        image_shape: tuple[int, int, int] = (32, 32, 3),
        seed: int = 0,
    ) -> None:
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.max_steps = int(max_steps)
        self.image_shape = tuple(image_shape)
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._task: str | None = None

    def observation_spec(self) -> ObservationSpec:
        spec = ObservationSpec(image_keys=("front",), proprio_shape=(self.proprio_dim,))
        spec.validate()
        return spec

    def action_spec(self) -> ActionSpec:
        spec = ActionSpec(shape=(self.action_dim,), minimum=-1.0, maximum=1.0)
        spec.validate()
        return spec

    def reset(self, task: str | None = None) -> Observation:
        self._step = 0
        self._task = task or "fake_task"
        return self._make_obs()

    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"expected action shape {(self.action_dim,)}, got {action.shape}")
        self._step += 1
        reward = float(1.0 - np.clip(np.linalg.norm(action), 0.0, 1.0))
        done = self._step >= self.max_steps
        truncated = False
        info = {"env_step": self._step, "success": done and reward > 0.0}
        return self._make_obs(), reward, done, truncated, info

    def _make_obs(self) -> Observation:
        image = self._rng.integers(0, 256, size=self.image_shape, dtype=np.uint8)
        proprio = self._rng.normal(size=(self.proprio_dim,)).astype(np.float32)
        obs = Observation(images={"front": image}, proprio=proprio, task=self._task, raw={"env_step": self._step})
        obs.validate()
        return obs


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
