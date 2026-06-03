from __future__ import annotations

import numpy as np

from vla_rl.data import ActionSpec, Observation, ObservationSpec
from vla_rl.envs.base import EnvBackend


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
