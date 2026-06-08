from __future__ import annotations

from typing import Any

import numpy as np

from vla_rl.data import ActionSpec, Observation, ObservationSpec
from vla_rl.envs.base import EnvBackend
from vla_rl.envs.libero.observation import LIBERO_IMAGE_KEYS, LIBERO_STATE_DIM, build_libero_observation
from vla_rl.runtime.remote_http import RemoteHttpRpcClient


class LiberoRemoteEnvBackend(EnvBackend):
    """Client-side LIBERO env backend over the VLA-RL pickle HTTP RPC protocol."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:23000",
        task_suite_name: str = "libero_spatial",
        task_id: int = 4,
        action_dim: int | None = 7,
        resolution: int = 256,
        num_steps_wait: int = 10,
        max_episode_steps: int | None = None,
        benchmark_root: str | None = None,
        libero_root: str | None = None,
        libero_config_dir: str | None = None,
        libero_datasets_root: str | None = None,
        seed: int | None = None,
        image_size: int = 224,
        timeout: float = 120.0,
        create_env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.url = url
        self.task_suite_name = task_suite_name
        self.task_id = int(task_id)
        self.benchmark_root = benchmark_root
        self.seed = seed
        self.image_size = int(image_size)
        self._episode_index = 0
        self.client = RemoteHttpRpcClient(url, timeout=timeout, retries=3, keep_alive=True)
        kwargs = dict(create_env_kwargs or {})
        kwargs.setdefault("suite_name", task_suite_name)
        kwargs.setdefault("task_id", self.task_id)
        kwargs.setdefault("env_seed", seed)
        if action_dim is not None:
            kwargs.setdefault("action_dim", int(action_dim))
        kwargs.setdefault("resolution", int(resolution))
        kwargs.setdefault("num_steps_wait", int(num_steps_wait))
        if max_episode_steps is not None:
            kwargs.setdefault("max_episode_steps", int(max_episode_steps))
        for key, value in {
            "libero_root": libero_root or benchmark_root,
            "libero_config_dir": libero_config_dir,
            "libero_datasets_root": libero_datasets_root,
        }.items():
            if value is not None:
                kwargs.setdefault(key, value)
        self.client.call("create_env", **kwargs)
        self.meta = self.client.call("get_meta")
        self._task = self._task_from_meta(self.meta)

    def observation_spec(self) -> ObservationSpec:
        spec = ObservationSpec(image_keys=LIBERO_IMAGE_KEYS, proprio_shape=(LIBERO_STATE_DIM,))
        spec.validate()
        return spec

    def action_spec(self) -> ActionSpec:
        action_dim = int(self.meta.get("action_dim", 7)) if isinstance(self.meta, dict) else 7
        spec = ActionSpec(shape=(action_dim,), minimum=-1.0, maximum=1.0)
        spec.validate()
        return spec

    def reset(self, task: str | None = None) -> Observation:
        response = self.client.call("reset", seed=self.seed, init_episode_idx=self._episode_index)
        self._episode_index += 1
        raw_obs = response["obs"] if isinstance(response, dict) and "obs" in response else response
        meta = response.get("meta", self.meta) if isinstance(response, dict) else self.meta
        self.meta = meta
        self._task = task or self._task_from_meta(meta)
        return build_libero_observation(raw_obs, task=self._task, image_size=self.image_size)

    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, bool, dict]:
        response = self.client.call("step", action=np.asarray(action, dtype=np.float32))
        raw_obs = response["obs"]
        reward = float(response.get("reward", 0.0))
        done = bool(response.get("done", False))
        truncated = bool(response.get("truncated", False))
        info = dict(response.get("info", {}))
        meta = response.get("meta", self.meta)
        self.meta = meta
        obs = build_libero_observation(raw_obs, task=self._task_from_meta(meta), image_size=self.image_size)
        return obs, reward, done, truncated, info

    def step_chunk(self, actions: np.ndarray, *, return_steps: bool = False) -> tuple[Observation, float, bool, bool, dict]:
        actions = np.asarray(actions, dtype=np.float32)
        response = self.client.call("step_chunk", actions=actions)
        meta = response.get("meta", self.meta)
        self.meta = meta
        task = self._task_from_meta(meta)
        self._task = task

        obs = build_libero_observation(response["obs"], task=task, image_size=self.image_size)
        info = dict(response["info"])
        info["executed_steps"] = int(response["num_steps"])
        if return_steps:
            info["observations"] = [
                build_libero_observation(raw_obs, task=task, image_size=self.image_size)
                for raw_obs in response["observations"]
            ]
            info["rewards"] = [float(item) for item in response["rewards"]]
            info["dones"] = [bool(item) for item in response["dones"]]
            info["truncateds"] = [bool(step.get("truncated", False)) for step in response["steps"]]
            info["infos"] = [dict(item) for item in response["infos"]]
            info["num_steps"] = int(response["num_steps"])
        return (
            obs,
            float(response["reward_sum"]),
            bool(response["done"]),
            bool(response["truncated"]),
            info,
        )

    def close(self, clear_cache: bool = False) -> None:
        try:
            self.client.call("close", clear_cache=clear_cache)
        finally:
            self.client.close()

    @staticmethod
    def _task_from_meta(meta: Any) -> str | None:
        if not isinstance(meta, dict):
            return None
        return meta.get("current_instruction") or meta.get("task_description")
