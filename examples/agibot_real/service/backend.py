from __future__ import annotations

from typing import Any

import numpy as np

from vla_rl.data import ActionSpec, Observation, ObservationSpec
from vla_rl.envs.base import EnvBackend

from .env.arm_layout import AGIBOT_ROBOT_ACTION_DIM, normalize_arm_layout, get_arm_layout_spec
from .env.fake_task_env import AgiBotFakeTaskEnv
from .env.observation import build_agibot_layout_state, extract_agibot_policy_images, extract_agibot_residual_images


class AgiBotEnvBackend(EnvBackend):
    """VLA-RL wrapper around AgiBot real or fake task env."""

    def __init__(
        self,
        *,
        task_name: str,
        prompt: str,
        backend: str = "fake",
        arm_layout: str = "dual_arm",
        action_dim: int | None = None,
        robot_action_dim: int = AGIBOT_ROBOT_ACTION_DIM,
        image_mode: str = "residual",
        image_keys: tuple[str, ...] = ("image_rgb_0", "image_rgb_1", "image_rgb_2"),
        control_mode: str = "camera_position",
        hz: float = 30.0,
        max_episode_steps: int = 300,
        controller: dict[str, Any] | None = None,
        reset_hook: str | None = None,
        success_hook: str | None = None,
        assets_root: str | None = None,
        retargeter_urdf_path: str | None = None,
        retargeter_camera_extrinsic_path: str | None = None,
        fake_seed: int = 0,
    ) -> None:
        self.task_name = str(task_name)
        self.prompt = str(prompt)
        self.arm_layout = normalize_arm_layout(arm_layout)
        self.image_mode = str(image_mode)
        self.image_keys = tuple(image_keys)
        self._action_dim = int(action_dim or get_arm_layout_spec(self.arm_layout).action_dim)
        env_kwargs = {
            "task_name": self.task_name,
            "prompt": self.prompt,
            "arm_layout": self.arm_layout,
            "action_dim": self._action_dim,
            "robot_action_dim": int(robot_action_dim),
            "control_mode": control_mode,
            "hz": float(hz),
            "max_episode_steps": int(max_episode_steps),
            "assets_root": assets_root,
            "retargeter_urdf_path": retargeter_urdf_path,
            "retargeter_camera_extrinsic_path": retargeter_camera_extrinsic_path,
            "controller": controller,
            "reset_hook": reset_hook,
            "success_hook": success_hook,
        }
        if str(backend) == "fake":
            env_kwargs["fake_seed"] = int(fake_seed)
            self._env = AgiBotFakeTaskEnv(**env_kwargs)
        else:
            self._env = _real_env_cls()(**env_kwargs)

    @property
    def raw_env(self) -> Any:
        return self._env

    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(image_keys=self.image_keys, proprio_shape=(self._action_dim,))

    def action_spec(self) -> ActionSpec:
        spec = ActionSpec(shape=(self._action_dim,), minimum=-np.inf, maximum=np.inf)
        spec.validate()
        return spec

    def reset(self, task: str | None = None) -> Observation:
        del task
        return agibot_raw_to_observation(
            self._env.reset(),
            task=self.prompt,
            arm_layout=self.arm_layout,
            image_mode=self.image_mode,
            image_keys=self.image_keys,
        )

    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        obs, reward, done, truncated, info = self._env.step(np.asarray(action, dtype=np.float32))
        return (
            agibot_raw_to_observation(
                obs,
                task=self.prompt,
                arm_layout=self.arm_layout,
                image_mode=self.image_mode,
                image_keys=self.image_keys,
            ),
            float(reward),
            bool(done),
            bool(truncated),
            dict(info),
        )

    def step_chunk(self, actions: np.ndarray) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        payload = self._env.step_chunk(np.asarray(actions, dtype=np.float32))
        info = dict(payload.get("info", {}))
        info["executed_steps"] = int(payload.get("num_steps", 0))
        return (
            agibot_raw_to_observation(
                payload["obs"],
                task=self.prompt,
                arm_layout=self.arm_layout,
                image_mode=self.image_mode,
                image_keys=self.image_keys,
            ),
            float(payload["reward_sum"]),
            bool(payload["done"]),
            bool(payload["truncated"]),
            info,
        )

    def close(self) -> None:
        self._env.close()

    def controller_meta(self) -> dict[str, Any]:
        return dict(self._env.get_controller_meta())


def agibot_raw_to_observation(
    raw_obs: dict[str, Any],
    *,
    task: str,
    arm_layout: str,
    image_mode: str = "residual",
    image_keys: tuple[str, ...] = ("image_rgb_0", "image_rgb_1", "image_rgb_2"),
) -> Observation:
    if str(image_mode) == "policy":
        images = extract_agibot_policy_images(raw_obs)
    else:
        images = extract_agibot_residual_images(raw_obs, image_keys=tuple(image_keys))
    proprio = build_agibot_layout_state(raw_obs, arm_layout=arm_layout)
    policy_obs = build_agibot_reference_observation(raw_obs, task=str(task), arm_layout=arm_layout)
    return Observation(
        images={key: np.asarray(value) for key, value in images.items()},
        proprio=np.asarray(proprio, dtype=np.float32).reshape(-1),
        task=str(task),
        raw={
            "agibot_raw_obs": raw_obs,
            "reference_policy_observation": policy_obs,
            "openpi_observation": policy_obs,
        },
    )


def build_agibot_reference_observation(raw_obs: dict[str, Any], *, task: str, arm_layout: str) -> dict[str, Any]:
    policy_images = extract_agibot_policy_images(raw_obs)
    state = build_agibot_layout_state(raw_obs, arm_layout=arm_layout)
    return {
        "observation/image": policy_images["image_rgb_0"],
        "observation/wrist_left_image": policy_images["image_rgb_1"],
        "observation/wrist_right_image": policy_images["image_rgb_2"],
        "observation/state": np.asarray(state, dtype=np.float32),
        "prompt": str(task),
    }


def _real_env_cls():
    from .env.task_env import AgiBotTaskEnv

    return AgiBotTaskEnv
