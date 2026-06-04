from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from examples.agibot_real.service import AgiBotEnvBackend


def create_agibot_env(cfg: DictConfig) -> AgiBotEnvBackend:
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
    robot_cfg = OmegaConf.to_container(cfg.robot, resolve=True)
    controller_cfg = OmegaConf.to_container(cfg.controller, resolve=True)
    return AgiBotEnvBackend(
        task_name=str(task_cfg["name"]),
        prompt=str(task_cfg["prompt"]),
        backend=str(env_cfg.get("backend", "fake")),
        arm_layout=str(env_cfg.get("arm_layout", "dual_arm")),
        action_dim=env_cfg.get("action_dim", None),
        robot_action_dim=int(env_cfg.get("robot_action_dim", 14)),
        image_mode=str(env_cfg.get("image_mode", "residual")),
        image_keys=tuple(env_cfg.get("image_keys", ("image_rgb_0", "image_rgb_1", "image_rgb_2"))),
        control_mode=str(task_cfg.get("control_mode", "camera_position")),
        hz=float(task_cfg.get("hz", 30.0)),
        max_episode_steps=int(task_cfg.get("max_episode_steps", 300)),
        controller=dict(controller_cfg),
        reset_hook=task_cfg.get("reset_hook", None),
        success_hook=task_cfg.get("success_hook", None),
        assets_root=robot_cfg.get("assets_root", None),
        retargeter_urdf_path=robot_cfg.get("retargeter_urdf_path", None),
        retargeter_camera_extrinsic_path=robot_cfg.get("retargeter_camera_extrinsic_path", None),
        fake_seed=int(env_cfg.get("fake_seed", 0)),
    )
