from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from examples.libero.common.async_eval import start_async_eval_worker
from examples.libero.common import eval_queue
from examples.libero.rlpd.config import validate_rlpd_cfg
from examples.libero.rlpd.scripts.train_rlpd import _learner_should_stop
from vla_rl.algorithms.rlpd import SACAgent
from vla_rl.data import ReplayBuffer, Transition


def make_rlpd_obs(value: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "image_image_rgb_0": np.full((3, 16, 16), float(value), dtype=np.float32),
        "proprio": np.full((3,), float(value), dtype=np.float32),
    }


def make_transition(index: int) -> Transition:
    return Transition(
        obs=make_rlpd_obs(float(index)),
        next_obs=make_rlpd_obs(float(index + 1)),
        action=np.zeros((2,), dtype=np.float32),
        reward=0.0,
        done=False,
        truncated=False,
        discount=0.99,
        executed_steps=1,
        env_steps=index + 1,
        info={"critic_terminal": False},
    )


def make_agent(*, utd_ratio: int) -> SACAgent:
    return SACAgent(
        image_keys=("image_rgb_0",),
        proprio_dim=3,
        action_dim=2,
        image_encoder_type="small",
        image_feature_dim=8,
        vector_latent_dim=4,
        encoder_hidden_dim=16,
        actor_hidden_dims=(16,),
        critic_hidden_dims=(16,),
        utd_ratio=utd_ratio,
        critic_actor_ratio=1,
        drq_random_crop=True,
        drq_padding=2,
        device="cpu",
    )


def make_replay(batch_size: int = 4) -> ReplayBuffer:
    replay = ReplayBuffer(capacity=16, seed=0)
    for index in range(batch_size):
        replay.add(make_transition(index))
    return replay


def test_rlpd_agent_update_supports_utd_ratio_one() -> None:
    agent = make_agent(utd_ratio=1)
    replay = make_replay(batch_size=4)

    info = agent.update(replay.sample(4))
    action = agent.sample_action(make_rlpd_obs(), deterministic=True)

    assert info["critic_updates"] == 1.0
    assert info["actor_updates"] == 1.0
    assert action.shape == (1, 2)


def test_rlpd_agent_update_supports_high_utd_split() -> None:
    agent = make_agent(utd_ratio=2)
    replay = make_replay(batch_size=4)

    info = agent.update(replay.sample(4))

    assert info["critic_updates"] == 2.0
    assert info["actor_updates"] == 1.0


def test_rlpd_async_eval_allows_local_env_without_remote_env(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeProc:
        pass

    def fake_launch_async_eval_worker(*, cmd, worker_log_path, env=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return FakeProc(), worker_log_path.open("a", encoding="utf-8")

    monkeypatch.setattr("examples.libero.common.async_eval.launch_async_eval_worker", fake_launch_async_eval_worker)

    runtime = OmegaConf.create(
        {
            "async_eval": {
                "enabled": True,
                "every_episodes": 50,
                "worker_cuda_visible_devices": "2",
                "worker_mujoco_egl_device_id": "2",
            }
        }
    )
    async_eval = start_async_eval_worker(runtime, run_dir=tmp_path, algorithm="rlpd")
    try:
        assert async_eval.enabled is True
        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2"
        assert captured["env"]["MUJOCO_EGL_DEVICE_ID"] == "2"
    finally:
        if async_eval.worker_log_fp is not None:
            async_eval.worker_log_fp.close()


def test_rlpd_eval_queue_request_uses_local_env_without_env_url(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"placeholder")
    output_dir = tmp_path / "eval_run"
    seen = {}

    def fake_run_eval(cfg, *, checkpoint_path, episodes, output_dir, max_env_steps_per_episode, save_videos, log_wandb):
        seen["env_target"] = str(cfg.env._target_)
        seen["has_env_url"] = "url" in cfg.env
        seen["checkpoint_path"] = str(checkpoint_path)
        seen["episodes"] = episodes
        seen["output_dir"] = str(output_dir)
        return {"success_rate": 0.5, "avg_return": 1.0, "avg_length": 2.0, "episodes": episodes}

    base_cfg = OmegaConf.create(
        {
            "env": {"_target_": "vla_rl.envs.libero.LiberoLocalEnvBackend", "task_suite_name": "libero_spatial", "task_id": 0},
            "runtime": {"max_env_steps": 1, "batch_size": 1},
            "algorithm": {"action_dim": 7},
            "rlpd_observation": {"image_keys": ["image_rgb_0", "image_rgb_1"]},
        }
    )
    result = eval_queue._run_request(
        base_cfg,
        {
            "eval_index": 3,
            "train_episode_id": 50,
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(output_dir),
            "episodes": 4,
            "max_env_steps_per_episode": 1,
            "save_videos": False,
        },
        spec=eval_queue.EVAL_QUEUE_SPECS["rlpd"],
        run_eval=fake_run_eval,
    )

    assert result["status"] == "ok"
    assert result["eval/success_rate"] == 0.5
    assert seen["env_target"] == "vla_rl.envs.libero.LiberoLocalEnvBackend"
    assert seen["has_env_url"] is False
    assert seen["episodes"] == 4


def test_rlpd_config_rejects_removed_max_update_steps() -> None:
    cfg = OmegaConf.create({"runtime": {"max_update_steps": 1}})

    with pytest.raises(ValueError, match="max_update_steps"):
        validate_rlpd_cfg(cfg)


def test_rlpd_learner_stop_follows_actor_lifecycle() -> None:
    assert _learner_should_stop(actor_done=True)
    assert not _learner_should_stop(actor_done=False)
