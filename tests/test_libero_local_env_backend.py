from __future__ import annotations

import sys
import types

import numpy as np

from vla_rl.envs.libero import LiberoLocalEnvBackend


def _raw_obs(value: int = 0) -> dict:
    image = np.full((4, 4, 3), value, dtype=np.uint8)
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image,
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }


class FakeLiberoTaskEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.current_instruction = "fake instruction"
        self.task_description = "fake instruction"
        self.step_limit = 220
        self.take_action_cnt = 0
        self.action_dim = int(kwargs["action_dim"])
        self.last_seed = None
        self.current_init_state_idx = None
        self.closed = False

    def reset(self, *, seed, init_episode_idx):
        self.last_seed = int(seed)
        self.current_init_state_idx = int(init_episode_idx)
        self.take_action_cnt = 0
        return _raw_obs(1)

    def step(self, action):
        assert np.asarray(action).shape == (self.action_dim,)
        self.take_action_cnt += 1
        return _raw_obs(2), 0.5, False, False, {"success": False}

    def step_chunk(self, actions):
        actions = np.asarray(actions)
        observations = []
        rewards = []
        dones = []
        infos = []
        steps = []
        for idx, action in enumerate(actions):
            assert action.shape == (self.action_dim,)
            prev_obs = _raw_obs(idx)
            next_obs = _raw_obs(idx + 1)
            observations.append(next_obs)
            rewards.append(0.25)
            dones.append(False)
            infos.append({"success": False, "idx": idx})
            steps.append(
                {
                    "obs": prev_obs,
                    "action": action,
                    "next_obs": next_obs,
                    "reward": 0.25,
                    "env_done": False,
                    "truncated": False,
                    "done": False,
                    "info": infos[-1],
                }
            )
            self.take_action_cnt += 1
        return {
            "steps": steps,
            "obs": observations[-1],
            "observations": observations,
            "reward_sum": float(sum(rewards)),
            "rewards": rewards,
            "dones": dones,
            "done": bool(dones[-1]),
            "truncated": False,
            "infos": infos,
            "info": infos[-1],
            "num_steps": len(rewards),
        }

    def close(self, clear_cache=False):
        self.closed = True


def install_fake_serl_torch(monkeypatch):
    module_names = [
        "serl_torch",
        "serl_torch.examples",
        "serl_torch.examples.libero",
        "serl_torch.examples.libero.env",
        "serl_torch.examples.libero.env.task_env",
    ]
    for name in module_names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["serl_torch.examples.libero.env.task_env"].LiberoTaskEnv = FakeLiberoTaskEnv


def test_libero_local_env_backend_wraps_serl_torch_task_env(monkeypatch):
    install_fake_serl_torch(monkeypatch)

    backend = LiberoLocalEnvBackend(
        task_suite_name="libero_spatial",
        task_id=0,
        action_dim=7,
        seed=123,
        image_size=4,
        serl_torch_root=None,
    )

    obs = backend.reset()
    assert obs.task == "fake instruction"
    assert obs.images["image_rgb_0"].shape == (4, 4, 3)
    assert backend.meta["current_init_state_idx"] == 0

    next_obs, reward, done, truncated, info = backend.step(np.zeros(7, dtype=np.float32))
    assert next_obs.task == "fake instruction"
    assert reward == 0.5
    assert done is False
    assert truncated is False
    assert info["success"] is False

    chunk_obs, reward_sum, done, truncated, chunk_info = backend.step_chunk(
        np.zeros((2, 7), dtype=np.float32),
        return_steps=True,
    )
    assert chunk_obs.task == "fake instruction"
    assert reward_sum == 0.5
    assert done is False
    assert truncated is False
    assert chunk_info["num_steps"] == 2
    assert chunk_info["observation_indices"] == [1, 2]
    assert len(chunk_info["observations"]) == 2

    backend.close()
    assert backend.env.closed is True
