import numpy as np

from vla_rl.envs.libero.observation import build_libero_observation


def test_build_libero_observation_from_synthetic_raw_obs():
    raw = {
        "agentview_image": np.zeros((16, 20, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((12, 10, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.5], dtype=np.float32),
    }

    obs = build_libero_observation(raw, task="open drawer", image_size=32)

    assert obs.task == "open drawer"
    assert obs.proprio.shape == (8,)
    assert set(obs.images) == {"image_rgb_0", "image_rgb_1", "image_rgb_2"}
    assert obs.images["image_rgb_0"].shape == (32, 32, 3)
    assert obs.images["image_rgb_1"].shape == (32, 32, 3)
    openpi_obs = obs.raw["openpi_observation"]
    assert openpi_obs["prompt"] == "open drawer"
    assert openpi_obs["state"].shape == (8,)
    assert openpi_obs["images"]["image_rgb_0"].shape == (32, 32, 3)
    assert openpi_obs["image_mask"]["image_rgb_2"] is False
