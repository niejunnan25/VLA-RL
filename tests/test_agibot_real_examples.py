from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from examples.agibot_real.rlt.handover import HandoverLabelCollector
from examples.agibot_real.rlt.train import build_rlt_obs, pad_action_chunk
from vla_rl.algorithms.pld import PLDObservationBuilder, build_pld_obs
from vla_rl.algorithms.rlt import RLTokenEncoder
from vla_rl.data import PolicyFeatures
from examples.agibot_real.service import AgiBotEnvBackend
from examples.agibot_real.service.env.arm_layout import embed_logical_action, project_chunk_to_layout


def _fake_agibot_env(arm_layout: str, action_dim: int) -> AgiBotEnvBackend:
    return AgiBotEnvBackend(
        task_name="smoke",
        prompt="test prompt",
        backend="fake",
        arm_layout=arm_layout,
        action_dim=action_dim,
        controller={"enabled": False},
        image_mode="residual",
        max_episode_steps=5,
    )


def test_agibot_fake_env_backend_step_chunk() -> None:
    env = _fake_agibot_env("right_arm", 7)
    obs = env.reset()
    assert obs.proprio.shape == (7,)
    assert set(obs.images) == {"image_rgb_0", "image_rgb_1", "image_rgb_2"}
    assert "openpi_observation" in obs.raw
    next_obs, reward, done, truncated, info = env.step_chunk(np.zeros((2, 7), dtype=np.float32))
    assert next_obs.proprio.shape == (7,)
    assert info["executed_steps"] == 2
    assert isinstance(reward, float)
    assert not done
    assert not truncated
    env.close()


def test_agibot_arm_layout_projection() -> None:
    chunk = np.arange(28, dtype=np.float32).reshape(2, 14)
    right = project_chunk_to_layout(chunk, "right_arm")
    assert right.shape == (2, 7)
    embedded = embed_logical_action(right[0], np.zeros(14, dtype=np.float32), "right_arm")
    assert embedded.shape == (14,)
    assert np.allclose(embedded[7:], right[0])


def test_agibot_rlt_obs_and_action_padding() -> None:
    encoder = RLTokenEncoder(input_dim=4, rl_token_dim=8, num_layers=1, num_heads=2, ff_dim=16)
    encoder.max_tokens = None
    features = PolicyFeatures(
        reference_actions=np.ones((15, 7), dtype=np.float32),
        embeddings={"prefix": np.ones((1, 3, 4), dtype=np.float32)},
        proprio=np.zeros(7, dtype=np.float32),
    )
    obs = build_rlt_obs(features, rl_token_encoder=encoder, action_dim=7, chunk_size=15)
    assert obs["z_rl"].shape == (8,)
    assert obs["reference_action"].shape == (105,)
    padded = pad_action_chunk(np.ones((3, 7), dtype=np.float32), chunk_size=15, action_dim=7)
    assert padded.shape == (15, 7)
    assert np.allclose(padded[:3], 1.0)
    assert np.allclose(padded[3:], 0.0)


def test_agibot_pld_obs_builder() -> None:
    env = _fake_agibot_env("dual_arm", 14)
    obs = env.reset()
    builder = PLDObservationBuilder(image_keys=("image_rgb_0", "image_rgb_1"), action_dim=14, chunk_horizon=4, alpha=0.005)
    pld_obs = build_pld_obs(obs, np.zeros((4, 14), dtype=np.float32), builder=builder)
    assert pld_obs["base_action_chunk"].shape == (4, 14)
    assert pld_obs["proprio"].shape == (14,)
    assert pld_obs["image_image_rgb_0"].shape == (3, 224, 224)
    env.close()


def test_handover_label_collector(tmp_path) -> None:
    output = tmp_path / "labels.jsonl"
    collector = HandoverLabelCollector(output, lookahead=2)
    collector.submit_hidden(episode_id=0, step=0, mean_hidden=np.zeros(4, dtype=np.float32))
    collector.submit_hidden(episode_id=0, step=1, mean_hidden=np.ones(4, dtype=np.float32))
    collector.mark_handover(episode_id=0)
    lines = output.read_text().strip().splitlines()
    assert len(lines) == 2
    assert '"handover": 1' in lines[0]
