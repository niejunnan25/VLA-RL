from pathlib import Path

import numpy as np
import pytest

from vla_rl.algorithms.pld import PLDObservationBuilder, PLDSACAgent, ResidualActionSpec, build_pld_obs
from vla_rl.algorithms.pld.modeling import PLDObsEncoder
from vla_rl.algorithms.pld.replay import load_pld_offline_replay, write_pld_offline_episode
from vla_rl.data import MixedReplaySampler, Observation, PolicyFeatures, ReplayBuffer, RolloutBatch, Transition
from vla_rl.envs.fake import FakeEnvBackend
from vla_rl.policies.fake import FakePolicyBackend
from vla_rl.runtime.run_utils import apply_actor_summary_file, read_actor_summary, send_actor_summary
from examples.libero.pld import config as pld_config
from examples.libero.pld.scripts import train as pld_train
from omegaconf import OmegaConf


def make_features(chunk_size: int = 3, action_dim: int = 4) -> PolicyFeatures:
    return PolicyFeatures(reference_actions=np.ones((chunk_size, action_dim), dtype=np.float32) * 0.2)


def make_obs() -> Observation:
    return Observation(
        images={
            "front": np.zeros((16, 16, 3), dtype=np.uint8),
            "wrist": np.ones((16, 16, 3), dtype=np.uint8) * 127,
        },
        proprio=np.ones((5,), dtype=np.float32),
    )


def make_pld_obs(value: float = 0.0) -> dict[str, np.ndarray]:
    builder = PLDObservationBuilder(image_keys=("front", "wrist"), action_dim=4, chunk_horizon=1, alpha=0.5)
    pld_obs = build_pld_obs(make_obs(), make_features(chunk_size=2, action_dim=4).reference_actions, builder=builder)
    pld_obs["proprio"] = np.ones((5,), dtype=np.float32) * value
    return pld_obs


def make_transition(done: bool = False) -> Transition:
    return Transition(
        obs=make_pld_obs(0.0),
        next_obs=None if done else make_pld_obs(1.0),
        action=np.zeros((4,), dtype=np.float32),
        reward=1.0,
        done=done,
        truncated=False,
        discount=0.0 if done else 0.99,
        executed_steps=1,
        env_steps=1,
        info={"mc_returns": 1.0, "mc_returns_valid": True},
    )


def make_agent() -> PLDSACAgent:
    return PLDSACAgent(
        image_keys=("front", "wrist"),
        proprio_dim=5,
        action_dim=4,
        chunk_horizon=1,
        alpha=0.5,
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
        image_feature_dim=8,
        encoder_hidden_dim=32,
        critic_actor_ratio=1,
        utd_ratio=1,
        cql_n_actions=2,
        device="cpu",
    )


def test_residual_action_spec_compose_mask_limits_and_gripper_clip():
    spec = ResidualActionSpec(
        full_action_dim=4,
        action_mask=(True, False, True, True),
        action_limits=(1.0, 1.0, 0.5, 1.0),
        alpha=0.5,
        clip_gripper=True,
        chunk_horizon=1,
    )
    final = spec.compose_chunk(
        base_action_chunk=np.asarray([[0.0, 0.1, 0.2, 0.9]], dtype=np.float32),
        residual_action=np.asarray([1.0, -1.0, 1.0], dtype=np.float32),
    )

    np.testing.assert_allclose(final, np.asarray([[0.5, 0.1, -0.05, 1.0]], dtype=np.float32), atol=1e-6)


def test_pld_observation_builder_preserves_image_shapes():
    builder = PLDObservationBuilder(image_keys=("front", "wrist"), action_dim=4, chunk_horizon=1, alpha=0.5)

    pld_obs = build_pld_obs(make_obs(), make_features(chunk_size=2, action_dim=4).reference_actions, builder=builder)

    assert pld_obs["image_front"].shape == (3, 16, 16)
    assert pld_obs["image_wrist"].shape == (3, 16, 16)
    assert pld_obs["base_action_chunk"].shape == (1, 4)
    assert pld_obs["alpha"].shape == (1,)


def test_pld_obs_encoder_can_match_serl_style_vector_projection_without_obs_projection():
    encoder = PLDObsEncoder(
        image_keys=("front", "wrist"),
        proprio_dim=5,
        action_dim=4,
        chunk_horizon=1,
        image_encoder_type="small",
        image_feature_dim=8,
        vector_latent_dim=6,
        project_obs=False,
    )
    batch = {
        key: np.expand_dims(value, axis=0)
        for key, value in make_pld_obs().items()
    }
    import torch

    torch_batch = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in batch.items()}
    out = encoder(torch_batch)

    assert out.shape == (1, 8 * 2 + 6)


def test_compact_replay_preserves_pld_image_shapes_and_mixed_sampling(tmp_path: Path):
    online = ReplayBuffer(capacity=8, seed=0)
    offline_dir = tmp_path / "offline"
    transition = Transition(
        obs=make_pld_obs(0.0),
        next_obs=make_pld_obs(1.0),
        action=np.zeros((4,), dtype=np.float32),
        reward=1.0,
        done=False,
        discount=0.99,
    )
    online.add(transition)
    write_pld_offline_episode(offline_dir, 0, [transition])
    offline, stats = load_pld_offline_replay(offline_dir, capacity=8)

    transition = online.sample(1).transitions[0]
    assert transition.obs["image_front"].shape == (3, 16, 16)
    assert stats["transitions_loaded"] == 1
    mixed = MixedReplaySampler(online, offline, offline_ratio=0.5).sample(4)
    assert mixed.mix == {"online": 2, "offline": 2}


def test_pld_agent_act_update_calql_and_terminal_no_bootstrap():
    agent = make_agent()
    action = agent.sample_action(make_pld_obs(), deterministic=True)
    assert action.shape == (1, 4)

    batch = RolloutBatch(transitions=[make_transition(False) for _ in range(2)])
    metrics = agent.update(batch)
    assert set(metrics) >= {"loss_critic", "loss_actor", "temperature", "updates"}

    calql = agent.update_critics_calql(batch, calql_alpha=1.0, calql_n_actions=2, calql_temperature=1.0)
    assert set(calql) >= {"critic_cql_penalty", "calql_bound_applied"}

    terminal_batch = RolloutBatch(transitions=[make_transition(True) for _ in range(2)])
    fb = agent._convert_batch(terminal_batch)
    assert float(fb["discount"].max()) == 0.0



def test_pld_train_config_uses_explicit_builders():
    cfg = OmegaConf.load("examples/libero/pld/configs/fake_pld_smoke.yaml")

    pld_config.validate_pld_cfg(cfg)
    agent = pld_config.create_pld_agent(cfg)
    builder = pld_config.create_pld_obs_builder(cfg)

    assert isinstance(agent, PLDSACAgent)
    assert isinstance(builder, PLDObservationBuilder)


def test_pld_config_validation_rejects_horizon_mismatch():
    cfg = OmegaConf.load("examples/libero/pld/configs/fake_pld_smoke.yaml")
    cfg.runtime.execute_horizon = 2

    with pytest.raises(ValueError, match="chunk_horizon must match runtime.execute_horizon"):
        pld_config.validate_pld_cfg(cfg)

def test_pld_agent_actor_updates_after_configured_critic_steps():
    agent = make_agent()
    agent.critic_actor_ratio = 2
    batch = RolloutBatch(transitions=[make_transition(False) for _ in range(2)])

    first = agent.update(batch)
    second = agent.update(batch)

    assert "loss_actor" not in first
    assert set(second) >= {"loss_critic", "loss_actor", "temperature", "updates"}


def test_pld_actor_weight_sync_roundtrip():
    agent = make_agent()
    clone = make_agent()
    clone.load_policy_state_dict(agent.policy_state_dict())

    a = agent.sample_action(make_pld_obs(), deterministic=True)
    b = clone.sample_action(make_pld_obs(), deterministic=True)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_libero_pld_train_helpers_validate_config():
    cfg = OmegaConf.create(
        {
            "algorithm": {"chunk_horizon": 1, "action_dim": 4},
            "pld_observation": {"chunk_horizon": 1, "action_dim": 4},
            "runtime": {"execute_horizon": 1},
        }
    )
    pld_config.validate_pld_cfg(cfg)


def test_libero_pld_train_offline_replay_optional():
    runtime = OmegaConf.create({"offline_replay_path": None, "require_offline": False})

    replay, stats = pld_train._load_offline_replay(runtime)

    assert replay is None
    assert stats == {"episodes_loaded": 0, "transitions_loaded": 0}


class FailingSummaryClient:
    def request(self, request_type, payload):
        raise RuntimeError(f"failed {request_type}")


def test_pld_actor_summary_failure_is_persisted_and_readable(tmp_path: Path):
    metrics = []
    summary = send_actor_summary(
        FailingSummaryClient(),
        "send-stats",
        {"role": "actor", "algorithm": "pld", "env_steps": 7, "episodes": 1},
        tmp_path,
        metrics.append,
    )

    assert summary["actor_summary_notified"] is False
    assert metrics[-1]["event"] == "actor_summary_send_failed"
    assert metrics[-1]["algorithm"] == "pld"
    persisted = read_actor_summary(tmp_path)
    assert persisted is not None
    assert persisted["env_steps"] == 7
    done, env_steps = apply_actor_summary_file(tmp_path, False, 0)
    assert done is True
    assert env_steps == 7


def test_pld_actor_summary_read_ignores_partial_json(tmp_path: Path):
    (tmp_path / "actor_summary.json").write_text("{")
    assert read_actor_summary(tmp_path) is None
