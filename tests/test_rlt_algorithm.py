from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from vla_rl.algorithms.rlt import RLTAgent, RLTStateBuilder, RLTokenEncoder
from vla_rl.algorithms.rlt.features import load_frozen_rlt_encoder
from vla_rl.data import Observation, PolicyFeatures, RolloutBatch, Transition
from vla_rl.envs.fake import FakeEnvBackend
from vla_rl.policies.fake import FakePolicyBackend
from examples.fake_debug.local_actor_learner import LocalActorLearnerRunner
from examples.libero_rlt import train as rlt_train


def make_agent() -> RLTAgent:
    return RLTAgent(
        z_rl_dim=16,
        proprio_dim=3,
        action_dim=2,
        chunk_size=2,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        num_critics=2,
        policy_update_freq=1,
        utd_ratio=1,
        device="cpu",
    )


def make_rlt_state(value: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "z_rl": np.full((16,), value, dtype=np.float32),
        "reference_action": np.zeros((4,), dtype=np.float32),
        "proprio": np.zeros((3,), dtype=np.float32),
    }


def make_transition(done: bool = False, discount: float = 0.25) -> Transition:
    return Transition(
        obs=make_rlt_state(0.0),
        next_obs=None if done else make_rlt_state(1.0),
        action=np.zeros((4,), dtype=np.float32),
        reward=1.0,
        done=done,
        truncated=False,
        discount=discount,
        executed_steps=2,
        env_steps=2,
    )


def test_rl_token_encoder_shape():
    encoder = RLTokenEncoder(input_dim=8, rl_token_dim=8, num_layers=1, num_heads=2, ff_dim=16)
    z_rl = encoder(torch.zeros(2, 5, 8))

    assert z_rl.shape == (2, 8)


def test_rlt_encoder_checkpoint_loading_and_max_tokens(tmp_path: Path):
    encoder = RLTokenEncoder(input_dim=8, rl_token_dim=8, num_layers=1, num_heads=2, ff_dim=16)
    path = tmp_path / "encoder.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "config": {"rlt": {"input_dim": 8, "rl_token_dim": 8, "num_encoder_layers": 1, "num_heads": 2, "ff_dim": 16}},
        },
        path,
    )

    loaded = load_frozen_rlt_encoder(str(path), device="cpu", max_tokens=3)

    assert getattr(loaded, "max_tokens") == 3
    assert loaded(torch.zeros(1, 3, 8)).shape == (1, 8)


def test_rlt_state_builder_outputs_rlt_state():
    encoder = RLTokenEncoder(input_dim=8, rl_token_dim=8, num_layers=1, num_heads=2, ff_dim=16)
    state_builder = RLTStateBuilder(
        device="cpu",
        chunk_size=2,
        action_dim=2,
        max_tokens=3,
        encoder=encoder,
    )
    features = PolicyFeatures(
        reference_actions=np.ones((5, 4), dtype=np.float32),
        embeddings={"prefix": np.zeros((1, 6, 8), dtype=np.float32)},
        proprio=np.ones((3,), dtype=np.float32),
    )

    rlt_state = state_builder.process(Observation(), features)

    assert rlt_state["z_rl"].shape == (8,)
    assert rlt_state["reference_action"].shape == (4,)
    assert rlt_state["proprio"].shape == (3,)


def test_rlt_config_uses_chunk_size_without_execute_horizon():
    repo_root = Path(__file__).resolve().parents[1]

    cfg = OmegaConf.load(repo_root / "examples/libero_rlt/configs/libero_spatial_task4_openpi_rlt.yaml")
    assert "rlt" in cfg
    assert int(cfg.rlt.chunk_size) == 10
    assert "execute_horizon" not in cfg.runtime
    assert "execute_horizon" not in cfg.algorithm
    agent_cfg = dict(OmegaConf.to_container(cfg.algorithm, resolve=True))
    agent_cfg.pop("_target_", None)
    agent_cfg["device"] = "cpu"
    agent = RLTAgent(**agent_cfg)
    assert agent.chunk_size == cfg.rlt.chunk_size


def test_rlt_agent_act_and_update():
    agent = make_agent()
    actions = agent.sample_action(make_rlt_state(), deterministic=True)

    assert actions.shape == (2, 2)

    metrics = agent.update(RolloutBatch(transitions=[make_transition() for _ in range(4)]))

    assert set(metrics) >= {"loss_critic", "target_q_mean", "predicted_q_mean", "loss_actor", "bc_loss", "updates"}


def test_rlt_agent_uses_transition_discount_and_terminal_no_bootstrap():
    agent = make_agent()
    batch = RolloutBatch(transitions=[make_transition(done=True, discount=1.0) for _ in range(4)])
    fb = agent._convert_batch(batch)

    np.testing.assert_allclose(fb["discount"].cpu().numpy(), np.ones((4, 1), dtype=np.float32))
    _, target_q_mean, _ = agent._critic_step(fb)
    assert abs(target_q_mean - 1.0) < 1e-6


def test_local_actor_learner_with_rlt_cpu_small_model(tmp_path: Path):
    encoder = RLTokenEncoder(input_dim=16, rl_token_dim=16, num_layers=1, num_heads=4, ff_dim=32)
    processor = RLTStateBuilder(device="cpu", chunk_size=2, action_dim=7, max_tokens=3, encoder=encoder)
    env = FakeEnvBackend(action_dim=7, proprio_dim=8, max_steps=20)
    policy = FakePolicyBackend(action_dim=7, chunk_size=2, embedding_dim=16)
    agent = RLTAgent(
        z_rl_dim=16,
        proprio_dim=8,
        action_dim=7,
        chunk_size=2,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        policy_update_freq=1,
        device="cpu",
    )
    runner = LocalActorLearnerRunner(
        env=env,
        policy=policy,
        algorithm=agent,
        feature_processor=processor,
        max_env_steps=10,
        max_update_steps=10,
        execute_horizon=2,
        batch_size=1,
        metrics_path=str(tmp_path / "metrics.jsonl"),
    )

    summary = runner.run()

    assert summary["env_steps"] == 10
    assert summary["update_steps"] == 10
    assert summary["replay_size"] == 5


class FailingSummaryClient:
    def request(self, request_type, payload):
        raise RuntimeError(f"failed {request_type}")


def test_rlt_actor_summary_failure_is_persisted_and_readable(tmp_path: Path):
    metrics = []
    summary = rlt_train._send_actor_summary(
        FailingSummaryClient(),
        "send-stats",
        {"role": "actor", "env_steps": 12, "episodes": 1},
        tmp_path,
        metrics.append,
    )

    assert summary["actor_summary_notified"] is False
    assert metrics[-1]["event"] == "actor_summary_send_failed"
    persisted = rlt_train._read_actor_summary(tmp_path)
    assert persisted is not None
    assert persisted["env_steps"] == 12
    assert persisted["actor_summary_notified"] is False
    done, env_steps = rlt_train._apply_actor_summary_file(tmp_path, False, 0)
    assert done is True
    assert env_steps == 12


def test_rlt_actor_summary_read_ignores_partial_json(tmp_path: Path):
    (tmp_path / "actor_summary.json").write_text("{")
    assert rlt_train._read_actor_summary(tmp_path) is None



def test_rlt_agent_reads_action_mask_for_bc_loss():
    agent = make_agent()
    transition = make_transition()
    assert isinstance(transition.obs, dict)
    transition.obs["action_mask"] = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    fb = agent._convert_batch(RolloutBatch(transitions=[transition, transition]))

    np.testing.assert_allclose(
        fb["action_mask"].cpu().numpy(),
        np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    )
    metrics = agent.update(RolloutBatch(transitions=[transition, transition]))
    assert "bc_loss" in metrics
