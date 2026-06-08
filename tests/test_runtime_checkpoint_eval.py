from pathlib import Path

import numpy as np

from tests.fakes import FakeAlgorithm
from vla_rl.algorithms.rlt import RLTAgent
from vla_rl.runtime import CheckpointManager


def test_rlt_state_dict_roundtrip_deterministic_action():
    agent = RLTAgent(
        z_rl_dim=16,
        action_dim=2,
        chunk_size=2,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        num_critics=2,
        device="cpu",
    )
    rlt_state = {
        "z_rl": np.ones((16,), dtype=np.float32),
        "reference_action": np.zeros((4,), dtype=np.float32),
        "proprio": np.zeros((3,), dtype=np.float32),
    }
    expected = agent.sample_action(rlt_state, deterministic=True)

    restored = RLTAgent(
        z_rl_dim=16,
        action_dim=2,
        chunk_size=2,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        num_critics=2,
        device="cpu",
    )
    restored.load_state_dict(agent.state_dict())
    actual = restored.sample_action(rlt_state, deterministic=True)

    np.testing.assert_allclose(actual, expected)


def test_rlt_policy_state_dict_roundtrip_deterministic_action():
    agent = RLTAgent(
        z_rl_dim=16,
        action_dim=2,
        chunk_size=2,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        num_critics=2,
        device="cpu",
    )
    rlt_state = {
        "z_rl": np.ones((16,), dtype=np.float32),
        "reference_action": np.zeros((4,), dtype=np.float32),
        "proprio": np.zeros((3,), dtype=np.float32),
    }
    expected = agent.sample_action(rlt_state, deterministic=True)

    restored = RLTAgent(
        z_rl_dim=16,
        action_dim=2,
        chunk_size=2,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        num_critics=2,
        device="cpu",
    )
    restored.load_policy_state_dict(agent.policy_state_dict())
    actual = restored.sample_action(rlt_state, deterministic=True)

    np.testing.assert_allclose(actual, expected)


def test_checkpoint_manager_saves_latest_and_loads_algorithm_state(tmp_path: Path):
    agent = FakeAlgorithm()
    agent.update_count = 7
    manager = CheckpointManager(tmp_path)

    path = manager.save(agent, env_steps=12, update_steps=9, episodes=2, total_reward=3.5, config={"x": 1})

    assert path.exists()
    assert manager.latest_path.exists()
    restored = FakeAlgorithm()
    payload = manager.load(manager.latest_path, restored)
    assert payload["env_steps"] == 12
    assert payload["update_steps"] == 9
    assert restored.update_count == 7
