import json
from pathlib import Path

import numpy as np

from vla_rl.algorithms.fake import FakeAlgorithm
from vla_rl.algorithms.rlt import RLTAgent
from vla_rl.envs.fake import FakeEnvBackend
from vla_rl.policies.fake import FakePolicyBackend
from examples.fake_debug.local_actor_learner import LocalActorLearnerRunner
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


def test_runner_writes_run_dir_outputs_and_checkpoints(tmp_path: Path):
    runner = LocalActorLearnerRunner(
        env=FakeEnvBackend(max_steps=20),
        policy=FakePolicyBackend(),
        algorithm=FakeAlgorithm(),
        max_env_steps=10,
        max_update_steps=10,
        execute_horizon=2,
        batch_size=1,
        run_dir=str(tmp_path),
        checkpoint_interval_env_steps=4,
        config_snapshot={"runtime": {"max_env_steps": 10}},
    )

    summary = runner.run()

    assert summary["env_steps"] == 10
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "checkpoints" / "latest.pt").exists()
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["summary"]["env_steps"] == 10


def test_runner_resume_from_checkpoint_continues_counters(tmp_path: Path):
    first = LocalActorLearnerRunner(
        env=FakeEnvBackend(max_steps=20),
        policy=FakePolicyBackend(),
        algorithm=FakeAlgorithm(),
        max_env_steps=6,
        max_update_steps=6,
        execute_horizon=2,
        batch_size=1,
        run_dir=str(tmp_path),
        checkpoint_interval_env_steps=6,
    )
    first.run()
    latest = tmp_path / "checkpoints" / "latest.pt"

    second = LocalActorLearnerRunner(
        env=FakeEnvBackend(max_steps=20),
        policy=FakePolicyBackend(),
        algorithm=FakeAlgorithm(),
        max_env_steps=10,
        max_update_steps=10,
        execute_horizon=2,
        batch_size=1,
        run_dir=str(tmp_path),
        resume_from=str(latest),
    )
    summary = second.run()

    assert summary["env_steps"] == 10
    assert summary["update_steps"] == 10


def test_runner_sync_eval_writes_eval_metrics(tmp_path: Path):
    runner = LocalActorLearnerRunner(
        env=FakeEnvBackend(max_steps=20),
        policy=FakePolicyBackend(),
        algorithm=FakeAlgorithm(),
        max_env_steps=10,
        max_update_steps=10,
        execute_horizon=2,
        batch_size=1,
        run_dir=str(tmp_path),
        eval_enabled=True,
        eval_env=FakeEnvBackend(max_steps=4),
        eval_interval_env_steps=4,
        eval_episodes=2,
        eval_deterministic=True,
    )

    runner.run()

    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    eval_metrics = [metric for metric in metrics if "eval/success_rate" in metric]
    assert eval_metrics
    assert set(eval_metrics[-1]) >= {"eval/success_rate", "eval/mean_return", "eval/mean_length"}
