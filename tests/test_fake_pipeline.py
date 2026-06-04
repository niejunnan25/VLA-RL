import json
from pathlib import Path

import numpy as np

from vla_rl.algorithms.fake import FakeAlgorithm
from vla_rl.data import RolloutBatch, Transition
from vla_rl.envs.fake import FakeEnvBackend
from examples.fake_debug.local_actor_learner import LocalActorLearnerRunner
from vla_rl.policies.fake import FakePolicyBackend
from examples.fake_debug.local_runner import LocalRunner


def test_fake_components_individually():
    env = FakeEnvBackend(max_steps=2)
    policy = FakePolicyBackend()
    algorithm = FakeAlgorithm()

    obs = env.reset("pick")
    reference = policy.sample_actions(obs)
    features = policy.extract_features(obs, reference)
    action_chunk = algorithm.act(obs, features)
    next_obs, reward, done, truncated, info = env.step(action_chunk.actions[0])

    assert reference.actions.shape == (4, 7)
    assert action_chunk.actions.shape == (4, 7)
    assert isinstance(reward, float)
    assert done is False
    assert truncated is False
    assert info["env_step"] == 1

    transition = Transition(
        obs=obs,
        action=np.zeros((7,), dtype=np.float32),
        reward=reward,
        next_obs=next_obs,
        done=done,
        truncated=truncated,
        discount=0.99,
    )
    metrics = algorithm.update(RolloutBatch(transitions=[transition]))
    assert metrics["updates"] == 1
    assert metrics["batch_size"] == 1


def test_local_runner_writes_metrics(tmp_path: Path):
    env = FakeEnvBackend(max_steps=10)
    policy = FakePolicyBackend()
    algorithm = FakeAlgorithm()
    metrics_path = tmp_path / "metrics.jsonl"

    runner = LocalRunner(
        env=env,
        policy=policy,
        algorithm=algorithm,
        max_steps=10,
        metrics_path=str(metrics_path),
    )
    summary = runner.run()

    assert summary["steps"] == 10
    assert summary["updates"] == 10
    lines = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert len(lines) == 11
    assert lines[-1]["summary"]["steps"] == 10


def test_local_actor_learner_runner_uses_replay(tmp_path: Path):
    env = FakeEnvBackend(max_steps=10)
    policy = FakePolicyBackend()
    algorithm = FakeAlgorithm()
    metrics_path = tmp_path / "actor_learner.jsonl"

    runner = LocalActorLearnerRunner(
        env=env,
        policy=policy,
        algorithm=algorithm,
        max_env_steps=10,
        max_update_steps=10,
        execute_horizon=2,
        batch_size=1,
        metrics_path=str(metrics_path),
    )
    summary = runner.run()

    assert summary["env_steps"] == 10
    assert summary["update_steps"] == 10
    assert summary["replay_size"] == 5
    lines = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert lines[-1]["summary"]["replay_size"] == 5
