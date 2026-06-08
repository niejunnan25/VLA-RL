import numpy as np

from tests.fakes import FakeAlgorithm
from vla_rl.data import RolloutBatch, Transition
from tests.fakes import FakeEnvBackend
from tests.fakes import FakePolicyBackend


def test_fake_components_individually():
    env = FakeEnvBackend(max_steps=2)
    policy = FakePolicyBackend()
    algorithm = FakeAlgorithm()

    obs = env.reset("pick")
    reference = policy.sample_actions(obs)
    features = policy.extract_features(obs, reference)
    actions = algorithm.sample_action(obs, features)
    next_obs, reward, done, truncated, info = env.step(actions[0])

    assert reference.shape == (4, 7)
    assert actions.shape == (4, 7)
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


def test_fake_env_step_chunk_returns_final_step_info():
    env = FakeEnvBackend(action_dim=2, max_steps=3)
    env.reset("pick")
    final_obs, reward, done, truncated, info = env.step_chunk(np.zeros((5, 2), dtype=np.float32))

    assert final_obs is not None
    assert isinstance(reward, float)
    assert done is True
    assert truncated is False
    assert info["executed_steps"] == 3
