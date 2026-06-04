import numpy as np
import pytest

from vla_rl.data import ActionChunk, ActionSpec, Observation, PolicyFeatures, RolloutBatch, Transition
from vla_rl.data.replay import ReplayBuffer


def test_schema_validates_basic_objects():
    obs = Observation(
        images={"front": np.zeros((8, 8, 3), dtype=np.uint8)},
        proprio=np.zeros((4,), dtype=np.float32),
        task="task",
    )
    obs.validate()

    chunk = ActionChunk(actions=np.zeros((3, 2), dtype=np.float32), horizon=2)
    chunk.validate()

    features = PolicyFeatures(
        reference_actions=np.zeros((3, 2), dtype=np.float32),
        embeddings={"z": np.zeros((5,), dtype=np.float32)},
        proprio=np.zeros((4,), dtype=np.float32),
    )
    features.validate()

    transition = Transition(
        obs=obs,
        action=np.zeros((2,), dtype=np.float32),
        reward=1.0,
        next_obs=obs,
        done=False,
        truncated=False,
        discount=0.99,
    )
    RolloutBatch(transitions=[transition]).validate()
    assert transition.truncated is False


def test_action_chunk_rejects_invalid_horizon():
    chunk = ActionChunk(actions=np.zeros((2, 3), dtype=np.float32), horizon=3)
    with pytest.raises(ValueError, match="exceeds"):
        chunk.validate()


def test_action_spec_rejects_invalid_bounds():
    spec = ActionSpec(shape=(2,), minimum=1.0, maximum=1.0)
    with pytest.raises(ValueError, match="greater"):
        spec.validate()


def test_replay_buffer_samples_and_evicts_fifo():
    obs = Observation(proprio=np.zeros((4,), dtype=np.float32))
    replay = ReplayBuffer(capacity=2, seed=0)
    transitions = [
        Transition(
            obs=obs,
            action=np.full((2,), index, dtype=np.float32),
            reward=float(index),
            next_obs=obs,
            done=False,
            discount=0.99,
        )
        for index in range(3)
    ]
    replay.extend(transitions)

    assert len(replay) == 2
    batch = replay.sample(4)
    assert len(batch.transitions) == 4
    assert all(transition.reward in {1.0, 2.0} for transition in batch.transitions)


def test_replay_buffer_rejects_empty_sample():
    replay = ReplayBuffer(capacity=2)
    with pytest.raises(ValueError, match="empty"):
        replay.sample(1)
