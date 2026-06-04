import numpy as np

from vla_rl.data import ReplayBuffer, Transition
from vla_rl.runtime.agentlace import json_sanitize, make_agentlace_replay_store, make_trainer_config


class FakeAgentlace:
    class DataStoreBase:
        pass

    class TrainerConfig:
        def __init__(self, port_number, broadcast_port, request_types):
            self.port_number = port_number
            self.broadcast_port = broadcast_port
            self.request_types = request_types


def make_obs_dict(value: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "z_rl": np.full((16,), value, dtype=np.float32),
        "reference_action": np.zeros((4,), dtype=np.float32),
        "proprio": np.zeros((3,), dtype=np.float32),
    }


def test_transition_replay_samples_dict_obs_batch():
    replay = ReplayBuffer(capacity=8, seed=0)
    replay.add(
        Transition(
            obs=make_obs_dict(0.0),
            next_obs=make_obs_dict(1.0),
            action=np.zeros((4,), dtype=np.float32),
            reward=1.0,
            done=False,
            truncated=False,
            discount=0.99**2,
            executed_steps=2,
            env_steps=2,
            info={"source": "test"},
        )
    )

    batch = replay.sample(1)
    transition = batch.transitions[0]

    assert len(replay) == 1
    assert replay.latest_env_steps == 2
    np.testing.assert_allclose(transition.obs["z_rl"], np.zeros((16,), dtype=np.float32))
    np.testing.assert_allclose(transition.next_obs["z_rl"], np.ones((16,), dtype=np.float32))


def test_terminal_transition_has_no_next_obs():
    transition = Transition(
        obs=make_obs_dict(0.0),
        next_obs=None,
        action=np.zeros((4,), dtype=np.float32),
        reward=1.0,
        done=True,
        truncated=False,
        discount=0.0,
        executed_steps=2,
        env_steps=2,
    )

    transition.validate()

    assert transition.done
    assert transition.next_obs is None
    assert transition.discount == 0.0


def test_agentlace_replay_store_helper_inserts_into_replay():
    replay = ReplayBuffer(capacity=8, seed=0)
    store = make_agentlace_replay_store(FakeAgentlace, replay)
    transition = Transition(
        obs=make_obs_dict(0.0),
        next_obs=make_obs_dict(1.0),
        action=np.zeros((4,), dtype=np.float32),
        reward=1.0,
        done=False,
        truncated=False,
        discount=0.99,
        executed_steps=1,
        env_steps=1,
    )

    store.insert(transition.to_payload())

    assert len(store) == 1
    assert len(replay) == 1
    assert store.latest_data_id() == 1
    assert store.get_latest_data(0) == []


def test_trainer_config_and_json_sanitize_helpers():
    cfg = make_trainer_config(FakeAgentlace, 1234, 1235, ["send-stats"])
    assert cfg.port_number == 1234
    assert cfg.broadcast_port == 1235
    assert cfg.request_types == ["send-stats"]

    value = json_sanitize({"x": np.array([1, 2]), "y": np.float32(1.5)})
    assert value == {"x": [1, 2], "y": 1.5}
