import numpy as np
import pytest

from vla_rl.data import Observation
from vla_rl.policies.openpi import OpenPIBackend


class MockOpenPIPolicy:
    def __init__(self):
        self.last_obs = None
        self.feature_calls = 0

    def sample_actions(self, obs, **kwargs):
        del kwargs
        self.last_obs = obs
        return np.ones((1, 3, 32), dtype=np.float32)

    def sample_actions_with_features(self, obs, **kwargs):
        del kwargs
        self.last_obs = obs
        self.feature_calls += 1
        return {
            "actions": np.ones((1, 3, 32), dtype=np.float32),
            "features": {
                "prefix": np.zeros((1, 5, 8), dtype=np.float32),
            },
        }


class NoFeaturePolicy:
    def sample_actions(self, obs, **kwargs):
        del obs, kwargs
        return np.zeros((2, 32), dtype=np.float32)


def make_obs():
    return Observation(
        images={"front": np.zeros((16, 16, 3), dtype=np.uint8)},
        proprio=np.zeros((8,), dtype=np.float32),
        task="open drawer",
    )


def test_openpi_sample_actions_returns_action_chunk():
    policy = MockOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    chunk = backend.sample_actions(make_obs())

    assert chunk.actions.shape == (3, 32)
    assert chunk.horizon == 3
    assert chunk.metadata["policy"] == "openpi"
    assert chunk.metadata["config_name"] == "pi05_libero"
    assert policy.last_obs["prompt"] == "open drawer"
    assert "image/front" in policy.last_obs
    assert "state" in policy.last_obs


def test_openpi_extract_features_returns_policy_features():
    policy = MockOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    features = backend.extract_features(make_obs())

    assert features.reference_actions.shape == (3, 32)
    assert features.embeddings["prefix"].shape == (1, 5, 8)
    assert "suffix" not in features.embeddings
    assert policy.feature_calls == 1
    assert features.metadata["checkpoint_path"] == "/tmp/ckpt"


def test_openpi_observation_raw_override():
    policy = MockOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    obs = Observation(raw={"openpi_observation": {"custom": True}})
    backend.sample_actions(obs)

    assert policy.last_obs == {"custom": True}


def test_openpi_missing_sample_actions_with_features_has_clear_error():
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=NoFeaturePolicy())
    with pytest.raises(RuntimeError, match="sample_actions_with_features"):
        backend.extract_features(make_obs())
