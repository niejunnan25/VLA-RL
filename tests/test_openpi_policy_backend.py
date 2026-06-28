import numpy as np

from vla_rl.data import Observation
from vla_rl.policies.openpi import OpenPIBackend


class MockOpenPIPolicy:
    def __init__(self):
        self.last_obs = None
        self.last_feature_method = None

    def sample_actions(self, obs, **kwargs):
        del kwargs
        self.last_obs = obs
        return np.ones((1, 3, 32), dtype=np.float32)

    def predict_action_with_features(self, obs, **kwargs):
        del kwargs
        self.last_obs = obs
        self.last_feature_method = "policy_prior_prefix"
        return {
            "actions": np.ones((1, 3, 32), dtype=np.float32),
            "features": {
                "prefix": np.zeros((1, 5, 8), dtype=np.float32),
            },
        }

    def predict_action_with_self_conditioned_features(self, obs, **kwargs):
        del kwargs
        self.last_obs = obs
        self.last_feature_method = "self_conditioned_prefix"
        return {
            "actions": np.full((1, 3, 32), 2.0, dtype=np.float32),
            "features": {
                "prefix": np.ones((1, 5, 8), dtype=np.float32),
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


def test_openpi_sample_actions_returns_actions():
    policy = MockOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    actions = backend.sample_actions(make_obs())

    assert actions.shape == (3, 32)
    assert policy.last_obs["prompt"] == "open drawer"
    assert "image/front" in policy.last_obs
    assert "state" in policy.last_obs


def test_openpi_extract_features_returns_policy_features():
    policy = MockOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    obs = make_obs()
    actions = backend.sample_actions(obs)
    features = backend.extract_features(obs, actions=actions)

    assert features.reference_actions.shape == (3, 32)
    assert features.embeddings["prefix"].shape == (1, 5, 8)
    assert "suffix" not in features.embeddings
    assert features.metadata["policy"] == "openpi"
    assert features.metadata["config_name"] == "pi05_libero"
    assert features.metadata["checkpoint_path"] == "/tmp/ckpt"
    assert features.metadata["feature_source"] == "policy_prior_prefix"
    assert policy.last_feature_method == "policy_prior_prefix"


def test_openpi_extract_features_can_use_self_conditioned_source():
    policy = MockOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    features = backend.extract_features(make_obs(), feature_source="self_conditioned_prefix")

    assert features.reference_actions.shape == (3, 32)
    assert np.allclose(features.reference_actions, 2.0)
    assert np.allclose(features.embeddings["prefix"], 1.0)
    assert features.metadata["feature_source"] == "self_conditioned_prefix"
    assert policy.last_feature_method == "self_conditioned_prefix"


def test_openpi_observation_raw_override():
    policy = MockOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    obs = Observation(raw={"openpi_observation": {"custom": True}})
    backend.sample_actions(obs)

    assert policy.last_obs == {"custom": True}


def test_openpi_missing_predict_action_with_features_returns_action_only_features():
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=NoFeaturePolicy())
    features = backend.extract_features(make_obs())

    assert features.reference_actions.shape == (2, 32)
    assert features.embeddings == {}
    assert features.metadata["feature_mode"] == "reference_actions_only"
