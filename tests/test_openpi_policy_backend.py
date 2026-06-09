import numpy as np
import torch
import pytest
from types import SimpleNamespace

from vla_rl.data import Observation
from vla_rl.policies.openpi import OpenPIBackend
from vla_rl.policies.openpi.backend import _OpenPIBasePolicy


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


def test_openpi_missing_predict_action_with_features_has_clear_error():
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=NoFeaturePolicy())
    with pytest.raises(AttributeError, match="predict_action_with_features"):
        backend.extract_features(make_obs())



class MockBatchOpenPIPolicy:
    def __init__(self):
        self.calls = 0
        self.last_batch = None
        self.last_num_steps = None
        self.last_feature_source = None

    def infer_batch_features(self, observations, num_steps=10, feature_source="policy_prior_prefix"):
        self.calls += 1
        self.last_batch = observations
        self.last_num_steps = num_steps
        self.last_feature_source = feature_source
        return [
            SimpleNamespace(
                reference_actions=np.full((num_steps, 32), index + 1, dtype=np.float32),
                prefix=np.full((1, 5, 8), index, dtype=np.float32),
            )
            for index, _ in enumerate(observations)
        ]


def test_openpi_extract_batch_features_uses_batch_policy_path():
    policy = MockBatchOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    features = backend.extract_batch_features(
        [make_obs(), make_obs()],
        feature_source="self_conditioned_prefix",
        num_steps=4,
    )

    assert policy.calls == 1
    assert policy.last_num_steps == 4
    assert policy.last_feature_source == "self_conditioned_prefix"
    assert len(policy.last_batch) == 2
    assert len(features) == 2
    assert np.allclose(features[0].reference_actions, 1.0)
    assert np.allclose(features[1].reference_actions, 2.0)
    assert features[0].embeddings["prefix"].shape == (1, 5, 8)
    assert features[1].metadata["feature_source"] == "self_conditioned_prefix"


def test_openpi_extract_batch_features_falls_back_to_single_policy_path():
    policy = MockOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    features = backend.extract_batch_features([make_obs(), make_obs()], feature_source="self_conditioned_prefix")

    assert len(features) == 2
    assert np.allclose(features[0].reference_actions, 2.0)
    assert np.allclose(features[1].embeddings["prefix"], 1.0)
    assert policy.last_feature_method == "self_conditioned_prefix"



class FakeOpenPIObservation:
    def __init__(self, data):
        self.data = data
        self.state = data["state"]

    @classmethod
    def from_dict(cls, data):
        return cls(data)


class FakeOpenPIModel:
    def __init__(self):
        self.calls = 0
        self.batch_shape = None
        self.image_shape = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def parameters(self):
        return []

    def predict_action_with_self_conditioned_features(self, device, observation, noise=None, num_steps=10):
        del noise
        self.calls += 1
        self.batch_shape = tuple(observation.state.shape)
        self.image_shape = tuple(observation.data["image"]["front"].shape)
        batch_size = observation.state.shape[0]
        return {
            "actions": torch.full((batch_size, num_steps, 32), 0.5, device=device),
            "features": {"prefix": torch.ones((batch_size, 5, 8), device=device)},
        }


class FakeTransformedOpenPIPolicy:
    def __init__(self):
        self._model = FakeOpenPIModel()

    def _input_transform(self, raw_obs):
        return {
            "image": {"front": np.asarray(raw_obs["image/front"], dtype=np.float32)},
            "state": np.asarray(raw_obs["state"], dtype=np.float32),
        }

    def _output_transform(self, out):
        return {"actions": out["actions"].numpy()}


def test_openpi_base_policy_batches_nested_observations_in_one_forward():
    policy = FakeTransformedOpenPIPolicy()
    base = _OpenPIBasePolicy(policy=policy, observation_cls=FakeOpenPIObservation, device="cpu")
    batches = base.infer_batch_features(
        [
            {"image/front": np.zeros((4, 4, 3), dtype=np.float32), "state": np.zeros((8,), dtype=np.float32)},
            {"image/front": np.ones((4, 4, 3), dtype=np.float32), "state": np.ones((8,), dtype=np.float32)},
        ],
        num_steps=4,
        feature_source="self_conditioned_prefix",
    )

    assert policy._model.calls == 1
    assert policy._model.batch_shape == (2, 8)
    assert policy._model.image_shape == (2, 3, 4, 4)
    assert len(batches) == 2
    assert batches[0].reference_actions.shape == (4, 32)
    assert batches[0].prefix.shape == (1, 5, 8)
    assert np.allclose(batches[1].reference_actions, 0.5)
