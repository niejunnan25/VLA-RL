import numpy as np
import pytest

from vla_rl.data import Observation
from vla_rl.policies.openpi import OpenPIBackend
from vla_rl.policies.openpi.backend import _OpenPIBasePolicy


class MockOpenPIPolicy:
    def __init__(self):
        self.last_obs = None
        self.last_feature_method = None

    def sample_actions(self, obs, **kwargs):
        self.last_feature_method = None
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


class InferOnlyPolicy:
    def __init__(self, actions):
        self.actions = np.asarray(actions, dtype=np.float32)
        self.last_obs = None
        self.infer_calls = 0

    def infer(self, obs):
        self.last_obs = obs
        self.infer_calls += 1
        return {"actions": self.actions.copy()}

    def infer_features(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("sample_actions must not use feature extraction")


class NoInferPolicy:
    pass


def make_openpi_base_policy(policy):
    wrapper = object.__new__(_OpenPIBasePolicy)
    wrapper.policy = policy
    return wrapper


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
    assert policy.last_feature_method is None


def test_openpi_base_policy_sample_actions_uses_policy_infer_and_slices():
    raw_actions = np.arange(12, dtype=np.float32).reshape(3, 4)
    policy = InferOnlyPolicy(raw_actions)
    wrapper = make_openpi_base_policy(policy)

    actions = wrapper.sample_actions({"prompt": "open drawer"}, num_steps=2)

    assert actions.shape == (2, 4)
    assert np.allclose(actions, raw_actions[:2])
    assert policy.infer_calls == 1
    assert policy.last_obs == {"prompt": "open drawer"}


def test_openpi_base_policy_sample_actions_rejects_feature_kwargs():
    policy = InferOnlyPolicy(np.zeros((3, 4), dtype=np.float32))
    wrapper = make_openpi_base_policy(policy)

    with pytest.raises(ValueError, match="only accepts num_steps"):
        wrapper.sample_actions({}, feature_source="policy_prior_prefix")


def test_openpi_base_policy_sample_actions_requires_policy_infer():
    wrapper = make_openpi_base_policy(NoInferPolicy())

    with pytest.raises(RuntimeError, match="requires policy.infer"):
        wrapper.sample_actions({}, num_steps=1)


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


def make_official_raw_with_extras():
    return {
        "observation/image": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((8, 8, 3), dtype=np.uint8),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": "pick object",
        "images": {"image_rgb_0": np.zeros((8, 8, 3), dtype=np.uint8)},
        "image_mask": {"image_rgb_0": True},
        "state": np.ones((8,), dtype=np.float32),
        "image/image_rgb_0": np.zeros((8, 8, 3), dtype=np.uint8),
        "image/image_rgb_1": np.zeros((8, 8, 3), dtype=np.uint8),
        "image/image_rgb_2": np.zeros((8, 8, 3), dtype=np.uint8),
    }


def assert_official_openpi_keys(payload):
    assert set(payload) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }


def test_openpi_libero_raw_payload_matches_official_eval_keys_for_actions():
    policy = MockOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    obs = Observation(raw={"openpi_observation": make_official_raw_with_extras()})
    backend.sample_actions(obs)

    assert_official_openpi_keys(policy.last_obs)
    assert policy.last_obs["prompt"] == "pick object"


def test_openpi_libero_raw_payload_matches_official_eval_keys_for_features():
    policy = MockOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    obs = Observation(raw={"openpi_observation": make_official_raw_with_extras()})
    backend.extract_features(obs)

    assert_official_openpi_keys(policy.last_obs)
    assert policy.last_obs["prompt"] == "pick object"
    assert policy.last_feature_method == "policy_prior_prefix"


def test_openpi_libero_raw_payload_task_override_updates_prompt():
    policy = MockOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    obs = Observation(raw={"openpi_observation": make_official_raw_with_extras()})
    backend.sample_actions(obs, task="override task")

    assert_official_openpi_keys(policy.last_obs)
    assert policy.last_obs["prompt"] == "override task"


def test_openpi_libero_raw_payload_task_override_can_supply_missing_raw_prompt():
    policy = MockOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    raw = make_official_raw_with_extras()
    raw.pop("prompt")
    obs = Observation(raw={"openpi_observation": raw})
    backend.sample_actions(obs, task="override task")

    assert_official_openpi_keys(policy.last_obs)
    assert policy.last_obs["prompt"] == "override task"


def test_openpi_libero_raw_payload_rejects_partial_official_keys():
    policy = MockOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    raw = make_official_raw_with_extras()
    raw.pop("observation/wrist_image")
    obs = Observation(raw={"openpi_observation": raw})

    with pytest.raises(ValueError, match="missing official keys"):
        backend.sample_actions(obs)


def test_openpi_non_libero_raw_payload_passes_through():
    policy = MockOpenPIPolicy()
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=policy)
    raw = {
        "observation/image": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/wrist_left_image": np.ones((8, 8, 3), dtype=np.uint8),
        "observation/wrist_right_image": np.ones((8, 8, 3), dtype=np.uint8),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": "handover object",
    }
    obs = Observation(raw={"openpi_observation": raw})
    backend.sample_actions(obs)

    assert policy.last_obs == raw


def test_openpi_missing_predict_action_with_features_returns_action_only_features():
    backend = OpenPIBackend(config_name="pi05_libero", checkpoint_path="/tmp/ckpt", policy=NoFeaturePolicy())
    features = backend.extract_features(make_obs())

    assert features.reference_actions.shape == (2, 32)
    assert features.embeddings == {}
    assert features.metadata["feature_mode"] == "reference_actions_only"
