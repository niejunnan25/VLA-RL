from __future__ import annotations

from http.server import ThreadingHTTPServer
import threading

import numpy as np

from vla_rl.policies.reference import ReferencePolicyClient
from vla_rl.data import Observation, PolicyFeatures
from vla_rl.runtime.remote_http import make_pickle_rpc_handler


def make_obs() -> Observation:
    return Observation(
        images={"front": np.zeros((8, 8, 3), dtype=np.uint8)},
        proprio=np.zeros((4,), dtype=np.float32),
        task="pick object",
    )


def test_reference_policy_client_returns_policy_features():
    def dispatch(method, kwargs):
        obs = kwargs["obs"]
        if method == "sample_actions":
            if kwargs["kwargs"]:
                assert obs.task == "override task"
                assert kwargs["kwargs"] == {"task": "override task"}
            else:
                assert obs.task == "pick object"
                assert kwargs["kwargs"] == {}
            return np.ones((2, 3), dtype=np.float32)

        assert method == "predict_action_with_features"
        assert obs.task == "pick object"
        if kwargs["kwargs"]:
            assert kwargs["kwargs"] == {"feature_source": "self_conditioned_prefix"}
        return PolicyFeatures(
            reference_actions=np.ones((2, 3), dtype=np.float32),
            embeddings={"prefix": np.zeros((1, 4, 5), dtype=np.float32)},
            proprio=obs.proprio,
            metadata={"policy": "fake_reference"},
        )

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_pickle_rpc_handler(dispatch))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ReferencePolicyClient(url=f"http://127.0.0.1:{server.server_port}", action_dim=3)
        features = client.extract_features(make_obs())
        actions = client.sample_actions(make_obs())
        override_actions = client.sample_actions(make_obs(), task="override task")
        base_actions, prefix_tokens, proprio = client.predict_actions_and_prefix(make_obs())
        client.predict_actions_and_prefix(make_obs(), feature_source="self_conditioned_prefix")
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert features.reference_actions.shape == (2, 3)
    assert features.embeddings["prefix"].shape == (1, 4, 5)
    assert actions.shape == (2, 3)
    assert override_actions.shape == (2, 3)
    assert base_actions.shape == (2, 3)
    assert prefix_tokens.shape == (1, 4, 5)
    assert proprio.shape == (4,)
    assert features.metadata["policy"] == "fake_reference"
