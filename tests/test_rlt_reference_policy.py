from __future__ import annotations

from http.server import ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np

from vla_rl.policies.reference import ReferencePolicyClient
from vla_rl.data import Observation, PolicyFeatures
from vla_rl.runtime.remote_http import make_pickle_rpc_handler



def test_policies_root_import_does_not_load_openpi_backend():
    code = """
import sys
import vla_rl.policies
from vla_rl.policies import FakePolicyBackend, PolicyBackend, ReferencePolicyClient
assert FakePolicyBackend is not None
assert PolicyBackend is not None
assert ReferencePolicyClient is not None
assert "vla_rl.policies.openpi" not in sys.modules
assert "vla_rl.policies.openpi.backend" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def make_obs() -> Observation:
    return Observation(
        images={"front": np.zeros((8, 8, 3), dtype=np.uint8)},
        proprio=np.zeros((4,), dtype=np.float32),
        task="pick object",
    )


def test_reference_policy_client_returns_policy_features():
    def dispatch(method, kwargs):
        assert method == "predict_action_with_features"
        obs = kwargs["obs"]
        assert obs.task == "pick object"
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
        chunk = client.sample_actions(make_obs())
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert features.reference_actions.shape == (2, 3)
    assert features.embeddings["prefix"].shape == (1, 4, 5)
    assert chunk.actions.shape == (2, 3)
    assert chunk.metadata["policy"] == "fake_reference"
