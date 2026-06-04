from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_import_check(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def test_policies_root_is_training_side_only():
    run_import_check(
        """
import sys
import vla_rl.policies as policies

assert sorted(policies.__all__) == ["FakePolicyBackend", "PolicyBackend", "ReferencePolicyClient"]
assert "vla_rl.policies.openpi" not in sys.modules
assert "vla_rl.policies.openpi.backend" not in sys.modules
assert not hasattr(policies, "OpenPIBackend")
assert not hasattr(policies, "create_reference_policy")
"""
    )


def test_envs_root_does_not_import_simulator_dependencies():
    run_import_check(
        """
import sys
import vla_rl.envs as envs

assert "EnvBackend" in envs.__all__
assert "FakeEnvBackend" in envs.__all__
assert "LiberoRemoteEnvBackend" in envs.__all__

for module_name in ("libero", "robosuite", "mujoco", "gym", "gymnasium"):
    assert module_name not in sys.modules, module_name
"""
    )


def test_algorithms_root_does_not_export_method_details_or_runners():
    run_import_check(
        """
import vla_rl.algorithms as algorithms

assert sorted(algorithms.__all__) == ["Algorithm", "FakeAlgorithm"]
for name in (
    "RLTAgent",
    "RLTFeatureProcessor",
    "PLDSACAgent",
    "PLDFeatureProcessor",
    "LocalRunner",
    "LocalActorLearnerRunner",
):
    assert not hasattr(algorithms, name), name
"""
    )


def test_runtime_root_exports_only_stable_helpers():
    run_import_check(
        """
import vla_rl.runtime as runtime

assert sorted(runtime.__all__) == ["CheckpointManager"]
assert not hasattr(runtime, "Runner")
assert not hasattr(runtime, "AgentlaceActorRuntime")
assert not hasattr(runtime, "AgentlaceLearnerRuntime")
"""
    )
