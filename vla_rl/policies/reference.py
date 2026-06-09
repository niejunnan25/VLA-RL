from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from vla_rl.data import ActionSpec, Observation, PolicyFeatures
from vla_rl.policies.base import PolicyBackend
from vla_rl.runtime.remote_http import RemoteHttpRpcClient


class ReferencePolicy(Protocol):
    """Frozen VLA used by RLT to provide reference actions and prefix features."""

    def action_spec(self) -> ActionSpec:
        ...

    def predict_action_with_features(self, obs: Observation, **kwargs: Any) -> PolicyFeatures:
        ...

    def predict_actions_and_prefix(self, obs: Observation, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...


@dataclass(slots=True)
class OpenPIReferencePolicy:
    """RLT-facing wrapper around the OpenPI policy adapter."""

    policy: Any

    def action_spec(self) -> ActionSpec:
        return self.policy.action_spec()

    def predict_action_with_features(self, obs: Observation, **kwargs: Any) -> PolicyFeatures:
        return self.policy.extract_features(obs, **kwargs)

    def predict_actions_and_prefix(self, obs: Observation, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = self.predict_action_with_features(obs, **kwargs)
        return _actions_prefix_proprio(features)


class ReferencePolicyClient(PolicyBackend):
    """RLT policy client for a remote reference-policy server.

    The client intentionally exposes the existing `PolicyBackend` methods so an
    RLT recipe can swap an in-process OpenPI policy for a server-backed frozen
    reference policy without changing the actor loop.
    """

    def __init__(
        self,
        url: str,
        action_dim: int = 32,
        timeout: float = 60.0,
        retries: int = 3,
        retry_sleep: float = 1.0,
    ) -> None:
        self.url = url
        self.action_dim = int(action_dim)
        self.client = RemoteHttpRpcClient(
            url=url,
            timeout=float(timeout),
            retries=int(retries),
            retry_sleep=float(retry_sleep),
        )

    def action_spec(self) -> ActionSpec:
        spec = ActionSpec(shape=(self.action_dim,), minimum=-1.0, maximum=1.0)
        spec.validate()
        return spec

    def predict_action_with_features(self, obs: Observation, **kwargs: Any) -> PolicyFeatures:
        result = self.client.call("predict_action_with_features", obs=obs, kwargs=kwargs)
        if not isinstance(result, PolicyFeatures):
            raise RuntimeError(
                "reference-policy server must return vla_rl.data.PolicyFeatures, "
                f"got {type(result).__name__}"
            )
        result.validate()
        return result

    def predict_actions_and_prefix(self, obs: Observation, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = self.predict_action_with_features(obs, **kwargs)
        return _actions_prefix_proprio(features)

    def sample_actions(self, obs: Observation, task: str | None = None, **kwargs: Any) -> np.ndarray:
        if task is not None:
            obs = Observation(
                images=obs.images,
                proprio=obs.proprio,
                task=task,
                raw=obs.raw,
            )
        features = self.predict_action_with_features(obs, **kwargs)
        if features.reference_actions is None:
            raise RuntimeError("reference-policy server returned no reference_actions")
        return np.asarray(features.reference_actions, dtype=np.float32)

    def extract_features(
        self,
        obs: Observation,
        actions: np.ndarray | None = None,
        **kwargs: Any,
    ) -> PolicyFeatures:
        del actions
        return self.predict_action_with_features(obs, **kwargs)

    def close(self) -> None:
        self.client.close()


def create_reference_policy(name: str, **kwargs: Any) -> ReferencePolicy:
    if name == "openpi":
        from vla_rl.policies.openpi import OpenPIBackend

        policy = OpenPIBackend(
            openpi_root=kwargs.get("policy_root") or kwargs.get("openpi_root"),
            config_name=kwargs.get("config_name", "pi0_libero"),
            checkpoint_path=kwargs.get("checkpoint_path"),
            action_dim=int(kwargs.get("action_dim", 32)),
            device=kwargs.get("device", "cuda"),
        )
        return OpenPIReferencePolicy(policy=policy)
    if name == "starvla":
        raise NotImplementedError("StarVLAReferencePolicy is not implemented yet")
    raise ValueError(f"unsupported reference policy: {name}")


def _actions_prefix_proprio(features: PolicyFeatures) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.reference_actions is None:
        raise RuntimeError("reference policy returned no reference_actions")
    if "prefix" not in features.embeddings:
        raise RuntimeError("reference policy returned no prefix embeddings")
    proprio = features.proprio
    if proprio is None:
        proprio = np.zeros((0,), dtype=np.float32)
    return (
        np.asarray(features.reference_actions, dtype=np.float32),
        np.asarray(features.embeddings["prefix"], dtype=np.float32),
        np.asarray(proprio, dtype=np.float32).reshape(-1),
    )
