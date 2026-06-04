from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
import datetime as _datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_rl.data import ActionSpec, Observation, PolicyFeatures
from vla_rl.policies.base import PolicyBackend


def _tree_map(fn, tree):
    if isinstance(tree, dict):
        return {key: _tree_map(fn, value) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_tree_map(fn, value) for value in tree)
    if isinstance(tree, list):
        return [_tree_map(fn, value) for value in tree]
    return fn(tree)


def _numpy_to_batched_torch(tree: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return _tree_map(
        lambda value: torch.from_numpy(np.array(value, copy=True)).to(device)[None, ...],
        tree,
    )


def _to_pytorch_image_layout(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 4 and image.shape[-1] == 3 and image.dtype != torch.uint8:
        return image.permute(0, 3, 1, 2).contiguous()
    return image


def _normalize_image_layouts(obs_torch: dict[str, Any]) -> dict[str, Any]:
    images = obs_torch.get("image")
    if isinstance(images, dict):
        obs_torch = dict(obs_torch)
        obs_torch["image"] = {key: _to_pytorch_image_layout(value) for key, value in images.items()}
    return obs_torch


def _patch_python310_datetime_utc() -> None:
    if not hasattr(_datetime, "UTC"):
        _datetime.UTC = _datetime.timezone.utc


@dataclass(frozen=True)
class _OpenPIFeatureBatch:
    prefix: torch.Tensor
    reference_actions: np.ndarray


class _OpenPIBasePolicy:
    def __init__(self, policy: Any, observation_cls: Any, device: str | torch.device) -> None:
        self.policy = policy
        self.model = policy._model
        self.observation_cls = observation_cls
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def raw_obs_to_torch(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        processed = self.policy._input_transform(raw_obs)
        return _normalize_image_layouts(_numpy_to_batched_torch(processed, self.device))

    def to_observation(self, obs_torch: dict[str, Any]) -> Any:
        return self.observation_cls.from_dict(obs_torch)

    def _unnormalize_actions(self, obs_torch: dict[str, Any], actions: torch.Tensor) -> np.ndarray:
        out = {
            "state": obs_torch["state"].detach().cpu()[0],
            "actions": actions.detach().cpu()[0],
        }
        return np.asarray(self.policy._output_transform(out)["actions"], dtype=np.float32)

    @torch.no_grad()
    def infer_features(self, raw_obs: dict[str, Any], num_steps: int = 10) -> _OpenPIFeatureBatch:
        obs_torch = self.raw_obs_to_torch(raw_obs)
        obs_obj = self.to_observation(obs_torch)
        if not hasattr(self.model, "predict_action_with_features"):
            raise RuntimeError(
                "OpenPI model does not expose predict_action_with_features(). "
                "Use the OpenPI RLT branch that returns reference actions and prefix features in one forward pass."
            )
        out = self.model.predict_action_with_features(
            device=self.device,
            observation=obs_obj,
            noise=None,
            num_steps=num_steps,
        )
        if not isinstance(out, dict) or "actions" not in out or "features" not in out:
            raise RuntimeError("predict_action_with_features() must return a dict with 'actions' and 'features'")
        features = out["features"]
        if not isinstance(features, dict) or "prefix" not in features:
            raise RuntimeError("predict_action_with_features()['features'] must contain 'prefix'")
        ref_actions = out["actions"]
        unnorm_actions = self._unnormalize_actions(obs_torch, ref_actions)
        return _OpenPIFeatureBatch(
            prefix=torch.as_tensor(features["prefix"], device=self.device).to(torch.float32),
            reference_actions=unnorm_actions,
        )

    def sample_actions(self, raw_obs: dict[str, Any], **kwargs) -> np.ndarray:
        num_steps = int(kwargs.pop("num_steps", 10))
        return self.infer_features(raw_obs, num_steps=num_steps).reference_actions


class OpenPIBackend(PolicyBackend):
    """Torch OpenPI policy adapter for VLA-RL.

    The adapter keeps OpenPI-specific imports and input formatting behind the
    `PolicyBackend` contract. Tests can inject a mock `policy` directly; real
    usage provides `openpi_root`, `config_name`, and `checkpoint_path`.
    """

    def __init__(
        self,
        openpi_root: str | None = None,
        config_name: str = "pi05_libero",
        checkpoint_path: str | None = None,
        action_dim: int = 32,
        device: str = "cuda",
        policy: Any | None = None,
    ) -> None:
        self.openpi_root = str(openpi_root) if openpi_root is not None else None
        self.config_name = config_name
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None
        self.action_dim = int(action_dim)
        self.device = device
        self.policy = policy if policy is not None else self._load_policy()

    def action_spec(self) -> ActionSpec:
        spec = ActionSpec(shape=(self.action_dim,), minimum=-1.0, maximum=1.0)
        spec.validate()
        return spec

    def sample_actions(self, obs: Observation, task: str | None = None, **kwargs) -> np.ndarray:
        openpi_obs = self._to_openpi_observation(obs, task=task)
        raw_actions = self._call_sample_actions(openpi_obs, **kwargs)
        return self._normalize_action_array(raw_actions)

    def extract_features(
        self,
        obs: Observation,
        actions: np.ndarray | None = None,
        **kwargs,
    ) -> PolicyFeatures:
        if hasattr(self.policy, "infer_features"):
            openpi_obs = self._to_openpi_observation(obs, task=obs.task)
            feature_batch = self.policy.infer_features(openpi_obs)
            features = PolicyFeatures(
                reference_actions=self._normalize_reference_actions(feature_batch.reference_actions),
                embeddings={
                    "prefix": self._to_numpy(feature_batch.prefix),
                },
                proprio=obs.proprio.copy() if obs.proprio is not None else None,
                metadata=self._metadata(),
            )
            features.validate()
            return features

        if not hasattr(self.policy, "predict_action_with_features"):
            raise RuntimeError(
                "OpenPI policy does not expose predict_action_with_features(). "
                "Use the OpenPI RLT branch that returns reference actions and prefix features in one call."
            )

        openpi_obs = self._to_openpi_observation(obs, task=obs.task)
        del actions
        out = self.policy.predict_action_with_features(openpi_obs, **kwargs)
        if not isinstance(out, dict) or "actions" not in out or "features" not in out:
            raise RuntimeError("predict_action_with_features() must return a dict with 'actions' and 'features'")
        feature_dict = out["features"]
        if not isinstance(feature_dict, dict) or "prefix" not in feature_dict:
            raise RuntimeError("predict_action_with_features()['features'] must contain 'prefix'")
        features = PolicyFeatures(
            reference_actions=self._normalize_reference_actions(out["actions"]),
            embeddings={
                "prefix": self._to_numpy(feature_dict["prefix"]),
            },
            proprio=obs.proprio.copy() if obs.proprio is not None else None,
            metadata=self._metadata(),
        )
        features.validate()
        return features

    def _load_policy(self) -> Any:
        if self.openpi_root is None or self.checkpoint_path is None:
            raise ValueError("openpi_root and checkpoint_path are required when policy is not injected")

        openpi_src = Path(self.openpi_root).expanduser().resolve() / "src"
        if str(openpi_src) not in sys.path:
            sys.path.insert(0, str(openpi_src))

        _patch_python310_datetime_utc()
        policy_config = importlib.import_module("openpi.policies.policy_config")
        model_module = importlib.import_module("openpi.models.model")
        train_config = importlib.import_module("openpi.training.config")
        if hasattr(train_config, "get_config"):
            config = train_config.get_config(self.config_name)
        elif hasattr(policy_config, "get_config"):
            config = policy_config.get_config(self.config_name)
        elif hasattr(policy_config, "_config") and hasattr(policy_config._config, "get_config"):
            config = policy_config._config.get_config(self.config_name)
        else:
            raise RuntimeError("OpenPI config modules do not expose get_config()")
        try:
            policy = policy_config.create_trained_policy(config, self.checkpoint_path, pytorch_device=str(self.device))
        except TypeError:
            policy = policy_config.create_trained_policy(config, self.checkpoint_path)
        if hasattr(policy, "_model") and hasattr(model_module, "Observation"):
            return _OpenPIBasePolicy(policy=policy, observation_cls=model_module.Observation, device=self.device)
        return policy

    def _to_openpi_observation(self, obs: Observation, task: str | None = None) -> dict[str, Any]:
        if "openpi_observation" in obs.raw:
            raw_obs = obs.raw["openpi_observation"]
            if not isinstance(raw_obs, dict):
                raise ValueError("Observation.raw['openpi_observation'] must be a dict")
            return raw_obs

        openpi_obs: dict[str, Any] = {}
        for key, image in obs.images.items():
            openpi_obs[f"image/{key}"] = np.asarray(image)
        if obs.proprio is not None:
            openpi_obs["state"] = np.asarray(obs.proprio, dtype=np.float32)
        prompt = task or obs.task
        if prompt is not None:
            openpi_obs["prompt"] = prompt
        return openpi_obs

    def _call_sample_actions(self, openpi_obs: dict[str, Any], **kwargs) -> np.ndarray:
        if hasattr(self.policy, "sample_actions"):
            return self.policy.sample_actions(openpi_obs, **kwargs)
        if hasattr(self.policy, "infer"):
            result = self.policy.infer(openpi_obs, **kwargs)
            if isinstance(result, dict):
                for key in ("actions", "action", "raw_actions"):
                    if key in result:
                        return result[key]
            return result
        raise RuntimeError("OpenPI policy must expose sample_actions() or infer()")

    def _normalize_action_array(self, actions: Any) -> np.ndarray:
        array = self._to_numpy(actions).astype(np.float32, copy=False)
        if array.ndim == 3:
            if array.shape[0] != 1:
                raise ValueError(f"batched actions must have batch=1, got shape={array.shape}")
            array = array[0]
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError(f"actions must be [T, D], got shape={array.shape}")
        if array.shape[-1] != self.action_dim:
            raise ValueError(f"expected action_dim={self.action_dim}, got shape={array.shape}")
        return array

    def _normalize_reference_actions(self, actions: Any) -> np.ndarray:
        array = self._to_numpy(actions).astype(np.float32, copy=False)
        if array.ndim == 3:
            if array.shape[0] != 1:
                raise ValueError(f"batched reference actions must have batch=1, got shape={array.shape}")
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"reference actions must be [T, D], got shape={array.shape}")
        return array

    def _metadata(self) -> dict[str, Any]:
        return {
            "policy": "openpi",
            "config_name": self.config_name,
            "checkpoint_path": self.checkpoint_path,
        }

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)
