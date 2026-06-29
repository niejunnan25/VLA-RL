from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
from typing import Any
from urllib.parse import urlparse

import numpy as np

from vla_rl.data import ActionSpec, Observation, PolicyFeatures
from vla_rl.policies.base import PolicyBackend


class OpenPIWebsocketPolicyClient(PolicyBackend):
    """OpenPI official websocket client behind the VLA-RL PolicyBackend API.

    This adapter is intentionally action-only. PLD residual SAC only needs the
    frozen base action chunk; RLT feature extraction should use the native
    OpenPIBackend path instead.
    """

    def __init__(
        self,
        url: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8001,
        action_dim: int = 7,
        serl_torch_root: str = "/vla/users/niejunnan/codebase/serl_torch",
        connect_timeout_sec: float = 30.0,
        ping_interval_sec: float = 20.0,
        ping_timeout_sec: float = 120.0,
        reconnect_retry_count: int = 1,
        reconnect_retry_backoff_sec: float = 0.05,
    ) -> None:
        if url:
            parsed = urlparse(str(url))
            if parsed.hostname:
                host = parsed.hostname
            if parsed.port is not None:
                port = int(parsed.port)
        self.url = url or f"http://{host}:{int(port)}"
        self.host = str(host)
        self.port = int(port)
        self.action_dim = int(action_dim)
        self.serl_torch_root = str(serl_torch_root)
        _add_serl_torch_paths(self.serl_torch_root)

        from serl_launcher.policy.openpi.client import OpenPIPolicyClient

        self._build_policy_input = _load_libero_policy_input_builder(self.serl_torch_root)
        self.client = OpenPIPolicyClient(
            host=self.host,
            port=self.port,
            action_dim=self.action_dim,
            connect_timeout_sec=connect_timeout_sec,
            ping_interval_sec=ping_interval_sec,
            ping_timeout_sec=ping_timeout_sec,
            reconnect_retry_count=reconnect_retry_count,
            reconnect_retry_backoff_sec=reconnect_retry_backoff_sec,
        )

    def action_spec(self) -> ActionSpec:
        spec = ActionSpec(shape=(self.action_dim,), minimum=-1.0, maximum=1.0)
        spec.validate()
        return spec

    def sample_actions(self, obs: Observation, task: str | None = None, **kwargs: Any) -> np.ndarray:
        num_steps = int(kwargs.pop("num_steps", 10))
        if kwargs:
            raise ValueError(
                "OpenPIWebsocketPolicyClient.sample_actions only accepts num_steps; "
                f"got unsupported kwargs={sorted(kwargs)}"
            )
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        prompt = task or obs.task
        if prompt is None:
            raw_openpi = obs.raw.get("openpi_observation", {}) if isinstance(obs.raw, dict) else {}
            prompt = raw_openpi.get("prompt")
        if prompt is None:
            raise ValueError("OpenPI websocket inference requires a task prompt")

        state = obs.proprio
        if state is None and isinstance(obs.raw, dict):
            raw_openpi = obs.raw.get("openpi_observation", {})
            if isinstance(raw_openpi, dict):
                state = raw_openpi.get("observation/state")
                if state is None:
                    state = raw_openpi.get("state")
        if state is None:
            raise ValueError("OpenPI websocket inference requires proprio/state")

        images = _libero_images(obs)
        policy_input = self._build_policy_input(
            prompt=str(prompt),
            state=np.asarray(state, dtype=np.float32),
            images=images,
        )
        action_chunk, _info = self.client.infer(policy_input)
        actions = np.asarray(action_chunk, dtype=np.float32)
        if actions.ndim == 1:
            if actions.size % self.action_dim != 0:
                raise ValueError(f"cannot reshape OpenPI actions with shape {actions.shape} to action_dim={self.action_dim}")
            actions = actions.reshape(-1, self.action_dim)
        if actions.ndim != 2:
            raise ValueError(f"OpenPI websocket actions must have shape [T, D], got {actions.shape}")
        if int(actions.shape[0]) < num_steps:
            raise ValueError(f"OpenPI websocket returned {int(actions.shape[0])} actions, expected at least {num_steps}")
        if int(actions.shape[1]) != self.action_dim:
            raise ValueError(f"OpenPI websocket action_dim mismatch: expected={self.action_dim} got={int(actions.shape[1])}")
        return actions[:num_steps]

    def extract_features(
        self,
        obs: Observation,
        actions: np.ndarray | None = None,
        **kwargs: Any,
    ) -> PolicyFeatures:
        raise NotImplementedError(
            "OpenPIWebsocketPolicyClient is action-only and is intended for PLD residual SAC; "
            "use vla_rl.policies.openpi.OpenPIBackend for feature extraction."
        )

    def close(self) -> None:
        self.client.close()


def _load_libero_policy_input_builder(serl_torch_root: str):
    path = Path(serl_torch_root).expanduser().resolve() / "examples/libero/env/policy_input.py"
    if not path.exists():
        raise FileNotFoundError(f"serl_torch LIBERO policy_input.py not found: {path}")
    module_name = "_vla_rl_serl_torch_libero_policy_input"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load serl_torch LIBERO policy_input.py from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.build_libero_policy_input


def _add_serl_torch_paths(serl_torch_root: str) -> None:
    root = Path(serl_torch_root).expanduser().resolve()
    for path in (root, root / "serl_launcher"):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def _libero_images(obs: Observation) -> dict[str, np.ndarray]:
    images = dict(obs.images)
    if not images and isinstance(obs.raw, dict):
        raw_openpi = obs.raw.get("openpi_observation", {})
        if isinstance(raw_openpi, dict) and isinstance(raw_openpi.get("images"), dict):
            images = dict(raw_openpi["images"])
    required = ("image_rgb_0", "image_rgb_1", "image_rgb_2")
    missing = [key for key in required if key not in images]
    if missing:
        raise ValueError(f"LIBERO OpenPI websocket observation is missing image keys: {missing}")
    return {key: np.asarray(images[key], dtype=np.uint8) for key in required}
