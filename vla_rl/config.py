from __future__ import annotations

import importlib
from typing import Any

from omegaconf import DictConfig, OmegaConf


def instantiate(cfg: DictConfig | dict, **kwargs) -> Any:
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping config, got {type(data).__name__}")
    target = data.pop("_target_", None)
    if not target:
        raise ValueError("config is missing _target_")
    data.update(kwargs)
    module_name, attr_name = str(target).rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, attr_name)
    return cls(**data)
