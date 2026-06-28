from __future__ import annotations

import datetime as _datetime
import math
from pathlib import Path
from socket import gethostname
from typing import Any

import numpy as np


_CLOUD_METRIC_EXCLUDED_KEYS = {
    "eval/episodes_run",
    "eval/force_zero_residual",
    "eval/residual_l1",
    "eval/residual_l2",
}


class NullWandBLogger:
    """No-op logger used when W&B-style logging is disabled."""

    enabled = False

    def log(self, data: dict[str, Any], *, step: int | None = None) -> None:
        del data, step

    def finish(self) -> None:
        return


class _NativeWandBRun:
    """Native W&B run used when SwanLab is unavailable or disabled.

    The learner owns the run. Actor metrics are forwarded to the learner through
    Agentlace and logged by the same W&B-style logger, matching the HIL-SERL pattern.
    """

    enabled = True

    def __init__(self, cfg: Any, *, variant: dict[str, Any], run_dir: Path | None) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B logging is enabled, but the 'wandb' package is not installed. "
                "Install wandb or set wandb.mode=disabled."
            ) from exc

        self._wandb = wandb
        project = _cfg_get(cfg, "project", "vla-rl")
        entity = _cfg_get(cfg, "entity", None)
        group = _cfg_get(cfg, "group", None)
        name = _cfg_get(cfg, "exp_name", None) or _cfg_get(cfg, "name", None)
        tags = _normal_tags(_cfg_get(cfg, "tags", None))
        mode = _wandb_mode(_cfg_get(cfg, "mode", "online"))
        save_code = bool(_cfg_get(cfg, "save_code", True))
        run_id = _cfg_get(cfg, "id", None)
        if not name:
            descriptor = _cfg_get(cfg, "exp_descriptor", "vla-rl")
            stamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{descriptor}_{stamp}"

        wandb_dir = Path(_cfg_get(cfg, "dir", None) or (run_dir / "wandb" if run_dir is not None else "/tmp"))
        wandb_dir.mkdir(parents=True, exist_ok=True)
        variant = dict(variant)
        variant.setdefault("hostname", gethostname())

        init_kwargs = {
            "project": project,
            "entity": entity,
            "group": group,
            "name": name,
            "tags": tags,
            "dir": str(wandb_dir),
            "config": variant,
            "save_code": save_code,
            "mode": mode,
        }
        if run_id:
            init_kwargs["id"] = str(run_id)

        self.run = wandb.init(**init_kwargs)
        define_metric = getattr(self._wandb, "define_metric", None)
        if callable(define_metric):
            define_metric("rollout/episode_id")
            define_metric("rollout/*", step_metric="rollout/episode_id")
            define_metric("learner/update_steps")
            define_metric("learner/*", step_metric="learner/update_steps")
            define_metric("eval/train_episode")
            define_metric("eval/*", step_metric="eval/train_episode")

    def log(self, data: dict[str, Any], *, step: int | None = None) -> None:
        payload = select_hil_serl_wandb_scalars(data)
        if payload:
            del step
            self._wandb.log(payload)

    def finish(self) -> None:
        self._wandb.finish()


class _SwanLabRun:
    """SwanLab run used as the default W&B-compatible backend."""

    enabled = True

    def __init__(self, cfg: Any, *, variant: dict[str, Any], run_dir: Path | None) -> None:
        try:
            import swanlab
        except ImportError as exc:
            raise ImportError("SwanLab is not installed.") from exc

        self._swanlab = swanlab
        project = _cfg_get(cfg, "project", "vla-rl")
        workspace = _cfg_get(cfg, "workspace", None) or _cfg_get(cfg, "entity", None)
        group = _cfg_get(cfg, "group", None)
        name = _cfg_get(cfg, "exp_name", None) or _cfg_get(cfg, "name", None)
        tags = _normal_tags(_cfg_get(cfg, "tags", None))
        mode = _swanlab_mode(_cfg_get(cfg, "mode", "online"))
        run_id = _cfg_get(cfg, "id", None)
        if not name:
            descriptor = _cfg_get(cfg, "exp_descriptor", "vla-rl")
            stamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{descriptor}_{stamp}"

        logdir = Path(_cfg_get(cfg, "dir", None) or (run_dir / "swanlab" if run_dir is not None else "/tmp"))
        logdir.mkdir(parents=True, exist_ok=True)
        variant = dict(variant)
        variant.setdefault("hostname", gethostname())

        init_kwargs = {
            "project": project,
            "workspace": workspace,
            "experiment_name": name,
            "group": group,
            "tags": tags,
            "config": variant,
            "logdir": str(logdir),
            "mode": mode,
        }
        if run_id:
            init_kwargs["id"] = str(run_id)

        self.run = swanlab.init(**init_kwargs)

    def log(self, data: dict[str, Any], *, step: int | None = None) -> None:
        payload = select_hil_serl_wandb_scalars(data)
        if payload:
            self._swanlab.log(payload, step=select_hil_serl_wandb_step(payload, default=step))

    def finish(self) -> None:
        self._swanlab.finish()


def make_wandb_logger(
    cfg: Any,
    *,
    variant: dict[str, Any],
    run_dir: Path | None,
) -> NullWandBLogger | _SwanLabRun | _NativeWandBRun:
    if cfg is None:
        return NullWandBLogger()
    if bool(_cfg_get(cfg, "debug", False)) or str(_cfg_get(cfg, "mode", "online")).lower() == "disabled":
        return NullWandBLogger()
    try:
        return _SwanLabRun(cfg, variant=variant, run_dir=run_dir)
    except ImportError as exc:
        print(f"[wandb] SwanLab unavailable, falling back to W&B: {exc}")
        return _NativeWandBRun(cfg, variant=variant, run_dir=run_dir)


def flatten_wandb_scalars(data: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    _flatten_into(flat, "", data)
    return flat


def select_hil_serl_wandb_scalars(data: dict[str, Any]) -> dict[str, Any]:
    """Return the cloud metric set used for RLT training diagnosis.

    Local JSONL files keep richer debug metrics such as timers and environment
    internals. SwanLab/W&B only gets the groups needed to judge training:
    rollout first, learner next, and eval last.
    """

    flat = flatten_wandb_scalars(data)
    selected: dict[str, Any] = {}
    for key, value in flat.items():
        if key in _CLOUD_METRIC_EXCLUDED_KEYS:
            continue
        if key.startswith(("rollout/", "learner/", "eval/")):
            selected[key] = value
    return selected


def select_hil_serl_wandb_step(data: dict[str, Any], *, default: int | None = None) -> int | None:
    """Choose the cloud x-axis for a single SERL-style metric payload."""

    if _has_only_prefix(data, "rollout/") and "rollout/episode_id" in data:
        return int(data["rollout/episode_id"])
    if _has_only_prefix(data, "eval/") and "eval/train_episode" in data:
        return int(data["eval/train_episode"])
    if _has_only_prefix(data, "learner/") and "learner/update_steps" in data:
        return int(data["learner/update_steps"])
    return default


def _has_only_prefix(data: dict[str, Any], prefix: str) -> bool:
    return bool(data) and all(key.startswith(prefix) for key in data)


def _flatten_into(flat: dict[str, Any], prefix: str, value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            _flatten_into(flat, child, item)
        return
    value = _to_wandb_scalar(value)
    if value is not None and prefix:
        flat[prefix] = value


def _to_wandb_scalar(value: Any) -> int | float | str | bool | None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        else:
            return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    return None


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normal_tags(tags: Any) -> list[str] | None:
    if tags is None:
        return None
    return [str(tag) for tag in list(tags)]


def _swanlab_mode(mode: Any) -> str | None:
    if mode is None:
        return None
    mode = str(mode).lower()
    if mode == "online":
        return "cloud"
    return mode


def _wandb_mode(mode: Any) -> str:
    mode = str(mode or "online").lower()
    if mode == "cloud":
        return "online"
    return mode
