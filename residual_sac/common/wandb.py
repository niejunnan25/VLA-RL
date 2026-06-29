import datetime
import tempfile
from copy import copy
from pathlib import Path
from socket import gethostname

import ml_collections

try:
    import swanlab
except ModuleNotFoundError:  # pragma: no cover - depends on training env
    swanlab = None


def _recursive_flatten_dict(d: dict):
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values


def _resolve_swanlab_mode(mode: str) -> str | None:
    resolved_mode = str(mode).lower()
    if resolved_mode == "disabled":
        return None
    if resolved_mode == "online":
        return "cloud"
    if resolved_mode == "offline":
        return "local"
    raise ValueError(f"Unsupported logging mode: {resolved_mode!r}")


class WandBLogger(object):
    @staticmethod
    def get_default_config():
        config = ml_collections.ConfigDict()
        config.project = "residual_sac"
        config.entity = ml_collections.config_dict.FieldReference(None, field_type=str)
        config.exp_descriptor = ""
        config.unique_identifier = ""
        config.group = None
        config.mode = None
        return config

    def __init__(
        self,
        wandb_config,
        variant,
        wandb_output_dir=None,
        mode=None,
        debug=False,
    ):
        self.config = wandb_config
        if self.config.unique_identifier == "":
            self.config.unique_identifier = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

        self.config.experiment_id = (
            self.experiment_id
        ) = f"{self.config.exp_descriptor}_{self.config.unique_identifier}"

        if wandb_output_dir is None:
            wandb_output_dir = tempfile.mkdtemp()

        self._variant = copy(variant)
        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        resolved_mode = mode
        if resolved_mode is None:
            resolved_mode = getattr(self.config, "mode", None)
        if resolved_mode in (None, "", "none"):
            resolved_mode = "disabled" if debug else "online"
        elif debug:
            resolved_mode = "disabled"

        self.run = None
        self._swanlab = None
        swanlab_mode = _resolve_swanlab_mode(str(resolved_mode).lower())
        if swanlab_mode is None:
            return
        if swanlab is None:
            raise ImportError(
                "SwanLab logging is enabled, but swanlab is not installed. "
                "Install swanlab or set wandb.mode=disabled."
            )

        output_dir = Path(wandb_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs = {
            "project": self.config.project,
            "experiment_name": self.config.exp_descriptor,
            "config": self._variant,
            "logdir": str(output_dir),
            "mode": swanlab_mode,
        }
        if self.config.entity not in (None, ""):
            init_kwargs["workspace"] = self.config.entity
        if self.config.group not in (None, ""):
            init_kwargs["group"] = self.config.group
        tags = getattr(self.config, "tag", None)
        if tags:
            init_kwargs["tags"] = tags
        if self.experiment_id:
            init_kwargs["id"] = self.experiment_id

        self._swanlab = swanlab
        self.run = swanlab.init(**init_kwargs)

    def log(self, data: dict, step: int = None):
        if self._swanlab is None:
            return
        data_flat = _recursive_flatten_dict(data)
        data = {k: v for k, v in zip(*data_flat)}
        if data:
            self._swanlab.log(data, step=step)

    def finish(self):
        if self._swanlab is not None:
            self._swanlab.finish()
