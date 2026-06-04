from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from vla_rl.runtime.agentlace import json_sanitize


def load_config(path: str, overrides: list[str]):
    cfg = OmegaConf.load(path)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def run_dir(runtime) -> Path | None:
    value = runtime.get("run_dir", None)
    if value is None or str(value) == "":
        return None
    return Path(str(value)).expanduser()


def make_jsonl_writer(path: Path | None):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def write(metric: dict[str, Any]) -> None:
        if path is None:
            return
        with path.open("a") as f:
            f.write(json.dumps(json_sanitize(metric), sort_keys=True) + "\n")

    return write


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_sanitize(payload), sort_keys=True, indent=2) + "\n")
