from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from vla_rl.algorithms import Algorithm


class CheckpointManager:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.checkpoint_dir / "latest.pt"

    def save(
        self,
        algorithm: Algorithm,
        *,
        env_steps: int,
        update_steps: int,
        episodes: int,
        total_reward: float,
        config: dict[str, Any] | None = None,
        tag: str | None = None,
    ) -> Path:
        name = tag or f"step_{int(env_steps)}.pt"
        path = self.checkpoint_dir / name
        payload = {
            "version": 1,
            "env_steps": int(env_steps),
            "update_steps": int(update_steps),
            "episodes": int(episodes),
            "total_reward": float(total_reward),
            "algorithm_state": algorithm.state_dict(),
            "config": config or {},
        }
        torch.save(payload, path)
        shutil.copy2(path, self.latest_path)
        return path

    def load(self, path: str | Path, algorithm: Algorithm, map_location: str = "cpu") -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=map_location)
        algorithm.load_state_dict(payload.get("algorithm_state", {}))
        return payload
