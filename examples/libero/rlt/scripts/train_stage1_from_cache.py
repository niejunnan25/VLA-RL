#!/usr/bin/env python3
"""Train RLT Stage 1 encoder/decoder from cached prefix embeddings."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.libero.rlt.scripts.train_stage1 import _build_modules, checkpoint_payload

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency for long runs.
    SummaryWriter = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PrefixCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.meta = json.loads((self.cache_dir / "meta.json").read_text())
        self.total_samples = int(self.meta["total_samples"])
        self.max_tokens = int(self.meta["max_tokens"])
        self.input_dim = int(self.meta["input_dim"])
        self._ranks = list(self.meta["ranks"])
        self._arrays = [np.load(self.cache_dir / str(rank_meta["path"]), mmap_mode="r") for rank_meta in self._ranks]
        self._starts = np.asarray([int(rank_meta["start"]) for rank_meta in self._ranks], dtype=np.int64)
        self._ends = np.asarray([int(rank_meta["end"]) for rank_meta in self._ranks], dtype=np.int64)

    def sample(self, batch_size: int, rng: np.random.Generator) -> torch.Tensor:
        indices = rng.integers(0, self.total_samples, size=int(batch_size), endpoint=False)
        out = np.empty((len(indices), self.max_tokens, self.input_dim), dtype=np.float16)
        for rank_idx, rank_meta in enumerate(self._ranks):
            mask = (indices >= self._starts[rank_idx]) & (indices < self._ends[rank_idx])
            if not mask.any():
                continue
            local_indices = indices[mask] - int(rank_meta["start"])
            out[mask] = self._arrays[rank_idx][local_indices]
        return torch.from_numpy(out)


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="RLT Stage 1 training from cached prefix embeddings")
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def main() -> None:
    cfg = parse_args()
    cache = PrefixCache(cfg.cache.input_dir)
    cfg.rlt.max_tokens = cache.max_tokens
    rng = np.random.default_rng(int(cfg.global_seed))
    torch.manual_seed(int(cfg.global_seed))

    output_dir = Path(str(cfg.training.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    writer = SummaryWriter(log_dir=str(output_dir / "logs")) if SummaryWriter is not None else None
    device = torch.device(str(cfg.vla.device) if torch.cuda.is_available() else "cpu")
    encoder, decoder, optimizer, scheduler = _build_modules(cfg, cache.input_dim, device)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("")

    logger.info(
        "Training RLT Stage 1 from cache: samples=%d tokens=%d input_dim=%d steps=%d",
        cache.total_samples,
        cache.max_tokens,
        cache.input_dim,
        int(cfg.training.steps),
    )
    for global_step in range(int(cfg.training.steps)):
        z_vla = cache.sample(int(cfg.training.batch_size), rng).to(device=device, dtype=torch.float32)
        z_rl = encoder(z_vla)
        z_recon = decoder(z_rl, z_vla)
        loss = F.mse_loss(z_recon, z_vla)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            float(cfg.training.clip_grad_norm),
        )
        optimizer.step()
        scheduler.step()

        lr = float(scheduler.get_last_lr()[0])
        metric = {
            "step": global_step,
            "loss_reconstruction": float(loss.item()),
            "lr": lr,
            "tokens": cache.max_tokens,
            "input_dim": cache.input_dim,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(metric, sort_keys=True) + "\n")
        if writer is not None:
            writer.add_scalar("loss/reconstruction", float(loss.item()), global_step)
            writer.add_scalar("train/lr", lr, global_step)
        if global_step % int(cfg.training.log_every) == 0:
            logger.info("step=%d/%d loss=%.6f lr=%.6g", global_step, int(cfg.training.steps), float(loss.item()), lr)

        step_number = global_step + 1
        if int(cfg.training.save_every) > 0 and step_number % int(cfg.training.save_every) == 0:
            checkpoint_path = output_dir / f"checkpoint_{step_number}.pt"
            torch.save(checkpoint_payload(cfg, cache.input_dim, step_number, encoder, decoder, optimizer, scheduler), checkpoint_path)
            logger.info("Saved checkpoint to %s", checkpoint_path)

    final_path = output_dir / "final_model.pt"
    torch.save(checkpoint_payload(cfg, cache.input_dim, int(cfg.training.steps), encoder, decoder, optimizer, scheduler), final_path)
    torch.save(encoder.state_dict(), output_dir / "rlt_encoder.pt")
    if writer is not None:
        writer.close()
    logger.info("Stage 1 completed. Saved final checkpoint to %s", final_path)


if __name__ == "__main__":
    main()
