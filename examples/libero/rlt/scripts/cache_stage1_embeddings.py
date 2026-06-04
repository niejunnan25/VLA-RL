#!/usr/bin/env python3
"""Cache frozen OpenPI prefix embeddings for RLT Stage 1."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vla_rl.policies.openpi.stage1 import OpenPIStage1Backend


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="Cache RLT Stage 1 prefix embeddings")
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value)
    return value if value else None


def _rank_info(total: int, rank: int, world_size: int) -> tuple[int, int]:
    per_rank = total // world_size
    remainder = total % world_size
    start = rank * per_rank + min(rank, remainder)
    end = start + per_rank + (1 if rank < remainder else 0)
    return start, end


def _collate(items: list[Any]) -> Any:
    first = items[0]
    if isinstance(first, dict):
        return {key: _collate([item[key] for item in items]) for key in first}
    if isinstance(first, tuple):
        return tuple(_collate([item[idx] for item in items]) for idx in range(len(first)))
    if isinstance(first, list):
        return [_collate([item[idx] for item in items]) for idx in range(len(first))]
    return np.stack([np.asarray(item) for item in items], axis=0)


def _truncate_prefix(prefix: torch.Tensor, max_tokens: int | None) -> torch.Tensor:
    if max_tokens is not None and max_tokens > 0:
        prefix = prefix[:, :max_tokens, :]
    return prefix.detach()


def _write_meta(output_dir: Path, meta: dict[str, Any]) -> None:
    path = output_dir / "meta.json"
    tmp_path = output_dir / "meta.json.tmp"
    tmp_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _worker_init(_worker_id: int) -> None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def main() -> None:
    cfg = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    lerobot_home = _optional_str(cfg.vla.get("lerobot_home", None))
    if lerobot_home:
        os.environ["HF_LEROBOT_HOME"] = str(Path(lerobot_home).expanduser().resolve())

    cache_cfg = cfg.get("cache", {})
    output_dir = Path(str(cache_cfg.get("output_dir", "outputs/rlt_stage1_cache/libero_openpi_prefix"))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and str(cfg.vla.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    backend = OpenPIStage1Backend(openpi_root=_optional_str(cfg.vla.openpi_root))
    dataset = backend.create_dataset(
        config_name=str(cfg.vla.config_name),
        assets_base_dir=_optional_str(cfg.vla.assets_base_dir),
        checkpoint_base_dir=_optional_str(cfg.vla.checkpoint_base_dir),
        exp_name=_optional_str(cfg.vla.exp_name),
        repo_id_override=_optional_str(cfg.vla.get("repo_id_override", None)),
    )
    total_samples = len(dataset)
    start, end = _rank_info(total_samples, rank, world_size)
    local_samples = end - start

    base_policy = backend.load_base_policy(
        config_name=str(cfg.vla.config_name),
        checkpoint_path=str(cfg.vla.checkpoint_path),
        device=device,
        assets_base_dir=_optional_str(cfg.vla.assets_base_dir),
        checkpoint_base_dir=_optional_str(cfg.vla.checkpoint_base_dir),
        exp_name=_optional_str(cfg.vla.exp_name),
    )

    batch_size = int(cache_cfg.get("batch_size", cfg.training.get("batch_size", 8)))
    num_workers = int(cache_cfg.get("num_workers", cfg.training.get("num_workers", 0)))
    max_tokens = cfg.rlt.get("max_tokens", None)
    max_tokens = None if max_tokens is None else int(max_tokens)
    subset = torch.utils.data.Subset(dataset, range(start, end))
    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        collate_fn=_collate,
        worker_init_fn=_worker_init,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    mmap = None
    input_dim = None
    token_count = None
    written = 0
    rank_path = output_dir / f"prefix_rank{rank:02d}.npy"

    logger.info("rank=%d/%d caching dataset indices [%d, %d) to %s", rank, world_size, start, end, rank_path)
    for batch in loader:
        obs_batch = {key: value for key, value in batch.items() if key != "actions"}
        obs_obj = backend.observation_to_device(obs_batch, device)
        with torch.no_grad():
            prefix = base_policy.extract_prefix_features(obs_obj, num_steps=int(cfg.vla.num_steps))
            prefix = _truncate_prefix(prefix, max_tokens=max_tokens)

        if mmap is None:
            token_count = int(prefix.shape[1])
            input_dim = int(prefix.shape[2])
            mmap = np.lib.format.open_memmap(
                rank_path,
                mode="w+",
                dtype=np.float16,
                shape=(local_samples, token_count, input_dim),
            )
            if rank == 0:
                ranks = [
                    {
                        "rank": r,
                        "start": _rank_info(total_samples, r, world_size)[0],
                        "end": _rank_info(total_samples, r, world_size)[1],
                        "path": f"prefix_rank{r:02d}.npy",
                    }
                    for r in range(world_size)
                ]
                _write_meta(
                    output_dir,
                    {
                        "format": "vla_rl.rlt_stage1_prefix_cache.v1",
                        "total_samples": total_samples,
                        "world_size": world_size,
                        "dtype": "float16",
                        "max_tokens": token_count,
                        "input_dim": input_dim,
                        "ranks": ranks,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                )

        batch_np = prefix.to(torch.float16).cpu().numpy()
        next_written = written + int(batch_np.shape[0])
        mmap[written:next_written] = batch_np
        written = next_written
        if written % max(batch_size * 20, 1) == 0 or written == local_samples:
            logger.info("rank=%d cached %d/%d", rank, written, local_samples)

    if mmap is not None:
        mmap.flush()
    rank_meta = {
        "rank": rank,
        "start": start,
        "end": end,
        "num_samples": local_samples,
        "path": rank_path.name,
        "input_dim": input_dim,
        "max_tokens": token_count,
        "written": written,
    }
    (output_dir / f"rank_{rank:02d}.json").write_text(json.dumps(rank_meta, indent=2, sort_keys=True) + "\n")
    logger.info("rank=%d done: wrote %d samples", rank, written)


if __name__ == "__main__":
    main()
