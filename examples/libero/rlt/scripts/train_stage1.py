#!/usr/bin/env python3
"""Train RLT Stage 1 encoder/decoder on frozen OpenPI prefix features."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.libero.rlt.config import feature_cfg, load_config, normalize_max_tokens
from vla_rl.algorithms.rlt.modeling import RLTokenDecoder, RLTokenEncoder

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency for long runs.
    SummaryWriter = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="RLT Stage 1 prefix reconstruction training")
    parser.add_argument("--config", required=True, help="Path to Stage 1 YAML config")
    args, overrides = parser.parse_known_args()
    return load_config(args.config, list(overrides))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    resolved = str(value)
    return resolved if resolved else None


def _lr_factor(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def _stage1_prefix_target(prefix: torch.Tensor, max_tokens: int | None) -> torch.Tensor:
    if prefix.dim() != 3:
        raise ValueError(f"prefix features must be [B, T, D], got {tuple(prefix.shape)}")
    if max_tokens is not None and max_tokens > 0:
        prefix = prefix[:, : int(max_tokens), :]
    return prefix.detach().to(torch.float32)


def _build_modules(cfg: Any, input_dim: int, device: torch.device):
    encoder = RLTokenEncoder(
        input_dim=input_dim,
        rl_token_dim=int(cfg.rlt.rl_token_dim),
        num_layers=int(cfg.rlt.num_encoder_layers),
        num_heads=int(cfg.rlt.num_heads),
        ff_dim=int(cfg.rlt.ff_dim),
        dropout=float(cfg.rlt.dropout),
    ).to(device)
    decoder = RLTokenDecoder(
        rl_token_dim=int(cfg.rlt.rl_token_dim),
        output_dim=input_dim,
        num_layers=int(cfg.rlt.num_decoder_layers),
        num_heads=int(cfg.rlt.num_heads),
        ff_dim=int(cfg.rlt.ff_dim),
        dropout=float(cfg.rlt.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_factor(
            step,
            total_steps=int(cfg.training.steps),
            warmup_steps=int(cfg.training.warmup_steps),
        ),
    )
    return encoder, decoder, optimizer, scheduler



def _distributed_state() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, rank, local_rank, world_size


def _module_state_dict(module):
    return module.module.state_dict() if isinstance(module, DistributedDataParallel) else module.state_dict()


def _log_scalar(value: float, device: torch.device, distributed: bool) -> float:
    if not distributed:
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return float(tensor.item())

def checkpoint_payload(cfg: Any, input_dim: int, global_step: int, encoder, decoder, optimizer, scheduler) -> dict[str, Any]:
    cfg_payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_payload, dict):
        raise TypeError("Stage 1 config must serialize to a dict")
    rlt_cfg = dict(cfg_payload.get("rlt", {}))
    rlt_cfg["input_dim"] = int(input_dim)
    cfg_payload["rlt"] = rlt_cfg
    return {
        "global_step": int(global_step),
        "encoder_state_dict": _module_state_dict(encoder),
        "decoder_state_dict": _module_state_dict(decoder),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg_payload,
    }


def main() -> None:
    cfg = parse_args()
    distributed, rank, local_rank, world_size = _distributed_state()
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    is_main = rank == 0

    np.random.seed(int(cfg.global_seed) + rank)
    torch.manual_seed(int(cfg.global_seed) + rank)

    lerobot_home = _optional_str(cfg.vla.get("lerobot_home", None))
    if lerobot_home:
        os.environ["HF_LEROBOT_HOME"] = str(Path(lerobot_home).expanduser().resolve())

    output_dir = Path(str(cfg.training.output_dir)).expanduser().resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    writer = SummaryWriter(log_dir=str(output_dir / "logs")) if is_main and SummaryWriter is not None else None
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(str(cfg.vla.device) if torch.cuda.is_available() else "cpu")

    encoder = decoder = optimizer = scheduler = None
    input_dim = None
    from vla_rl.policies.openpi.stage1 import OpenPIStage1Backend

    backend = OpenPIStage1Backend(openpi_root=_optional_str(cfg.vla.openpi_root))
    base_policy = backend.load_base_policy(
        config_name=str(cfg.vla.config_name),
        checkpoint_path=str(cfg.vla.checkpoint_path),
        device=device,
        assets_base_dir=_optional_str(cfg.vla.assets_base_dir),
        checkpoint_base_dir=_optional_str(cfg.vla.checkpoint_base_dir),
        exp_name=_optional_str(cfg.vla.exp_name),
    )
    dataloader = backend.create_dataloader(
        config_name=str(cfg.vla.config_name),
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        assets_base_dir=_optional_str(cfg.vla.assets_base_dir),
        checkpoint_base_dir=_optional_str(cfg.vla.checkpoint_base_dir),
        exp_name=_optional_str(cfg.vla.exp_name),
        repo_id_override=_optional_str(cfg.vla.get("repo_id_override", None)),
        shuffle=bool(cfg.training.shuffle),
    )
    data_iter = iter(dataloader)

    if is_main:
        with (output_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    steps = int(cfg.training.steps)
    max_tokens = normalize_max_tokens(cfg.rlt.get("max_tokens", None))
    feature = feature_cfg(cfg)
    feature_source = str(feature.get("source", "policy_prior_prefix"))
    feature_num_steps = int(feature.get("num_steps", cfg.vla.get("num_steps", 10)))
    metrics_path = output_dir / "metrics.jsonl"
    if is_main:
        metrics_path.write_text("")

    default_precision = "fp32" if distributed else ("bf16" if device.type == "cuda" else "fp32")
    precision = str(cfg.training.get("precision", default_precision))
    autocast_enabled = bool(device.type == "cuda" and precision == "bf16")

    if is_main:
        logger.info("Starting RLT Stage 1 online prefix reconstruction for %d steps", steps)
        logger.info("Stage 1 precision=%s autocast=%s distributed=%s world_size=%d", precision, autocast_enabled, distributed, world_size)
    progress = tqdm(range(steps), desc="rlt-stage1", dynamic_ncols=True, disable=not is_main)
    for global_step in progress:
        step_started_at = time.perf_counter()
        try:
            observation, actions = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            observation, actions = next(data_iter)

        obs_obj = backend.observation_to_device(observation, device)
        with torch.no_grad():
            if feature_source == "policy_prior_prefix":
                prefix, _ = base_policy.extract_policy_prior_prefix(obs_obj, num_steps=feature_num_steps)
            elif feature_source == "self_conditioned_prefix":
                prefix, _ = base_policy.extract_self_conditioned_prefix(obs_obj, num_steps=feature_num_steps)
            elif feature_source == "expert_conditioned_prefix":
                prefix = base_policy.extract_expert_conditioned_prefix(obs_obj, actions)
            else:
                raise ValueError(f"unsupported feature.source: {feature_source}")
            z_vla = _stage1_prefix_target(prefix, max_tokens=max_tokens)

        if encoder is None:
            input_dim = int(z_vla.shape[-1])
            encoder, decoder, optimizer, scheduler = _build_modules(cfg, input_dim, device)
            if distributed:
                encoder = DistributedDataParallel(encoder, device_ids=[local_rank])
                decoder = DistributedDataParallel(decoder, device_ids=[local_rank])
            if is_main:
                logger.info(
                    "Initialized Stage 1 modules: input_dim=%d rl_token_dim=%d tokens=%d",
                    input_dim,
                    int(cfg.rlt.rl_token_dim),
                    int(z_vla.shape[1]),
                )

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
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
        loss_value = _log_scalar(float(loss.item()), device, distributed)
        metric = {
            "step": global_step,
            "loss_reconstruction": loss_value,
            "lr": lr,
            "tokens": int(z_vla.shape[1]),
            "input_dim": int(z_vla.shape[-1]),
            "feature_source": feature_source,
            "step_time_sec": float(time.perf_counter() - step_started_at),
        }
        if is_main:
            with metrics_path.open("a") as f:
                f.write(json.dumps(metric, sort_keys=True) + "\n")
            if writer is not None:
                writer.add_scalar("loss/reconstruction", loss_value, global_step)
                writer.add_scalar("train/lr", lr, global_step)
            progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{lr:.3g}")
            if global_step % int(cfg.training.log_every) == 0:
                logger.info("step=%d/%d loss=%.6f lr=%.6g", global_step, steps, loss_value, lr)

        step_number = global_step + 1
        if is_main and int(cfg.training.save_every) > 0 and step_number % int(cfg.training.save_every) == 0:
            checkpoint_path = output_dir / f"checkpoint_{step_number}.pt"
            torch.save(checkpoint_payload(cfg, input_dim, step_number, encoder, decoder, optimizer, scheduler), checkpoint_path)
            logger.info("Saved checkpoint to %s", checkpoint_path)

    if encoder is None or decoder is None or optimizer is None or scheduler is None or input_dim is None:
        raise RuntimeError("Stage 1 did not run any optimization steps")

    if is_main:
        final_path = output_dir / "final_model.pt"
        torch.save(checkpoint_payload(cfg, input_dim, steps, encoder, decoder, optimizer, scheduler), final_path)
        torch.save(_module_state_dict(encoder), output_dir / "rlt_encoder.pt")
        if writer is not None:
            writer.close()
        logger.info("Stage 1 completed. Saved final checkpoint to %s", final_path)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
