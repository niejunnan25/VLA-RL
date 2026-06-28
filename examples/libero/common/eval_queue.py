#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import time
from typing import Any, Callable

from omegaconf import OmegaConf

from vla_rl.runtime.agentlace import json_sanitize


@dataclass(frozen=True)
class EvalQueueSpec:
    load_config: str
    validate_config: str
    run_eval: str
    apply_env_url: bool = False
    apply_policy_url: bool = False
    supports_force_zero_residual: bool = False
    include_residual_metrics: bool = False


EVAL_QUEUE_SPECS = {
    "pld": EvalQueueSpec(
        load_config="examples.libero.pld.config.load_config",
        validate_config="examples.libero.pld.config.validate_pld_cfg",
        run_eval="examples.libero.pld.scripts.eval_residual_sac.run_eval",
        apply_env_url=True,
        apply_policy_url=True,
        supports_force_zero_residual=True,
        include_residual_metrics=True,
    ),
    "rlpd": EvalQueueSpec(
        load_config="examples.libero.rlpd.config.load_config",
        validate_config="examples.libero.rlpd.config.validate_rlpd_cfg",
        run_eval="examples.libero.rlpd.scripts.eval_rlpd.run_eval",
    ),
    "rlt": EvalQueueSpec(
        load_config="examples.libero.rlt.config.load_config",
        validate_config="examples.libero.rlt.config.validate_rlt_cfg",
        run_eval="examples.libero.rlt.scripts.eval_stage2.run_eval",
        apply_env_url=True,
        apply_policy_url=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a LIBERO async eval queue.")
    parser.add_argument("--algorithm", required=True, choices=sorted(EVAL_QUEUE_SPECS))
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--queue-file", required=True)
    parser.add_argument("--summary-jsonl", required=True)
    parser.add_argument("--poll-interval-sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_eval_queue(args)


def process_eval_queue(args: argparse.Namespace) -> None:
    spec = EVAL_QUEUE_SPECS[str(args.algorithm)]
    load_config = _load_object(spec.load_config)
    validate_config = _load_object(spec.validate_config)
    run_eval = _load_object(spec.run_eval)

    cfg = load_config(args.train_config, [])
    validate_config(cfg)
    queue_path = Path(args.queue_file)
    summary_path = Path(args.summary_jsonl)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0

    while True:
        requests, processed = _read_new_requests(queue_path, processed)
        if not requests:
            time.sleep(float(args.poll_interval_sec))
            continue
        for request in requests:
            if request.get("type") == "stop":
                return
            result = _run_request(cfg, request, spec=spec, run_eval=run_eval)
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(json_sanitize(result), sort_keys=True) + "\n")


def _read_new_requests(queue_path: Path, processed: int) -> tuple[list[dict[str, Any]], int]:
    if not queue_path.exists():
        return [], processed
    records: list[dict[str, Any]] = []
    line_count = 0
    with queue_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_count = idx + 1
            if idx < processed:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records, line_count


def _run_request(
    base_cfg: Any,
    request: dict[str, Any],
    *,
    spec: EvalQueueSpec,
    run_eval: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    checkpoint_path = Path(str(request.get("checkpoint_path", "")))
    if not checkpoint_path.exists():
        return {
            "status": "failed",
            "eval_index": request.get("eval_index"),
            "error": f"checkpoint not found: {checkpoint_path}",
            "request": request,
        }

    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    if spec.apply_env_url and request.get("env_url"):
        cfg.env.url = str(request["env_url"])
    if spec.apply_policy_url and request.get("policy_url"):
        cfg.policy.url = str(request["policy_url"])

    try:
        eval_kwargs: dict[str, Any] = {
            "checkpoint_path": checkpoint_path,
            "episodes": int(request.get("episodes", 50)),
            "output_dir": Path(str(request["output_dir"])),
            "max_env_steps_per_episode": int(request.get("max_env_steps_per_episode", 0) or 0),
            "save_videos": bool(request.get("save_videos", False)),
            "log_wandb": False,
        }
        if spec.supports_force_zero_residual:
            eval_kwargs["force_zero_residual"] = bool(request.get("force_zero_residual", False))
        summary = run_eval(cfg, **eval_kwargs)
    except Exception as exc:
        return {
            "status": "failed",
            "eval_index": request.get("eval_index"),
            "error": str(exc),
            "request": request,
        }

    train_episode = int(request.get("train_episode_id", 0) or 0)
    result = {
        "status": "ok",
        "eval_index": request.get("eval_index"),
        "request": request,
        "eval/train_episode": train_episode,
        "eval/success_rate": summary["success_rate"],
        "eval/mean_return": summary["avg_return"],
        "eval/mean_steps": summary["avg_length"],
        "eval/episodes_run": summary["episodes"],
    }
    if spec.include_residual_metrics:
        result.update(
            {
                "eval/residual_l1": summary.get("avg_residual_l1", 0.0),
                "eval/residual_l2": summary.get("avg_residual_l2", 0.0),
                "eval/force_zero_residual": bool(summary.get("force_zero_residual", False)),
            }
        )
    return result


def _load_object(path: str) -> Any:
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


if __name__ == "__main__":
    main()
