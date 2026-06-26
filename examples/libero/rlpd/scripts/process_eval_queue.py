#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.libero.rlpd.config import load_config, validate_rlpd_cfg
from examples.libero.rlpd.scripts.eval_rlpd import run_eval
from vla_rl.runtime.agentlace import json_sanitize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process LIBERO RLPD async eval queue.")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--queue-file", required=True)
    parser.add_argument("--summary-jsonl", required=True)
    parser.add_argument("--poll-interval-sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.train_config, [])
    validate_rlpd_cfg(cfg)
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
            result = _run_request(cfg, request)
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


def _run_request(base_cfg, request: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = Path(str(request.get("checkpoint_path", "")))
    if not checkpoint_path.exists():
        return {
            "status": "failed",
            "eval_index": request.get("eval_index"),
            "error": f"checkpoint not found: {checkpoint_path}",
            "request": request,
        }

    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    if request.get("env_url"):
        cfg.env.url = str(request["env_url"])

    output_dir = Path(str(request["output_dir"]))
    try:
        summary = run_eval(
            cfg,
            checkpoint_path=checkpoint_path,
            episodes=int(request.get("episodes", 50)),
            output_dir=output_dir,
            max_env_steps_per_episode=int(request.get("max_env_steps_per_episode", 0) or 0),
            save_videos=bool(request.get("save_videos", False)),
            log_wandb=False,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "eval_index": request.get("eval_index"),
            "error": str(exc),
            "request": request,
        }

    train_episode = int(request.get("train_episode_id", 0) or 0)
    return {
        "status": "ok",
        "eval_index": request.get("eval_index"),
        "request": request,
        "eval/train_episode": train_episode,
        "eval/success_rate": summary["success_rate"],
        "eval/mean_return": summary["avg_return"],
        "eval/mean_steps": summary["avg_length"],
        "eval/episodes_run": summary["episodes"],
    }


if __name__ == "__main__":
    main()
