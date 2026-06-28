from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Literal

from omegaconf import DictConfig

from vla_rl.runtime.async_eval import (
    AsyncEvalRuntime,
    count_jsonl_lines,
    launch_async_eval_worker,
    resolve_async_eval_path,
)


AlgorithmName = Literal["pld", "rlpd", "rlt"]


_ASYNC_EVAL_REQUIRES_POLICY_URL = {
    "pld": True,
    "rlpd": False,
    "rlt": True,
}


def start_async_eval_worker(
    runtime: DictConfig,
    *,
    run_dir: Path | None,
    algorithm: AlgorithmName,
) -> AsyncEvalRuntime:
    if algorithm not in _ASYNC_EVAL_REQUIRES_POLICY_URL:
        raise ValueError(f"unsupported async eval algorithm: {algorithm}")

    async_cfg = runtime.get("async_eval", None)
    if async_cfg is None or not bool(async_cfg.get("enabled", False)):
        return AsyncEvalRuntime()
    if run_dir is None:
        raise ValueError("runtime.run_dir is required when runtime.async_eval.enabled=true")
    if _ASYNC_EVAL_REQUIRES_POLICY_URL[algorithm] and not async_cfg.get("policy_url", None):
        raise ValueError("runtime.async_eval.policy_url is required when async eval is enabled")

    every_episodes = int(async_cfg.get("every_episodes", 50))
    if every_episodes <= 0:
        raise ValueError("runtime.async_eval.every_episodes must be positive")

    queue_path = resolve_async_eval_path(async_cfg.get("queue_file", "eval_queue.jsonl"), run_dir=run_dir)
    summary_path = resolve_async_eval_path(async_cfg.get("summary_jsonl", "eval_summary.jsonl"), run_dir=run_dir)
    worker_log_path = resolve_async_eval_path(async_cfg.get("worker_log_file", "eval_worker.log"), run_dir=run_dir)
    eval_checkpoint_dir = resolve_async_eval_path(async_cfg.get("checkpoint_dir", "eval_checkpoints"), run_dir=run_dir)

    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text("")
    summary_path.touch(exist_ok=True)

    worker_proc, worker_log_fp = launch_async_eval_worker(
        cmd=[
            sys.executable,
            "-m",
            "examples.libero.common.eval_queue",
            "--algorithm",
            algorithm,
            "--train-config",
            str(run_dir / "config.yaml"),
            "--queue-file",
            str(queue_path),
            "--summary-jsonl",
            str(summary_path),
            "--poll-interval-sec",
            str(float(async_cfg.get("poll_interval_sec", 5.0))),
        ],
        worker_log_path=worker_log_path,
        env=_async_eval_worker_env(async_cfg),
    )

    return AsyncEvalRuntime(
        enabled=True,
        every_episodes=every_episodes,
        queue_path=queue_path,
        summary_jsonl_path=summary_path,
        worker_log_path=worker_log_path,
        worker_proc=worker_proc,
        worker_log_fp=worker_log_fp,
        eval_checkpoint_dir=eval_checkpoint_dir,
        processed_summary_lines=count_jsonl_lines(summary_path),
    )


def _async_eval_worker_env(async_cfg: DictConfig) -> dict[str, str]:
    env = dict(os.environ)
    project_root = Path(__file__).resolve().parents[3]
    existing_pythonpath = env.get("PYTHONPATH", "")
    paths = [str(project_root)]
    if existing_pythonpath:
        paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(paths)

    cuda_visible_devices = async_cfg.get("worker_cuda_visible_devices", None)
    mujoco_egl_device_id = async_cfg.get("worker_mujoco_egl_device_id", None)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    if mujoco_egl_device_id is not None:
        env["MUJOCO_EGL_DEVICE_ID"] = str(mujoco_egl_device_id)
    return env
