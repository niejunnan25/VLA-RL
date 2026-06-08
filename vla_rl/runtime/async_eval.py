from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, IO, Sequence

from vla_rl.runtime.agentlace import json_sanitize


@dataclass
class AsyncEvalRuntime:
    enabled: bool = False
    every_episodes: int = 0
    queue_path: Path | None = None
    summary_jsonl_path: Path | None = None
    worker_log_path: Path | None = None
    worker_proc: subprocess.Popen | None = None
    worker_log_fp: IO[str] | None = None
    eval_checkpoint_dir: Path | None = None
    processed_summary_lines: int = 0
    triggered_count: int = 0
    worker_dead_reported: bool = False


def resolve_async_eval_path(path_value: Any, *, run_dir: Path) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = run_dir / path
    return path


def count_jsonl_lines(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_sanitize(payload), sort_keys=True) + "\n")


def append_async_eval_request(async_eval: AsyncEvalRuntime, payload: dict[str, Any]) -> None:
    if not async_eval.enabled or async_eval.queue_path is None:
        return
    record = dict(payload)
    record["type"] = "eval"
    record.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    append_jsonl(async_eval.queue_path, record)
    async_eval.triggered_count += 1


def append_async_eval_stop(async_eval: AsyncEvalRuntime) -> None:
    if not async_eval.enabled or async_eval.queue_path is None:
        return
    append_jsonl(
        async_eval.queue_path,
        {"type": "stop", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())},
    )


def load_new_async_eval_results(async_eval: AsyncEvalRuntime) -> list[dict[str, Any]]:
    path = async_eval.summary_jsonl_path
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    start = int(async_eval.processed_summary_lines)
    if start < 0 or start > len(lines):
        start = 0
    records: list[dict[str, Any]] = []
    for line in lines[start:]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    async_eval.processed_summary_lines = len(lines)
    return records


def launch_async_eval_worker(
    *,
    cmd: Sequence[str],
    worker_log_path: Path,
) -> tuple[subprocess.Popen, IO[str]]:
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = worker_log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(list(cmd), stdout=log_fp, stderr=subprocess.STDOUT)
    return proc, log_fp


def check_async_eval_worker(async_eval: AsyncEvalRuntime) -> None:
    if not async_eval.enabled or async_eval.worker_proc is None or async_eval.worker_dead_reported:
        return
    return_code = async_eval.worker_proc.poll()
    if return_code is None:
        return
    async_eval.worker_dead_reported = True
    raise RuntimeError(f"async eval worker exited early with returncode={return_code}; see {async_eval.worker_log_path}")


def wait_for_async_eval_worker(async_eval: AsyncEvalRuntime, *, poll_interval_sec: float = 5.0) -> int | None:
    if not async_eval.enabled or async_eval.worker_proc is None:
        return None
    while True:
        return_code = async_eval.worker_proc.poll()
        if return_code is not None:
            if async_eval.worker_log_fp is not None:
                async_eval.worker_log_fp.close()
                async_eval.worker_log_fp = None
            return return_code
        time.sleep(max(0.1, float(poll_interval_sec)))
