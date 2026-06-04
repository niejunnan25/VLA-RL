from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from vla_rl.runtime.agentlace import json_sanitize


def run_dir_from_runtime(runtime: Any) -> Path | None:
    value = runtime.get("run_dir", None)
    return Path(value) if value else None


def next_interval(current: int, interval: int) -> int:
    if interval <= 0:
        return 0
    return ((int(current) // int(interval)) + 1) * int(interval)


def runtime_float(runtime: Any, key: str, default: float) -> float:
    return float(runtime.get(key, default))


def make_jsonl_metric_writer(run_dir: Path | None, filename: str) -> Callable[[dict[str, Any]], None]:
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / filename).write_text("")

    def write_metric(metric: dict[str, Any]) -> None:
        if run_dir is None:
            return
        with (run_dir / filename).open("a") as f:
            f.write(json.dumps(json_sanitize(metric), sort_keys=True) + "\n")

    return write_metric


def _write_actor_summary(run_dir: Path | None, summary: dict[str, Any]) -> None:
    if run_dir is None:
        return
    (run_dir / "actor_summary.json").write_text(json.dumps(json_sanitize(summary), indent=2, sort_keys=True) + "\n")


def read_actor_summary(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    path = run_dir / "actor_summary.json"
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def apply_actor_summary_file(run_dir: Path | None, actor_done: bool, actor_done_env_steps: int) -> tuple[bool, int]:
    summary = read_actor_summary(run_dir)
    if summary is None:
        return actor_done, actor_done_env_steps
    return True, max(int(actor_done_env_steps), int(summary.get("env_steps", 0)))


def send_actor_summary(
    client: Any,
    request_type: str,
    summary: dict[str, Any],
    run_dir: Path | None,
    write_metric: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    final_summary = dict(summary)
    final_summary["actor_summary_notified"] = False
    _write_actor_summary(run_dir, final_summary)
    try:
        client.request(str(request_type), {"event": "actor_summary", **final_summary})
    except Exception as exc:
        metric = {
            "role": final_summary.get("role", "actor"),
            "event": "actor_summary_send_failed",
            "env_steps": int(final_summary.get("env_steps", 0)),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if "algorithm" in final_summary:
            metric["algorithm"] = final_summary["algorithm"]
        write_metric(metric)
    else:
        final_summary["actor_summary_notified"] = True
        metric = {
            "role": final_summary.get("role", "actor"),
            "event": "actor_summary_sent",
            "env_steps": int(final_summary.get("env_steps", 0)),
        }
        if "algorithm" in final_summary:
            metric["algorithm"] = final_summary["algorithm"]
        write_metric(metric)
    _write_actor_summary(run_dir, final_summary)
    return final_summary


def save_checkpoint(
    checkpoints: Any,
    algorithm: Any,
    env_steps: int,
    update_steps: int,
    episodes: int,
    total_reward: float,
    config: dict[str, Any],
    tag: str | None = None,
) -> None:
    checkpoints.save(
        algorithm,
        env_steps=env_steps,
        update_steps=update_steps,
        episodes=episodes,
        total_reward=total_reward,
        config=config,
        tag=tag,
    )
