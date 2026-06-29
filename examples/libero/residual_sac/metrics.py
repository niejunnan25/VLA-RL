from __future__ import annotations

"""SERL-style cloud metric aliases for LIBERO residual SAC.

The training loop keeps detailed timers and structured payloads in local JSONL
files. This module defines the small metric surface that should go to SwanLab
so dashboards stay readable and comparable with the older PLD runs.
"""

from collections.abc import Mapping
from typing import Any


def rollout_metric_aliases(
    payload: Mapping[str, Any],
    *,
    actor_env_steps_per_sec: float | None = None,
) -> dict[str, Any]:
    rollout = _mapping(payload.get("rollout"))
    reward = _mapping(payload.get("reward"))
    residual = _mapping(payload.get("residual"))
    if not rollout:
        return {}

    metrics: dict[str, Any] = {
        "rollout/episode_id": _int(rollout.get("episode_id")),
        "rollout/episode_return": _float(rollout.get("episode_return")),
        "rollout/episode_steps": _int(rollout.get("episode_steps")),
        "rollout/success": _int_bool(rollout.get("success")),
        "rollout/cumulative_success_rate": _float(
            rollout.get("cumulative_success_rate")
        ),
        "rollout/env_steps": _int(payload.get("env_steps")),
    }

    recent_50 = _float(rollout.get("recent_success_rate_50"))
    if recent_50 is not None:
        metrics["rollout/recent_success_rate_50"] = recent_50

    if actor_env_steps_per_sec is not None:
        metrics["rollout/actor_env_steps_per_sec"] = float(actor_env_steps_per_sec)

    for source_key, metric_key in (
        ("episode_env_return", "rollout/reward_env_return"),
        ("episode_train_return", "rollout/reward_train_return"),
        ("last_progress", "rollout/reward_last_progress"),
        ("last_rpc_sec", "rollout/reward_last_rpc_sec"),
        ("rpc_errors", "rollout/reward_rpc_errors"),
    ):
        value = _float(reward.get(source_key))
        if value is not None:
            metrics[metric_key] = value

    for source_key, metric_key in (
        ("mean_abs", "rollout/residual_mean_abs"),
        ("max_abs", "rollout/residual_max_abs"),
        ("saturation_rate", "rollout/residual_saturation_rate"),
        ("action_delta_mean_abs", "rollout/action_delta_mean_abs"),
        ("action_delta_max_abs", "rollout/action_delta_max_abs"),
    ):
        value = _float(residual.get(source_key))
        if value is not None:
            metrics[metric_key] = value

    return _drop_none(metrics)


def learner_metric_aliases(
    update_info: Mapping[str, Any],
    *,
    update_steps: int,
    env_steps: int,
    replay_size: int,
    updates_per_sec: float | None = None,
    eval_queue_backlog: int | None = None,
    batch_mix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "learner/update_steps": int(update_steps),
        "learner/env_steps": int(env_steps),
        "learner/replay_size": int(replay_size),
    }
    if updates_per_sec is not None:
        metrics["learner/updates_per_sec"] = float(updates_per_sec)
    if eval_queue_backlog is not None:
        metrics["learner/eval_queue_backlog"] = int(eval_queue_backlog)

    for source_key, metric_key in (
        ("critic_loss", "learner/loss_critic"),
        ("critic_td_loss", "learner/loss_critic_td"),
        ("actor_loss", "learner/loss_actor"),
        ("predicted_qs", "learner/q_predicted_mean"),
        ("target_qs", "learner/q_target_mean"),
        ("actor_predicted_q", "learner/q_actor_predicted_mean"),
        ("predicted_q_gap", "learner/q_predicted_gap"),
        ("temperature", "learner/temperature"),
        ("entropy", "learner/entropy"),
    ):
        value = _float(update_info.get(source_key))
        if value is not None:
            metrics[metric_key] = value

    if batch_mix is not None:
        online = _float(batch_mix.get("online_batch_size"))
        offline = _float(batch_mix.get("offline_batch_size"))
        if online is not None:
            metrics["learner/replay_sample_online_count"] = online
        if offline is not None:
            metrics["learner/replay_sample_offline_count"] = offline
        if online is not None and offline is not None and online + offline > 0:
            metrics["learner/replay_offline_ratio_actual"] = offline / (online + offline)

    return _drop_none(metrics)


def eval_metric_aliases(
    payload: Mapping[str, Any],
    *,
    eval_queue_backlog: int | None = None,
) -> dict[str, Any]:
    summary = _mapping(payload.get("summary"))
    status = str(payload.get("status", "")).strip().lower()
    train_episode = _int(payload.get("train_episode_id"))
    if train_episode is None:
        return {}

    metrics: dict[str, Any] = {
        "eval/train_episode": train_episode,
        "eval/status_failed": 0 if status == "ok" else 1,
        "eval/eval_index": _int(payload.get("eval_index")),
        "eval/train_update_step": _int(payload.get("train_update_step")),
        "eval/train_env_step": _int(payload.get("train_env_step")),
        "eval/duration_sec": _float(payload.get("duration_sec")),
    }
    if eval_queue_backlog is not None:
        metrics["eval/queue_backlog"] = int(eval_queue_backlog)

    for source_key, metric_key in (
        ("success_rate", "eval/success_rate"),
        ("mean_return", "eval/mean_return"),
        ("mean_episode_steps", "eval/mean_steps"),
        ("episodes_completed", "eval/episodes_completed"),
        ("env_steps", "eval/env_steps"),
        ("policy_requests", "eval/policy_requests"),
        ("policy_samples", "eval/policy_samples"),
        ("policy_requests_per_env_step", "eval/policy_requests_per_env_step"),
    ):
        value = _float(summary.get(source_key))
        if value is not None:
            metrics[metric_key] = value

    return _drop_none(metrics)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _drop_none(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in metrics.items() if value is not None}


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _int_bool(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))
    return None
