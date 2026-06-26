from __future__ import annotations

from typing import Any


def rlpd_rollout_metric_aliases(
    *,
    episode_id: int,
    episode_return: float,
    episode_steps: int,
    success: bool,
    recent_success_rate_50: float,
) -> dict[str, Any]:
    return {
        "rollout/episode_id": int(episode_id),
        "rollout/episode_return": float(episode_return),
        "rollout/episode_steps": int(episode_steps),
        "rollout/success": int(success),
        "rollout/recent_success_rate_50": float(recent_success_rate_50),
    }


def actor_episode_metric(
    *,
    episode_id: int,
    episode_return: float,
    episode_steps: int,
    success: bool,
    env_steps: int,
    recent_success_rate_50: float,
) -> dict[str, Any]:
    return {
        "environment": {
            "episode": {
                "episode_id": int(episode_id),
                "return": float(episode_return),
                "length": int(episode_steps),
                "success": bool(success),
                "env_steps": int(env_steps),
            }
        },
        **rlpd_rollout_metric_aliases(
            episode_id=episode_id,
            episode_return=episode_return,
            episode_steps=episode_steps,
            success=success,
            recent_success_rate_50=recent_success_rate_50,
        ),
    }


def actor_speed_stats(
    *,
    env_steps: int,
    episodes: int,
    successes: int,
    total_reward: float,
    wall_time_sec: float,
    active_rollout_time_sec: float,
) -> dict[str, Any]:
    return {
        "env_steps": int(env_steps),
        "episodes": int(episodes),
        "successes": int(successes),
        "total_reward": float(total_reward),
        "wall_time_sec": float(wall_time_sec),
        "active_env_steps_per_sec": int(env_steps) / max(float(active_rollout_time_sec), 1e-9),
        "wall_env_steps_per_sec": int(env_steps) / max(float(wall_time_sec), 1e-9),
    }


def rlpd_learner_metric_aliases(
    update_info: dict[str, Any],
    *,
    update_steps: int,
    env_steps: int,
    replay_size: int,
    offline_replay_size: int,
    batch_mix: dict[str, int] | None = None,
) -> dict[str, Any]:
    metric: dict[str, Any] = {
        "learner/update_steps": int(update_steps),
        "learner/env_steps": int(env_steps),
        "learner/replay_size": int(replay_size),
        "learner/offline_replay_size": int(offline_replay_size),
    }
    if batch_mix is not None:
        metric["learner/batch_online"] = int(batch_mix.get("online", 0))
        metric["learner/batch_offline"] = int(batch_mix.get("offline", 0))
    for key in (
        "loss_critic",
        "target_q_mean",
        "predicted_q_mean",
        "loss_actor",
        "actor_log_prob_mean",
        "loss_temperature",
        "temperature",
        "updates",
    ):
        if key in update_info:
            metric[f"learner/{key}"] = update_info[key]
    return metric
