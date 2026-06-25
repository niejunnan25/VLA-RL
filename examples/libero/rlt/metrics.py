from __future__ import annotations

from typing import Any


def rlt_rollout_metric_aliases(
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
                "return": float(episode_return),
                "length": int(episode_steps),
                "success": bool(success),
                "env_steps": int(env_steps),
            }
        },
        **rlt_rollout_metric_aliases(
            episode_id=episode_id,
            episode_return=episode_return,
            episode_steps=episode_steps,
            success=success,
            recent_success_rate_50=recent_success_rate_50,
        ),
    }


def rlt_learner_metric_aliases(
    update_info: dict[str, Any],
    *,
    update_steps: int,
    env_steps: int,
    replay_size: int,
) -> dict[str, Any]:
    metric = {
        "learner/update_steps": int(update_steps),
        "learner/env_steps": int(env_steps),
        "learner/replay_size": int(replay_size),
    }
    for key in (
        "loss_critic",
        "target_q_mean",
        "predicted_q_mean",
        "loss_actor",
        "bc_loss",
        "q_value_mean",
    ):
        if key in update_info:
            metric[f"learner/{key}"] = update_info[key]
    return metric


def actor_speed_stats(
    *,
    env_steps: int,
    episodes: int,
    total_reward: float,
    wall_time_sec: float,
    active_rollout_time_sec: float,
) -> dict[str, Any]:
    return {
        "env_steps": int(env_steps),
        "episodes": int(episodes),
        "total_reward": float(total_reward),
        "wall_time_sec": float(wall_time_sec),
        "active_env_steps_per_sec": int(env_steps) / max(float(active_rollout_time_sec), 1e-9),
        "wall_env_steps_per_sec": int(env_steps) / max(float(wall_time_sec), 1e-9),
    }


def actor_chunk_metric(
    *,
    env_steps: int,
    episode: int,
    reward: float,
    done: bool,
    truncated: bool,
    executed_steps: int,
    chunk_size: int,
    replay_transitions: int,
    submitted_transitions: int,
    wall_time_sec: float,
    chunk_time_sec: float,
    reset_time_sec: float,
    reward_stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": "actor",
        "env_steps": int(env_steps),
        "episode": int(episode),
        "reward": float(reward),
        "done": bool(done),
        "truncated": bool(truncated),
        "executed_steps": int(executed_steps),
        "chunk_size": int(chunk_size),
        "replay_transitions": int(replay_transitions),
        "submitted_transitions": int(submitted_transitions),
        "wall_time_sec": float(wall_time_sec),
        "chunk_time_sec": float(chunk_time_sec),
        "time/reset_env_sec": float(reset_time_sec),
        **{f"reward/{key}": value for key, value in reward_stats.items()},
    }
