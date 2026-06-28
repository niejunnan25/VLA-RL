from __future__ import annotations

from typing import Any


def pld_rollout_metric_aliases(
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
        **pld_rollout_metric_aliases(
            episode_id=episode_id,
            episode_return=episode_return,
            episode_steps=episode_steps,
            success=success,
            recent_success_rate_50=recent_success_rate_50,
        ),
    }


def residual_sac_learner_metric_aliases(
    update_info: dict[str, Any],
    *,
    update_steps: int,
    env_steps: int,
    replay_size: int,
) -> dict[str, Any]:
    metric: dict[str, Any] = {
        "learner/update_steps": int(update_steps),
        "learner/env_steps": int(env_steps),
        "learner/replay_size": int(replay_size),
    }
    for key in (
        "loss_critic",
        "loss_actor",
        "target_q_mean",
        "predicted_q_mean",
        "temperature",
    ):
        if key in update_info:
            metric[f"learner/{key}"] = update_info[key]
    return metric
