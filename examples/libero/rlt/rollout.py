from __future__ import annotations

from typing import Any

import numpy as np

from vla_rl.data import Transition


def subsample_observation_steps(action_steps: int, chunk_size: int, subsample_stride: int) -> list[int]:
    if subsample_stride <= 1:
        return []
    return [
        step
        for step in range(subsample_stride, action_steps + 1, subsample_stride)
        if step < chunk_size
    ]


def reference_actions_for_chunk(actions: np.ndarray, chunk_size: int, stride: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if stride > 1:
        return actions[: chunk_size * stride : stride]
    return actions[:chunk_size]


def predict_reference_actions_for_chunk(
    reference_policy: Any,
    observation: Any,
    *,
    feature_source: str,
    num_steps: int,
    chunk_size: int,
    action_stride: int,
) -> tuple[np.ndarray, Any, Any]:
    base_actions, prefix_tokens, proprio = reference_policy.predict_actions_and_prefix(
        observation,
        feature_source=feature_source,
        num_steps=num_steps,
    )
    base_actions = reference_actions_for_chunk(base_actions, chunk_size, action_stride)
    return base_actions, prefix_tokens, proprio


def transition_discount(*, gamma: float, executed_steps: int, critic_terminal: bool) -> float:
    if critic_terminal:
        return 0.0
    return float(gamma) ** int(executed_steps)


def step_info_success(info: dict[str, Any]) -> bool:
    return bool(
        info.get("success", False)
        or info.get("env_done", False)
        or info.get("is_success", False)
    )


def should_collect_window_start(
    *,
    window_replay_enabled: bool,
    done: bool,
    truncated: bool,
    executed_steps: int,
    subsample_stride: int,
    chunk_size: int,
) -> bool:
    if not window_replay_enabled or bool(done or truncated):
        return False
    return executed_steps % subsample_stride == 0 and executed_steps < chunk_size


def make_window_replay_chunk(
    *,
    rlt_obs: dict[str, np.ndarray],
    window_start_rlt_obs: list[dict[str, np.ndarray]],
    actions: np.ndarray,
    reward: float,
    done: bool,
    truncated: bool,
    terminal: bool,
    critic_terminal: bool,
    executed_steps: int,
    env_steps: int,
    info: dict[str, Any],
    chunk_start_env_steps: int,
    replan_steps: int,
) -> dict[str, Any]:
    return {
        "rlt_obs": rlt_obs,
        "window_start_rlt_obs": window_start_rlt_obs,
        "actions": np.asarray(actions, dtype=np.float32).copy(),
        "next_actions": None,
        "reward": float(reward),
        "done": bool(done),
        "truncated": bool(truncated),
        "terminal": bool(terminal),
        "critic_terminal": bool(critic_terminal),
        "executed_steps": int(executed_steps),
        "env_steps": int(env_steps),
        "info": dict(info),
        "chunk_start_env_steps": int(chunk_start_env_steps),
        "replan_steps": int(replan_steps),
    }


def make_chunk_transition(
    *,
    rlt_obs: dict[str, np.ndarray],
    next_rlt_obs: dict[str, np.ndarray],
    actions: np.ndarray,
    reward: float,
    done: bool,
    truncated: bool,
    gamma: float,
    critic_terminal: bool,
    executed_steps: int,
    env_steps: int,
    info: dict[str, Any],
    chunk_start_env_steps: int,
    replan_steps: int,
) -> Transition:
    return Transition(
        obs=rlt_obs,
        next_obs=next_rlt_obs,
        action=np.asarray(actions, dtype=np.float32).reshape(-1),
        reward=float(reward),
        done=bool(done),
        truncated=bool(truncated),
        discount=transition_discount(
            gamma=float(gamma),
            executed_steps=int(executed_steps),
            critic_terminal=bool(critic_terminal),
        ),
        executed_steps=int(executed_steps),
        env_steps=int(env_steps),
        info={
            **dict(info),
            "chunk_start_env_steps": int(chunk_start_env_steps),
            "replan_steps": int(replan_steps),
            "critic_terminal": bool(critic_terminal),
        },
    )


def insert_window_replay_transitions(
    pending_chunk: dict[str, Any],
    *,
    next_rlt_obs: dict[str, np.ndarray],
    next_window_start_rlt_obs: list[dict[str, np.ndarray]],
    data_store: Any,
    subsample_stride: int,
    chunk_size: int,
    gamma: float,
) -> int:
    """Insert yixin-style RLT replay windows for one completed chunk."""

    action_chunk = np.asarray(pending_chunk["actions"], dtype=np.float32)
    next_action_chunk = pending_chunk.get("next_actions")
    inserted = 0

    for pos_idx, position in enumerate(range(0, chunk_size, subsample_stride)):
        if position == 0:
            obs = pending_chunk["rlt_obs"]
            next_obs = next_rlt_obs
            action = action_chunk
        else:
            sub_idx = pos_idx - 1
            if (
                sub_idx >= len(pending_chunk["window_start_rlt_obs"])
                or sub_idx >= len(next_window_start_rlt_obs)
            ):
                break

            obs = pending_chunk["window_start_rlt_obs"][sub_idx]
            next_obs = next_window_start_rlt_obs[sub_idx]
            tail = action_chunk[position:]

            if next_action_chunk is None:
                head = np.zeros((position, action_chunk.shape[1]), dtype=np.float32)
            else:
                head = np.asarray(next_action_chunk, dtype=np.float32)[:position]

            action = np.concatenate([tail, head], axis=0)

        critic_terminal = bool(pending_chunk.get("critic_terminal", pending_chunk["terminal"]))
        transition = Transition(
            obs=obs,
            next_obs=next_obs,
            action=action.reshape(-1),
            reward=float(pending_chunk["reward"]),
            done=bool(pending_chunk["done"]),
            truncated=bool(pending_chunk["truncated"]),
            discount=0.0 if critic_terminal else float(gamma) ** int(chunk_size),
            executed_steps=int(pending_chunk["executed_steps"]),
            env_steps=int(pending_chunk["env_steps"]),
            info={
                **dict(pending_chunk["info"]),
                "chunk_start_env_steps": int(pending_chunk["chunk_start_env_steps"]),
                "subsample_stride": int(subsample_stride),
                "subsample_position": int(position),
                "critic_terminal": critic_terminal,
            },
        )
        data_store.insert(transition.to_payload())
        inserted += 1

    return inserted
