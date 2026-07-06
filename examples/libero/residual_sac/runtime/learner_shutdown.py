from __future__ import annotations

"""Learner shutdown predicates for LIBERO residual training."""

from typing import Any
from typing import Mapping


ACTOR_DONE_ASYNC_EVAL_DRAINED_STOP_REASON = "actor_done_async_eval_drained"
ACTOR_DONE_DATA_COMMITTED_STOP_REASON = ACTOR_DONE_ASYNC_EVAL_DRAINED_STOP_REASON


def actor_done_ready_for_async_eval_shutdown(
    *,
    actor_done: bool,
    target_env_steps: int,
) -> bool:
    """Return True once actor rollout is done and learner should drain eval."""

    if int(target_env_steps) <= 0:
        return False
    return bool(actor_done)


def actor_done_data_committed(
    *,
    actor_done: bool,
    target_env_steps: int,
    latest_data_id: int,
    transport_status: Mapping[str, Any] | None = None,
    require_transport_commit: bool = True,
) -> bool:
    """Compatibility wrapper for the old shutdown predicate name.

    Replay data ids and trainer transport update ids are not env-step counters,
    so they must not gate learner shutdown after the actor has finished.
    """

    _ = latest_data_id
    _ = transport_status
    _ = require_transport_commit
    return actor_done_ready_for_async_eval_shutdown(
        actor_done=actor_done,
        target_env_steps=target_env_steps,
    )


__all__ = [
    "ACTOR_DONE_ASYNC_EVAL_DRAINED_STOP_REASON",
    "ACTOR_DONE_DATA_COMMITTED_STOP_REASON",
    "actor_done_ready_for_async_eval_shutdown",
    "actor_done_data_committed",
]
