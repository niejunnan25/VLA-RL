from __future__ import annotations


def learner_should_stop(
    *,
    update_steps: int,
    env_steps: int,
    max_update_steps: int,
    max_env_steps: int,
    actor_done: bool,
) -> bool:
    if update_steps < max_update_steps:
        return False
    if max_env_steps <= 0:
        return True
    return env_steps >= max_env_steps or actor_done
