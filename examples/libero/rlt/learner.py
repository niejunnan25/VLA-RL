from __future__ import annotations


def learner_should_stop(*, actor_done: bool) -> bool:
    return bool(actor_done)
