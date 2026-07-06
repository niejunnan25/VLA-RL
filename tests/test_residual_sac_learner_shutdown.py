from examples.libero.residual_sac.runtime.learner_shutdown import (
    ACTOR_DONE_ASYNC_EVAL_DRAINED_STOP_REASON,
)
from examples.libero.residual_sac.runtime.learner_shutdown import (
    actor_done_data_committed,
)
from examples.libero.residual_sac.runtime.learner_shutdown import (
    actor_done_ready_for_async_eval_shutdown,
)


def test_actor_done_shutdown_ignores_replay_and_transport_update_ids():
    assert (
        ACTOR_DONE_ASYNC_EVAL_DRAINED_STOP_REASON
        == "actor_done_async_eval_drained"
    )
    assert actor_done_ready_for_async_eval_shutdown(
        actor_done=True,
        target_env_steps=300_000,
    )
    assert actor_done_data_committed(
        actor_done=True,
        target_env_steps=300_000,
        latest_data_id=63_000,
        transport_status={
            "accepted_update_id": 63_000,
            "committed_update_id": 63_000,
        },
    )


def test_actor_done_shutdown_waits_for_actor_done():
    assert not actor_done_ready_for_async_eval_shutdown(
        actor_done=False,
        target_env_steps=300_000,
    )
    assert not actor_done_ready_for_async_eval_shutdown(
        actor_done=True,
        target_env_steps=0,
    )
