from __future__ import annotations

import numpy as np

from vla_rl.data import MemoryEfficientReplayBuffer
from vla_rl.data import Transition


def _obs(value: int) -> dict[str, np.ndarray]:
    return {
        "image_image_rgb_0": np.full((3, 8, 8), value, dtype=np.uint8),
        "image_image_rgb_1": np.full((3, 8, 8), value + 1, dtype=np.uint8),
        "proprio": np.full((4,), float(value), dtype=np.float32),
    }


def _transition(index: int, *, done: bool = False) -> Transition:
    return Transition(
        obs=_obs(index),
        next_obs=_obs(index + 1),
        action=np.full((2,), float(index), dtype=np.float32),
        reward=float(done),
        done=done,
        truncated=False,
        discount=0.0 if done else 0.99,
        executed_steps=1,
        env_steps=index + 1,
        info={"critic_terminal": bool(done)},
    )


def test_memory_efficient_replay_preserves_uint8_images_and_reconstructs_next_images() -> None:
    replay = MemoryEfficientReplayBuffer(capacity=8, seed=0)
    replay.add(_transition(0))
    replay.add(_transition(1))

    transition = next(replay.iter_transitions())

    assert transition.obs["image_image_rgb_0"].dtype == np.uint8
    assert transition.next_obs is not None
    np.testing.assert_array_equal(transition.next_obs["image_image_rgb_0"], _obs(1)["image_image_rgb_0"])
    np.testing.assert_array_equal(transition.next_obs["proprio"], _obs(1)["proprio"])


def test_memory_efficient_replay_wraps_as_fifo_ring_buffer() -> None:
    replay = MemoryEfficientReplayBuffer(capacity=2, seed=0)
    replay.add(_transition(0))
    replay.add(_transition(1))
    replay.add(_transition(2, done=True))

    assert len(replay) == 2
    assert [transition.env_steps for transition in replay.iter_transitions()] == [2, 3]
    batch = replay.sample(4)
    assert {transition.env_steps for transition in batch.transitions}.issubset({2, 3})
