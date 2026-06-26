from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from examples.libero.rlpd.async_eval import start_async_eval_worker
from vla_rl.algorithms.rlpd import SACAgent
from vla_rl.data import ReplayBuffer, Transition


def make_rlpd_obs(value: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "image_image_rgb_0": np.full((3, 16, 16), float(value), dtype=np.float32),
        "proprio": np.full((3,), float(value), dtype=np.float32),
    }


def make_transition(index: int) -> Transition:
    return Transition(
        obs=make_rlpd_obs(float(index)),
        next_obs=make_rlpd_obs(float(index + 1)),
        action=np.zeros((2,), dtype=np.float32),
        reward=0.0,
        done=False,
        truncated=False,
        discount=0.99,
        executed_steps=1,
        env_steps=index + 1,
        info={"critic_terminal": False},
    )


def make_agent(*, utd_ratio: int) -> SACAgent:
    return SACAgent(
        image_keys=("image_rgb_0",),
        proprio_dim=3,
        action_dim=2,
        image_encoder_type="small",
        image_feature_dim=8,
        vector_latent_dim=4,
        encoder_hidden_dim=16,
        actor_hidden_dims=(16,),
        critic_hidden_dims=(16,),
        utd_ratio=utd_ratio,
        critic_actor_ratio=1,
        drq_random_crop=True,
        drq_padding=2,
        device="cpu",
    )


def make_replay(batch_size: int = 4) -> ReplayBuffer:
    replay = ReplayBuffer(capacity=16, seed=0)
    for index in range(batch_size):
        replay.add(make_transition(index))
    return replay


def test_rlpd_agent_update_supports_utd_ratio_one() -> None:
    agent = make_agent(utd_ratio=1)
    replay = make_replay(batch_size=4)

    info = agent.update(replay.sample(4))
    action = agent.sample_action(make_rlpd_obs(), deterministic=True)

    assert info["critic_updates"] == 1.0
    assert info["actor_updates"] == 1.0
    assert action.shape == (1, 2)


def test_rlpd_agent_update_supports_high_utd_split() -> None:
    agent = make_agent(utd_ratio=2)
    replay = make_replay(batch_size=4)

    info = agent.update(replay.sample(4))

    assert info["critic_updates"] == 2.0
    assert info["actor_updates"] == 1.0


def test_rlpd_async_eval_requires_dedicated_env_url(tmp_path: Path) -> None:
    runtime = OmegaConf.create({"async_eval": {"enabled": True, "every_episodes": 50}})

    with pytest.raises(ValueError, match="runtime.async_eval.env_url is required"):
        start_async_eval_worker(runtime, run_dir=tmp_path)
