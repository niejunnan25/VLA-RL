from pathlib import Path

import numpy as np

from vla_rl.algorithms.rlt import RLTAgent, RLTFeatureProcessor, RLTokenEncoder
from vla_rl.data import CompactReplayBuffer, CompactTransition
from vla_rl.envs.fake import FakeEnvBackend
from vla_rl.policies.fake import FakePolicyBackend
from vla_rl.runtime.agentlace import AgentlaceActorRuntime, AgentlaceLearnerRuntime


def make_agent_obs(value: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "z_rl": np.full((16,), value, dtype=np.float32),
        "reference_action": np.zeros((8,), dtype=np.float32),
        "proprio": np.zeros((3,), dtype=np.float32),
    }


def test_compact_transition_replay_samples_rlt_batch():
    replay = CompactReplayBuffer(capacity=8, seed=0)
    replay.add(
        CompactTransition(
            agent_obs=make_agent_obs(0.0),
            next_agent_obs=make_agent_obs(1.0),
            action=np.zeros((8,), dtype=np.float32),
            reward=1.0,
            done=False,
            truncated=False,
            discount=0.99**2,
            executed_steps=2,
            env_steps=2,
            info={"source": "test"},
        )
    )

    batch = replay.sample(1)
    transition = batch.transitions[0]

    assert len(replay) == 1
    assert replay.latest_env_steps == 2
    assert transition.agent_obs is not None
    assert transition.next_agent_obs is not None
    np.testing.assert_allclose(transition.agent_obs["z_rl"], np.zeros((16,), dtype=np.float32))
    np.testing.assert_allclose(transition.next_agent_obs["z_rl"], np.ones((16,), dtype=np.float32))


def test_compact_terminal_transition_has_no_next_agent_obs():
    compact = CompactTransition(
        agent_obs=make_agent_obs(0.0),
        next_agent_obs=None,
        action=np.zeros((8,), dtype=np.float32),
        reward=1.0,
        done=True,
        truncated=False,
        discount=0.0,
        executed_steps=2,
        env_steps=2,
    )

    transition = compact.to_transition()

    assert transition.done
    assert transition.next_agent_obs is None
    assert transition.discount == 0.0


def test_agentlace_runtime_constructors_do_not_import_agentlace(tmp_path: Path):
    agent = RLTAgent(
        z_rl_dim=16,
        action_dim=2,
        chunk_size=4,
        execute_horizon=2,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        num_critics=2,
        device="cpu",
    )
    learner = AgentlaceLearnerRuntime(
        algorithm=agent,
        max_update_steps=1,
        batch_size=1,
        run_dir=str(tmp_path / "learner"),
        trainer_port=15588,
        broadcast_port=15589,
    )
    encoder = RLTokenEncoder(input_dim=16, rl_token_dim=16, num_layers=1, num_heads=4, ff_dim=32)
    actor = AgentlaceActorRuntime(
        env=FakeEnvBackend(action_dim=7, proprio_dim=8),
        policy=FakePolicyBackend(action_dim=7, chunk_size=4, embedding_dim=16),
        algorithm=agent,
        feature_processor=RLTFeatureProcessor(device="cpu", chunk_size=4, action_dim=7, encoder=encoder),
        max_env_steps=1,
        execute_horizon=1,
        trainer_port=15588,
        broadcast_port=15589,
        run_dir=str(tmp_path / "actor"),
    )

    assert learner.max_update_steps == 1
    assert actor.max_env_steps == 1
