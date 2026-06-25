import inspect

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from vla_rl.algorithms.rlt import RLTAgent, RLTokenEncoder, encode_rlt_obs
from vla_rl.algorithms.rlt.features import load_frozen_rlt_encoder
from vla_rl.data import RolloutBatch, Transition
from vla_rl.runtime.async_eval import (
    AsyncEvalRuntime,
    append_async_eval_request,
    append_async_eval_stop,
    launch_async_eval_worker,
    load_new_async_eval_results,
)
from vla_rl.runtime.run_utils import apply_actor_summary_file, read_actor_summary, send_actor_summary
from examples.libero.rlt.config import validate_rlt_cfg
from examples.libero.rlt.metrics import (
    actor_chunk_metric,
    actor_episode_metric,
    rlt_learner_metric_aliases,
    rlt_rollout_metric_aliases,
)
from examples.libero.rlt.rollout import insert_window_replay_transitions, transition_discount


def make_agent() -> RLTAgent:
    return RLTAgent(
        z_rl_dim=16,
        proprio_dim=3,
        action_dim=2,
        chunk_size=2,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        num_critics=2,
        policy_update_freq=1,
        utd_ratio=1,
        device="cpu",
    )


def make_rlt_state(value: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "z_rl": np.full((16,), value, dtype=np.float32),
        "reference_action": np.zeros((4,), dtype=np.float32),
        "proprio": np.zeros((3,), dtype=np.float32),
        "action_mask": np.ones((4,), dtype=np.float32),
    }


def make_transition(done: bool = False, discount: float = 0.25) -> Transition:
    return Transition(
        obs=make_rlt_state(0.0),
        next_obs=None if done else make_rlt_state(1.0),
        action=np.zeros((4,), dtype=np.float32),
        reward=1.0,
        done=done,
        truncated=False,
        discount=discount,
        executed_steps=2,
        env_steps=2,
    )


def test_rl_token_encoder_shape():
    encoder = RLTokenEncoder(input_dim=8, rl_token_dim=8, num_layers=1, num_heads=2, ff_dim=16)
    z_rl = encoder(torch.zeros(2, 5, 8))

    assert z_rl.shape == (2, 8)


def test_rlt_encoder_checkpoint_loading_and_max_tokens(tmp_path: Path):
    encoder = RLTokenEncoder(input_dim=8, rl_token_dim=8, num_layers=1, num_heads=2, ff_dim=16)
    path = tmp_path / "encoder.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "config": {
                "rlt": {
                    "input_dim": 8,
                    "rl_token_dim": 8,
                    "num_encoder_layers": 1,
                    "num_heads": 2,
                    "ff_dim": 16,
                }
            },
        },
        path,
    )

    loaded = load_frozen_rlt_encoder(str(path), device="cpu", max_tokens=3)

    assert getattr(loaded, "max_tokens") == 3
    assert loaded(torch.zeros(1, 3, 8)).shape == (1, 8)


def test_encode_rlt_obs_outputs_rlt_state():
    encoder = RLTokenEncoder(input_dim=8, rl_token_dim=8, num_layers=1, num_heads=2, ff_dim=16)
    encoder.max_tokens = 3

    rlt_state = encode_rlt_obs(
        np.zeros((1, 6, 8), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.ones((3,), dtype=np.float32),
        rl_token_encoder=encoder,
    )

    assert rlt_state["z_rl"].shape == (8,)
    assert rlt_state["reference_action"].shape == (4,)
    assert rlt_state["proprio"].shape == (3,)
    np.testing.assert_allclose(rlt_state["action_mask"], np.ones((4,), dtype=np.float32))


def test_rlt_config_uses_chunk_size_without_execution_horizon():
    repo_root = Path(__file__).resolve().parents[1]

    cfg = OmegaConf.load(
        repo_root / "examples/libero/rlt/configs/reward_model/libero_spatial_task4_self_cond_512_stage2.yaml"
    )
    assert "rlt" in cfg
    assert int(cfg.rlt.chunk_size) > 0
    assert "execute_horizon" not in cfg.runtime
    assert "execute_horizon" not in cfg.algorithm
    agent_cfg = dict(OmegaConf.to_container(cfg.algorithm, resolve=True))
    agent_cfg.pop("_target_", None)
    agent_cfg["device"] = "cpu"
    agent = RLTAgent(**agent_cfg)
    assert agent.chunk_size == cfg.rlt.chunk_size


def test_rlt_config_rejects_invalid_replan_steps():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(
        repo_root / "examples/libero/rlt/configs/reward_model/libero_spatial_task4_self_cond_512_stage2.yaml"
    )

    cfg.rlt.replan_steps = 0
    with pytest.raises(ValueError, match="rlt.replan_steps"):
        validate_rlt_cfg(cfg)

    cfg.rlt.replan_steps = int(cfg.rlt.chunk_size) + 1
    with pytest.raises(ValueError, match="rlt.replan_steps"):
        validate_rlt_cfg(cfg)


def test_rlt_transition_discount_uses_executed_steps():
    assert np.isclose(
        transition_discount(gamma=0.9, executed_steps=5, critic_terminal=False),
        0.9**5,
    )
    assert transition_discount(gamma=0.9, executed_steps=5, critic_terminal=True) == 0.0


def test_rlt_agent_act_and_update():
    agent = make_agent()
    actions = agent.sample_action(make_rlt_state(), deterministic=True)

    assert actions.shape == (2, 2)

    metrics = agent.update(RolloutBatch(transitions=[make_transition() for _ in range(4)]))

    assert set(metrics) >= {"loss_critic", "target_q_mean", "predicted_q_mean", "loss_actor", "bc_loss", "updates"}


def test_rlt_agent_state_config_matches_constructor():
    agent = make_agent()
    config = agent.state_dict()["config"]
    constructor_params = set(inspect.signature(RLTAgent).parameters)

    assert not (set(config) - constructor_params)
    restored = RLTAgent(**config, device="cpu")
    assert restored.target_noise_std == agent.target_noise_std
    assert restored.target_noise_clip == agent.target_noise_clip


def test_rlt_agent_uses_transition_discount_and_terminal_no_bootstrap():
    agent = make_agent()
    batch = RolloutBatch(transitions=[make_transition(done=True, discount=1.0) for _ in range(4)])
    fb = agent._convert_batch(batch)

    np.testing.assert_allclose(fb["discount"].cpu().numpy(), np.ones((4, 1), dtype=np.float32))
    _, target_q_mean, _ = agent._critic_step(fb)
    assert abs(target_q_mean - 1.0) < 1e-6


class FailingSummaryClient:
    def request(self, request_type, payload):
        raise RuntimeError(f"failed {request_type}")


def test_rlt_actor_summary_failure_is_persisted_and_readable(tmp_path: Path):
    metrics = []
    summary = send_actor_summary(
        FailingSummaryClient(),
        "send-stats",
        {"role": "actor", "env_steps": 12, "episodes": 1},
        tmp_path,
        metrics.append,
    )

    assert summary["actor_summary_notified"] is False
    assert metrics[-1]["event"] == "actor_summary_send_failed"
    persisted = read_actor_summary(tmp_path)
    assert persisted is not None
    assert persisted["env_steps"] == 12
    assert persisted["actor_summary_notified"] is False
    done, env_steps = apply_actor_summary_file(tmp_path, False, 0)
    assert done is True
    assert env_steps == 12


def test_async_eval_queue_and_result_roundtrip(tmp_path: Path):
    runtime = AsyncEvalRuntime(
        enabled=True,
        queue_path=tmp_path / "eval_queue.jsonl",
        summary_jsonl_path=tmp_path / "eval_summary.jsonl",
    )

    append_async_eval_request(runtime, {"eval_index": 0, "checkpoint_path": "ckpt.pt"})
    append_async_eval_stop(runtime)

    queue_lines = runtime.queue_path.read_text().splitlines()
    assert len(queue_lines) == 2
    assert '"type": "eval"' in queue_lines[0]
    assert '"type": "stop"' in queue_lines[1]

    runtime.summary_jsonl_path.write_text('{"eval/train_episode": 50, "eval/success_rate": 1.0, "eval_index": 0}\n')
    results = load_new_async_eval_results(runtime)

    assert results == [{"eval/train_episode": 50, "eval/success_rate": 1.0, "eval_index": 0}]
    assert load_new_async_eval_results(runtime) == []


def test_launch_async_eval_worker_accepts_worker_env(monkeypatch, tmp_path: Path):
    captured = {}

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, env=None):
            captured["cmd"] = cmd
            captured["stdout"] = stdout
            captured["stderr"] = stderr
            captured["env"] = env

    monkeypatch.setattr("vla_rl.runtime.async_eval.subprocess.Popen", FakePopen)

    proc, log_fp = launch_async_eval_worker(
        cmd=["python", "worker.py"],
        worker_log_path=tmp_path / "eval_worker.log",
        env={"CUDA_VISIBLE_DEVICES": "7", "MUJOCO_EGL_DEVICE_ID": "7"},
    )
    try:
        assert isinstance(proc, FakePopen)
        assert captured["cmd"] == ["python", "worker.py"]
        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "7"
        assert captured["env"]["MUJOCO_EGL_DEVICE_ID"] == "7"
    finally:
        log_fp.close()


def test_start_async_eval_worker_allows_local_env_without_env_url(monkeypatch, tmp_path: Path):
    from examples.libero.rlt.async_eval import start_async_eval_worker

    captured = {}

    class FakeProc:
        pass

    def fake_launch_async_eval_worker(*, cmd, worker_log_path, env=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return FakeProc(), worker_log_path.open("a", encoding="utf-8")

    monkeypatch.setattr("examples.libero.rlt.async_eval.launch_async_eval_worker", fake_launch_async_eval_worker)

    runtime = OmegaConf.create(
        {
            "run_dir": str(tmp_path),
            "async_eval": {
                "enabled": True,
                "policy_url": "http://127.0.0.1:8999",
                "worker_cuda_visible_devices": "2",
                "worker_mujoco_egl_device_id": "2",
            },
        }
    )
    async_eval = start_async_eval_worker(runtime, run_dir=tmp_path)
    try:
        assert async_eval.enabled is True
        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2"
        assert captured["env"]["MUJOCO_EGL_DEVICE_ID"] == "2"
    finally:
        if async_eval.worker_log_fp is not None:
            async_eval.worker_log_fp.close()


def test_rlt_actor_summary_read_ignores_partial_json(tmp_path: Path):
    (tmp_path / "actor_summary.json").write_text("{")
    assert read_actor_summary(tmp_path) is None


def test_rlt_agent_reads_action_mask_for_bc_loss():
    agent = make_agent()
    transition = make_transition()
    transition.obs["action_mask"] = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    fb = agent._convert_batch(RolloutBatch(transitions=[transition, transition]))

    np.testing.assert_allclose(
        fb["action_mask"].cpu().numpy(),
        np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    )
    metrics = agent.update(RolloutBatch(transitions=[transition, transition]))
    assert "bc_loss" in metrics


class ListDataStore:
    def __init__(self):
        self.payloads = []

    def insert(self, payload):
        self.payloads.append(payload)


def test_rlt_window_replay_uses_cross_chunk_actions():
    store = ListDataStore()
    current_actions = np.arange(20, dtype=np.float32).reshape(10, 2)
    next_actions = np.arange(20, 40, dtype=np.float32).reshape(10, 2)
    pending = {
        "rlt_obs": make_rlt_state(0.0),
        "window_start_rlt_obs": [
            make_rlt_state(2.0),
            make_rlt_state(4.0),
            make_rlt_state(6.0),
            make_rlt_state(8.0),
        ],
        "actions": current_actions,
        "next_actions": next_actions,
        "reward": 1.0,
        "done": False,
        "truncated": False,
        "terminal": False,
        "executed_steps": 10,
        "env_steps": 10,
        "info": {},
        "chunk_start_env_steps": 0,
    }

    inserted = insert_window_replay_transitions(
        pending,
        next_rlt_obs=make_rlt_state(10.0),
        next_window_start_rlt_obs=[
            make_rlt_state(12.0),
            make_rlt_state(14.0),
            make_rlt_state(16.0),
            make_rlt_state(18.0),
        ],
        data_store=store,
        subsample_stride=2,
        chunk_size=10,
        gamma=0.99,
    )

    assert inserted == 5
    transitions = [Transition.from_payload(payload) for payload in store.payloads]
    assert [t.info["subsample_position"] for t in transitions] == [0, 2, 4, 6, 8]
    np.testing.assert_allclose(transitions[0].action.reshape(10, 2), current_actions)
    for idx, position in enumerate((2, 4, 6, 8), start=1):
        np.testing.assert_allclose(
            transitions[idx].action.reshape(10, 2),
            np.concatenate([current_actions[position:], next_actions[:position]], axis=0),
        )


def test_rlt_rollout_metric_aliases_match_wandb_names():
    metrics = rlt_rollout_metric_aliases(
        episode_id=3,
        episode_return=2.5,
        episode_steps=40,
        success=True,
        recent_success_rate_50=0.25,
    )

    assert metrics == {
        "rollout/episode_id": 3,
        "rollout/episode_return": 2.5,
        "rollout/episode_steps": 40,
        "rollout/success": 1,
        "rollout/recent_success_rate_50": 0.25,
    }


def test_rlt_actor_episode_metric_contains_env_and_rollout_aliases():
    metric = actor_episode_metric(
        episode_id=3,
        episode_return=2.5,
        episode_steps=40,
        success=True,
        env_steps=100,
        recent_success_rate_50=0.25,
    )

    assert metric["environment"]["episode"] == {
        "return": 2.5,
        "length": 40,
        "success": True,
        "env_steps": 100,
    }
    assert metric["rollout/episode_id"] == 3
    assert metric["rollout/recent_success_rate_50"] == 0.25


def test_rlt_actor_chunk_metric_separates_committed_and_submitted_transitions():
    metric = actor_chunk_metric(
        env_steps=10,
        episode=1,
        reward=0.0,
        done=False,
        truncated=False,
        executed_steps=5,
        chunk_size=5,
        replay_transitions=0,
        submitted_transitions=1,
        wall_time_sec=2.0,
        chunk_time_sec=0.5,
        reset_time_sec=0.0,
        reward_stats={"committed": 0, "pending": 1},
    )

    assert metric["replay_transitions"] == 0
    assert metric["submitted_transitions"] == 1
    assert metric["reward/committed"] == 0
    assert metric["reward/pending"] == 1


def test_rlt_learner_metric_aliases_keep_available_losses_only():
    metrics = rlt_learner_metric_aliases(
        {
            "loss_critic": 1.0,
            "target_q_mean": 0.5,
            "predicted_q_mean": 0.25,
            "bc_loss": 0.1,
            "updates": 5,
        },
        update_steps=7,
        env_steps=70,
        replay_size=32,
    )

    assert metrics == {
        "learner/update_steps": 7,
        "learner/env_steps": 70,
        "learner/replay_size": 32,
        "learner/loss_critic": 1.0,
        "learner/target_q_mean": 0.5,
        "learner/predicted_q_mean": 0.25,
        "learner/bc_loss": 0.1,
    }
