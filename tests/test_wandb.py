from __future__ import annotations

import sys
import types

import numpy as np

from vla_rl.runtime.wandb import (
    make_wandb_logger,
    flatten_wandb_scalars,
    select_hil_serl_wandb_scalars,
)


def test_flatten_wandb_scalars_keeps_small_metrics_and_skips_arrays() -> None:
    payload = flatten_wandb_scalars(
        {
            "role": "learner",
            "train": {"critic_loss": np.float32(1.25)},
            "speed": {"updates_per_sec": 4.0},
            "raw_action": np.zeros((2, 7), dtype=np.float32),
        }
    )

    assert payload["role"] == "learner"
    assert payload["train/critic_loss"] == np.float32(1.25).item()
    assert payload["speed/updates_per_sec"] == 4.0
    assert "raw_action" not in payload


def test_hil_serl_wandb_filter_keeps_readable_rlt_aliases() -> None:
    payload = select_hil_serl_wandb_scalars(
        {
            "role": "learner",
            "env_steps": 100,
            "replay_size": 50,
            "speed": {"updates_per_sec": 4.0},
            "train/loss_critic": 1.0,
            "rollout": {"episode_id": 3, "success": 1, "recent_success_rate_50": 0.35},
            "learner": {"loss_actor": 2.0, "bc_loss": 0.4},
            "eval": {"success_rate": 0.5, "episodes_run": 10},
            "time/algorithm_update_sec": 0.02,
            "time/publish_network_sec": 0.03,
            "time/save_checkpoint_sec": 0.04,
            "timer": {"sample_actions": 0.1},
            "environment": {"episode": {"return": 1.0, "success": True}},
            "summary": {"update_steps": 10},
        }
    )

    assert payload == {
        "rollout/episode_id": 3,
        "rollout/success": 1,
        "rollout/recent_success_rate_50": 0.35,
        "learner/loss_actor": 2.0,
        "learner/bc_loss": 0.4,
        "eval/success_rate": 0.5,
        "eval/episodes_run": 10,
    }


def test_disabled_wandb_logger_is_noop() -> None:
    wandb_logger = make_wandb_logger({"mode": "disabled"}, variant={}, run_dir=None)

    wandb_logger.log({"train": {"loss": 1.0}}, step=1)
    wandb_logger.finish()


def test_wandb_logger_falls_back_to_native_wandb_when_swanlab_is_missing(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {"logs": []}

    def fake_init(**kwargs):
        calls["init"] = kwargs
        return object()

    def fake_log(data, step=None):
        calls["logs"].append((data, step))

    def fake_finish():
        calls["finished"] = True

    def fake_define_metric(*args, **kwargs):
        calls.setdefault("defined", []).append((args, kwargs))

    fake_wandb = types.SimpleNamespace(
        init=fake_init,
        log=fake_log,
        finish=fake_finish,
        define_metric=fake_define_metric,
    )
    monkeypatch.setitem(sys.modules, "swanlab", None)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    wandb_logger = make_wandb_logger(
        {
            "project": "vla-rl-test",
            "exp_name": "unit-test",
            "tags": ["test"],
            "mode": "online",
            "save_code": False,
        },
        variant={"runtime": {"max_env_steps": 10}},
        run_dir=tmp_path,
    )
    wandb_logger.log({"learner": {"loss_actor": 2.0}, "big": np.zeros((3, 3))}, step=7)
    wandb_logger.finish()

    assert calls["init"]["project"] == "vla-rl-test"
    assert calls["init"]["name"] == "unit-test"
    assert calls["init"]["mode"] == "online"
    assert "id" not in calls["init"]
    assert calls["defined"] == [
        (("rollout/episode_id",), {}),
        (("rollout/*",), {"step_metric": "rollout/episode_id"}),
        (("learner/update_steps",), {}),
        (("learner/*",), {"step_metric": "learner/update_steps"}),
        (("eval/episodes_run",), {}),
        (("eval/*",), {"step_metric": "eval/episodes_run"}),
    ]
    assert calls["logs"] == [({"learner/loss_actor": 2.0}, 7)]
    assert calls["finished"] is True


def test_wandb_logger_prefers_swanlab(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {"logs": []}

    def fake_init(**kwargs):
        calls["init"] = kwargs
        return object()

    def fake_log(data, step=None):
        calls["logs"].append((data, step))

    def fake_finish():
        calls["finished"] = True

    fake_swanlab = types.SimpleNamespace(init=fake_init, log=fake_log, finish=fake_finish)
    monkeypatch.setitem(sys.modules, "swanlab", fake_swanlab)

    wandb_logger = make_wandb_logger(
        {
            "project": "vla-rl-test",
            "exp_name": "swanlab-unit-test",
            "mode": "online",
        },
        variant={},
        run_dir=tmp_path,
    )
    wandb_logger.log({"timer": {"step_env": 0.2}, "rollout": {"success": 1}}, step=5)
    wandb_logger.finish()

    assert calls["init"]["project"] == "vla-rl-test"
    assert calls["init"]["experiment_name"] == "swanlab-unit-test"
    assert calls["init"]["mode"] == "cloud"
    assert "id" not in calls["init"]
    assert calls["logs"] == [({"rollout/success": 1}, 5)]
    assert calls["finished"] is True


def test_wandb_logger_uses_explicit_run_id_only(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {"logs": []}

    def fake_init(**kwargs):
        calls["init"] = kwargs
        return object()

    fake_swanlab = types.SimpleNamespace(
        init=fake_init,
        log=lambda data, step=None: calls["logs"].append((data, step)),
        finish=lambda: calls.setdefault("finished", True),
    )
    monkeypatch.setitem(sys.modules, "swanlab", fake_swanlab)

    wandb_logger = make_wandb_logger(
        {
            "project": "vla-rl-test",
            "exp_name": "same-exp-name",
            "id": "explicit-run-id",
            "mode": "online",
        },
        variant={},
        run_dir=tmp_path,
    )
    wandb_logger.finish()

    assert calls["init"]["experiment_name"] == "same-exp-name"
    assert calls["init"]["id"] == "explicit-run-id"


def test_wandb_logger_does_not_hide_swanlab_init_errors(monkeypatch, tmp_path) -> None:
    def bad_init(**kwargs):
        del kwargs
        raise RuntimeError("bad swanlab config")

    fake_swanlab = types.SimpleNamespace(init=bad_init, log=lambda *args, **kwargs: None, finish=lambda: None)
    monkeypatch.setitem(sys.modules, "swanlab", fake_swanlab)
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        types.SimpleNamespace(
            init=lambda **kwargs: object(),
            log=lambda *args, **kwargs: None,
            finish=lambda: None,
        ),
    )

    try:
        make_wandb_logger(
            {"project": "vla-rl-test"},
            variant={},
            run_dir=tmp_path,
        )
    except RuntimeError as exc:
        assert "bad swanlab config" in str(exc)
    else:
        raise AssertionError("SwanLab init errors should not silently fall back to native W&B")
