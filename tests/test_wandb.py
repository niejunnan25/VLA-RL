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


def test_hil_serl_wandb_filter_keeps_only_update_timer_and_environment() -> None:
    payload = select_hil_serl_wandb_scalars(
        {
            "role": "learner",
            "env_steps": 100,
            "replay_size": 50,
            "speed": {"updates_per_sec": 4.0},
            "train/loss_critic": 1.0,
            "time/algorithm_update_sec": 0.02,
            "time/publish_network_sec": 0.03,
            "time/save_checkpoint_sec": 0.04,
            "timer": {"sample_actions": 0.1},
            "environment": {"episode": {"return": 1.0, "success": True}},
            "summary": {"update_steps": 10},
        }
    )

    assert payload == {
        "train/loss_critic": 1.0,
        "timer/algorithm_update_sec": 0.02,
        "timer/sample_actions": 0.1,
        "environment/episode/return": 1.0,
        "environment/episode/success": True,
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

    fake_wandb = types.SimpleNamespace(init=fake_init, log=fake_log, finish=fake_finish)
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
    wandb_logger.log({"train": {"actor_loss": 2.0}, "big": np.zeros((3, 3))}, step=7)
    wandb_logger.finish()

    assert calls["init"]["project"] == "vla-rl-test"
    assert calls["init"]["name"] == "unit-test"
    assert calls["init"]["mode"] == "online"
    assert "id" not in calls["init"]
    assert calls["logs"] == [({"train/actor_loss": 2.0}, 7)]
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
    wandb_logger.log({"timer": {"step_env": 0.2}, "speed": {"env_per_sec": 3.0}}, step=5)
    wandb_logger.finish()

    assert calls["init"]["project"] == "vla-rl-test"
    assert calls["init"]["experiment_name"] == "swanlab-unit-test"
    assert calls["init"]["mode"] == "cloud"
    assert "id" not in calls["init"]
    assert calls["logs"] == [({"timer/step_env": 0.2}, 5)]
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
