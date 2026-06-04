#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vla_rl.algorithms.rlt import RLTokenEncoder
from vla_rl.data import ReplayBuffer, Transition
from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer

from vla_rl.runtime.agentlace import json_sanitize, make_agentlace_replay_store
from vla_rl.runtime.checkpoint import CheckpointManager
from vla_rl.runtime.wandb import make_wandb_logger
from vla_rl.runtime.timer import Timer
from vla_rl.runtime.run_utils import (
    apply_actor_summary_file,
    make_jsonl_metric_writer,
    next_interval,
    run_dir_from_runtime,
    runtime_float,
    save_checkpoint,
    send_actor_summary,
)
from examples.libero.rlt.config import (
    build_reference_policy,
    create_env,
    create_rlt_agent,
    load_config,
    load_rl_token_encoder,
    rlt_cfg,
    validate_rlt_cfg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIBERO RLT actor or learner.")
    parser.add_argument("--config", required=True, help="Path to an RLT YAML config.")
    parser.add_argument("--role", required=True, choices=("actor", "learner"))
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="OmegaConf dotlist overrides after --.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    cfg.runtime.role = args.role
    validate_rlt_cfg(cfg)

    if args.role == "learner":
        summary = run_learner(cfg)
    else:
        summary = run_actor(cfg)
    print(json.dumps(json_sanitize(summary), sort_keys=True))


def run_learner(cfg: DictConfig) -> dict[str, Any]:
    """SERL-style RLT learner loop.

    The learner owns the trainable actor/critic, replay, updates,
    checkpointing, and metrics. Agentlace is used only as transport for actor
    transitions and actor-weight broadcasts.
    """

    runtime = cfg.runtime
    agent = create_rlt_agent(cfg)
    run_dir = run_dir_from_runtime(runtime)
    checkpoints = CheckpointManager(run_dir) if run_dir is not None else None
    replay = ReplayBuffer(capacity=int(runtime.replay_capacity), seed=int(runtime.replay_seed))
    config_snapshot = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = make_wandb_logger(cfg.get("wandb", None), variant=config_snapshot, run_dir=run_dir)
    wandb_finished = False

    def finish_wandb_logger() -> None:
        nonlocal wandb_finished
        if not wandb_finished:
            wandb_logger.finish()
            wandb_finished = True

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(config_snapshot), run_dir / "config.yaml")
        if not runtime.get("resume_from", None):
            (run_dir / "metrics.jsonl").write_text("")
        else:
            (run_dir / "metrics.jsonl").touch(exist_ok=True)

    update_steps = 0
    env_steps = 0
    episodes = 0
    total_reward = 0.0
    if runtime.get("resume_from", None):
        payload = checkpoints.load(runtime.resume_from, agent) if checkpoints is not None else {}
        update_steps = int(payload.get("update_steps", 0))
        env_steps = int(payload.get("env_steps", 0))
        episodes = int(payload.get("episodes", 0))
        total_reward = float(payload.get("total_reward", 0.0))

    actor_done = False
    actor_done_env_steps = 0

    def current_env_steps(value: int) -> int:
        return max(int(value), int(replay.latest_env_steps), int(actor_done_env_steps))

    def write_metric(metric: dict[str, Any]) -> None:
        if run_dir is None:
            wandb_logger.log(metric, step=update_steps)
            return
        with (run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(json_sanitize(metric), sort_keys=True) + "\n")
        wandb_logger.log(metric, step=update_steps)

    def apply_actor_summary_file() -> None:
        nonlocal actor_done, actor_done_env_steps
        actor_done, actor_done_env_steps = apply_actor_summary_file(run_dir, actor_done, actor_done_env_steps)

    def request_callback(request_type: str, payload: Any) -> Any:
        nonlocal actor_done, actor_done_env_steps
        if request_type != str(runtime.request_type):
            return {"ok": False, "error": f"unsupported request type: {request_type}"}
        stat = dict(payload or {})
        stat.setdefault("role", "actor")
        if stat.get("event") == "actor_summary":
            actor_done = True
            actor_done_env_steps = max(actor_done_env_steps, int(stat.get("env_steps", 0)))
        write_metric(stat)
        return {"ok": True}

    trainer_config = TrainerConfig(
        port_number=int(runtime.trainer_port),
        broadcast_port=int(runtime.broadcast_port),
        request_types=[str(runtime.request_type)],
    )
    server = TrainerServer(trainer_config, request_callback=request_callback)
    server.register_data_store(str(runtime.store_name), make_agentlace_replay_store(replay))
    server.start(threaded=True)
    server.publish_network(agent.policy_state_dict())

    next_publish_at = next_interval(update_steps, int(runtime.publish_interval_updates))
    next_ckpt_env_at = next_interval(env_steps, int(runtime.checkpoint_interval_env_steps))
    next_ckpt_update_at = next_interval(update_steps, int(runtime.checkpoint_interval_updates))
    start_time = time.perf_counter()
    active_update_time_sec = 0.0
    last_wait_metric_time = 0.0
    last_wait_publish_time = time.perf_counter()
    last_update: dict[str, Any] = {}

    completed_loop = False
    try:
        while True:
            apply_actor_summary_file()
            env_steps = current_env_steps(env_steps)
            if _learner_should_stop(
                update_steps=update_steps,
                env_steps=env_steps,
                max_update_steps=int(runtime.max_update_steps),
                max_env_steps=int(runtime.max_env_steps),
                actor_done=actor_done,
            ):
                break

            if update_steps >= int(runtime.max_update_steps):
                now = time.perf_counter()
                if now - last_wait_publish_time >= 1.0:
                    server.publish_network(agent.policy_state_dict())
                    last_wait_publish_time = now
                if now - last_wait_metric_time >= 1.0:
                    write_metric(
                        {
                            "role": "learner",
                            "event": "waiting_for_actor_env",
                            "replay_size": len(replay),
                            "target_env_steps": int(runtime.max_env_steps),
                            "update_steps": update_steps,
                            "env_steps": env_steps,
                            "actor_done": actor_done,
                            "wall_time_sec": now - start_time,
                        }
                    )
                    last_wait_metric_time = now
                if checkpoints is not None and int(runtime.checkpoint_interval_env_steps) > 0 and env_steps >= next_ckpt_env_at:
                    save_checkpoint(checkpoints, agent, env_steps, update_steps, episodes, total_reward, config_snapshot)
                    next_ckpt_env_at += int(runtime.checkpoint_interval_env_steps)
                time.sleep(runtime_float(runtime, "update_sleep_sec", 0.05))
                continue

            min_replay_size = max(int(runtime.training_starts), int(runtime.batch_size))
            if len(replay) < min_replay_size:
                now = time.perf_counter()
                if actor_done:
                    write_metric(
                        {
                            "role": "learner",
                            "event": "stopping_replay_below_training_starts",
                            "replay_size": len(replay),
                            "required_replay_size": min_replay_size,
                            "training_starts": int(runtime.training_starts),
                            "update_steps": update_steps,
                            "env_steps": max(env_steps, replay.latest_env_steps),
                            "wall_time_sec": now - start_time,
                        }
                    )
                    break
                if now - last_wait_publish_time >= 1.0:
                    server.publish_network(agent.policy_state_dict())
                    last_wait_publish_time = now
                if now - last_wait_metric_time >= 1.0:
                    write_metric(
                        {
                            "role": "learner",
                            "event": "waiting_for_replay",
                            "replay_size": len(replay),
                            "training_starts": int(runtime.training_starts),
                            "update_steps": update_steps,
                            "env_steps": max(env_steps, replay.latest_env_steps),
                            "wall_time_sec": now - start_time,
                        }
                    )
                    last_wait_metric_time = now
                time.sleep(runtime_float(runtime, "update_sleep_sec", 0.05))
                continue

            step_timer = Timer()
            with step_timer.context("sample_replay"):
                batch = replay.sample(int(runtime.batch_size))
            with step_timer.context("algorithm_update"):
                last_update = agent.update(batch)
            timing = step_timer.get_average_times(reset=False, prefix="time/", suffix="_sec")
            active_update_time_sec += sum(step_timer.get_total_times(reset=False).values())
            update_steps += 1
            env_steps = current_env_steps(env_steps)

            publish_time_sec = 0.0
            checkpoint_time_sec = 0.0
            if int(runtime.publish_interval_updates) > 0 and update_steps >= next_publish_at:
                publish_start = time.perf_counter()
                server.publish_network(agent.policy_state_dict())
                publish_time_sec += time.perf_counter() - publish_start
                next_publish_at += int(runtime.publish_interval_updates)
            if checkpoints is not None and int(runtime.checkpoint_interval_env_steps) > 0 and env_steps >= next_ckpt_env_at:
                ckpt_start = time.perf_counter()
                save_checkpoint(checkpoints, agent, env_steps, update_steps, episodes, total_reward, config_snapshot)
                checkpoint_time_sec += time.perf_counter() - ckpt_start
                next_ckpt_env_at += int(runtime.checkpoint_interval_env_steps)
            if checkpoints is not None and int(runtime.checkpoint_interval_updates) > 0 and update_steps >= next_ckpt_update_at:
                ckpt_start = time.perf_counter()
                save_checkpoint(
                    checkpoints,
                    agent,
                    env_steps,
                    update_steps,
                    episodes,
                    total_reward,
                    config_snapshot,
                    tag=f"update_{update_steps}.pt",
                )
                checkpoint_time_sec += time.perf_counter() - ckpt_start
                next_ckpt_update_at += int(runtime.checkpoint_interval_updates)

            wall_time_sec = time.perf_counter() - start_time
            write_metric(
                {
                    "role": "learner",
                    "env_steps": env_steps,
                    "update_steps": update_steps,
                    "replay_size": len(replay),
                    "update_time_sec": timing.get("time/sample_replay_sec", 0.0)
                    + timing.get("time/algorithm_update_sec", 0.0),
                    "time/publish_network_sec": publish_time_sec,
                    "time/save_checkpoint_sec": checkpoint_time_sec,
                    "speed/learner_wall_updates_per_sec": update_steps / max(wall_time_sec, 1e-9),
                    "speed/learner_active_updates_per_sec": update_steps / max(active_update_time_sec, 1e-9),
                    "wall_time_sec": wall_time_sec,
                    **timing,
                    **{f"train/{key}": value for key, value in last_update.items()},
                }
            )
        completed_loop = True
    finally:
        stop = getattr(server, "stop", None)
        if callable(stop):
            stop()
        if not completed_loop:
            finish_wandb_logger()

    env_steps = current_env_steps(env_steps)
    summary = {
        "role": "learner",
        "env_steps": env_steps,
        "update_steps": update_steps,
        "replay_size": len(replay),
        "last_algorithm_updates": last_update.get("updates", 0),
    }
    try:
        if checkpoints is not None:
            if int(runtime.checkpoint_interval_env_steps) > 0 and env_steps >= next_ckpt_env_at:
                save_checkpoint(checkpoints, agent, env_steps, update_steps, episodes, total_reward, config_snapshot)
            save_checkpoint(checkpoints, agent, env_steps, update_steps, episodes, total_reward, config_snapshot, tag="final.pt")
        if run_dir is not None:
            (run_dir / "summary.json").write_text(json.dumps(json_sanitize(summary), indent=2, sort_keys=True) + "\n")
        write_metric({"summary": summary})
    finally:
        finish_wandb_logger()
    return summary


def run_actor(cfg: DictConfig) -> dict[str, Any]:
    """SERL-style RLT actor loop.

    The actor keeps the RLT data path explicit:
    observation -> frozen VLA output -> base action + RLT observation ->
    RLT actor/reference action -> env chunk step -> learner transition.
    """

    runtime = cfg.runtime
    env = create_env(cfg)
    reference_policy = build_reference_policy(cfg)
    rl_token_encoder = load_rl_token_encoder(cfg)
    agent = create_rlt_agent(cfg)
    run_dir = run_dir_from_runtime(runtime)
    write_actor_metric = make_jsonl_metric_writer(run_dir, "actor_metrics.jsonl")

    data_store = QueuedDataStore(int(runtime.actor_queue_capacity))
    trainer_config = TrainerConfig(
        port_number=int(runtime.trainer_port),
        broadcast_port=int(runtime.broadcast_port),
        request_types=[str(runtime.request_type)],
    )
    client = TrainerClient(
        str(runtime.actor_name),
        str(runtime.trainer_ip),
        trainer_config,
        data_store,
        wait_for_server=True,
    )
    has_policy_state = False

    def update_actor(payload: dict[str, Any]) -> None:
        nonlocal has_policy_state
        if payload is not None:
            agent.load_policy_state_dict(payload)
            has_policy_state = True

    client.recv_network_callback(update_actor)
    _wait_for_initial_weights(
        client,
        has_policy_state_fn=lambda: has_policy_state,
        timeout_sec=runtime_float(runtime, "initial_weight_timeout_sec", 300.0),
        sleep_sec=runtime_float(runtime, "client_update_sleep_sec", 0.1),
    )

    try:
        obs = env.reset()
        env_steps = 0
        episodes = 0
        total_reward = 0.0
        episode_return = 0.0
        active_rollout_time_sec = 0.0
        next_weight_update_at = next_interval(env_steps, int(runtime.weight_update_interval_steps))
        next_stats_at = next_interval(env_steps, int(runtime.stats_interval_env_steps))
        start_time = time.perf_counter()
        timer = Timer()
        rlt = rlt_cfg(cfg)
        chunk_size = int(rlt.chunk_size)
        episode_step = 0

        while env_steps < int(runtime.max_env_steps):
            timer.tick("total")
            chunk_start = time.perf_counter()
            chunk_start_env_steps = env_steps

            with timer.context("reference_policy"):
                base_actions, prefix_tokens, proprio = reference_policy.predict_actions_and_prefix(obs)
                base_actions = np.asarray(base_actions, dtype=np.float32)[:chunk_size]

            with timer.context("encode_rlt_obs"):
                rlt_obs = encode_rlt_obs(
                    prefix_tokens,
                    base_actions,
                    proprio,
                    rl_token_encoder=rl_token_encoder,
                )

            with timer.context("sample_actions"):
                if env_steps < int(runtime.warmup_steps):
                    actions = base_actions
                else:
                    actions = agent.sample_action(rlt_obs)

            actions = np.asarray(actions, dtype=np.float32)

            with timer.context("step_env"):
                next_obs, reward, done, truncated, info = env.step_chunk(actions)

            info = dict(info)
            executed_steps = int(info.get("executed_steps", len(actions)))
            reward = float(reward)
            total_reward += reward
            episode_return += reward
            terminal = bool(done or truncated)

            next_rlt_state = None
            if not terminal:
                with timer.context("next_reference_policy"):
                    next_base_actions, next_prefix_tokens, next_proprio = reference_policy.predict_actions_and_prefix(next_obs)
                    next_base_actions = np.asarray(next_base_actions, dtype=np.float32)[:chunk_size]
                with timer.context("next_encode_rlt_obs"):
                    next_rlt_state = encode_rlt_obs(
                        next_prefix_tokens,
                        next_base_actions,
                        next_proprio,
                        rl_token_encoder=rl_token_encoder,
                    )

            action_mask = np.zeros_like(actions, dtype=np.float32)
            action_mask[:executed_steps] = 1.0
            rlt_obs["action_mask"] = action_mask.reshape(-1)
            transition = Transition(
                obs=rlt_obs,
                next_obs=next_rlt_state,
                action=actions.reshape(-1),
                reward=reward,
                done=bool(done),
                truncated=bool(truncated),
                discount=0.0 if terminal else float(runtime.gamma) ** executed_steps,
                executed_steps=executed_steps,
                env_steps=env_steps + executed_steps,
                info={**info, "chunk_start_env_steps": chunk_start_env_steps},
            )
            with timer.context("send_transition"):
                data_store.insert(transition.to_payload())
                client.update()

            env_steps += executed_steps
            episode_step += executed_steps

            metric_episode = episodes
            chunk_time_sec = time.perf_counter() - chunk_start
            active_rollout_time_sec += chunk_time_sec
            reset_time_sec = 0.0
            if terminal:
                client.request(
                    str(runtime.request_type),
                    {
                        "environment": {
                            "episode": {
                                "return": episode_return,
                                "length": episode_step,
                                "success": bool(info.get("success", False) or info.get("env_done", False) or info.get("is_success", False)),
                                "env_steps": env_steps,
                            }
                        }
                    },
                )
                episodes += 1
                episode_step = 0
                episode_return = 0.0
                reset_start = time.perf_counter()
                with timer.context("reset_env"):
                    obs = env.reset()
                reset_time_sec = time.perf_counter() - reset_start
            else:
                obs = next_obs

            wall_time_sec = time.perf_counter() - start_time
            timer.tock("total")
            metric = {
                "role": "actor",
                "env_steps": env_steps,
                "episode": metric_episode,
                "reward": float(reward),
                "done": bool(done),
                "truncated": bool(truncated),
                "executed_steps": executed_steps,
                "chunk_size": chunk_size,
                "wall_time_sec": wall_time_sec,
                "chunk_time_sec": chunk_time_sec,
                "time/reset_env_sec": reset_time_sec,
            }

            write_actor_metric(metric)
            if int(runtime.stats_interval_env_steps) > 0 and env_steps >= next_stats_at:
                actor_stats = {
                    "env_steps": env_steps,
                    "episodes": episodes,
                    "total_reward": total_reward,
                    "wall_time_sec": wall_time_sec,
                    "active_env_steps_per_sec": env_steps / max(active_rollout_time_sec, 1e-9),
                    "wall_env_steps_per_sec": env_steps / max(wall_time_sec, 1e-9),
                }
                client.request(str(runtime.request_type), {"timer": timer.get_average_times(), "actor": actor_stats})
                next_stats_at += int(runtime.stats_interval_env_steps)
            if int(runtime.weight_update_interval_steps) > 0 and env_steps >= next_weight_update_at:
                client.update()
                next_weight_update_at += int(runtime.weight_update_interval_steps)

        summary = {
            "role": "actor",
            "env_steps": env_steps,
            "episodes": episodes,
            "total_reward": total_reward,
            "received_policy_state": has_policy_state,
        }
        summary = send_actor_summary(client, str(runtime.request_type), summary, run_dir, write_actor_metric)
        write_actor_metric({"summary": summary})
        return summary
    finally:
        stop = getattr(client, "stop", None)
        if callable(stop):
            stop()
        close = getattr(env, "close", None)
        if callable(close):
            close()
        close = getattr(reference_policy, "close", None)
        if callable(close):
            close()


def encode_rlt_obs(
    prefix_tokens: np.ndarray,
    base_actions: np.ndarray,
    proprio: np.ndarray,
    *,
    rl_token_encoder: RLTokenEncoder,
) -> dict[str, np.ndarray]:
    z_vla = torch.as_tensor(
        prefix_tokens,
        dtype=torch.float32,
        device=next(rl_token_encoder.parameters()).device,
    )
    if z_vla.dim() == 2:
        z_vla = z_vla.unsqueeze(0)
    if z_vla.dim() != 3:
        raise ValueError(f"prefix embeddings must be [B, T, D] or [T, D], got {tuple(z_vla.shape)}")

    max_tokens = getattr(rl_token_encoder, "max_tokens", None)
    if max_tokens is not None:
        z_vla = z_vla[:, : int(max_tokens), :]

    z_rl = rl_token_encoder(z_vla).squeeze(0).detach().cpu().numpy().astype(np.float32)

    reference_action = np.asarray(base_actions, dtype=np.float32).reshape(-1)
    return {
        "z_rl": z_rl,
        "reference_action": reference_action,
        "proprio": np.asarray(proprio, dtype=np.float32).reshape(-1),
    }



def _wait_for_initial_weights(client: Any, *, has_policy_state_fn, timeout_sec: float, sleep_sec: float) -> None:
    deadline = time.perf_counter() + timeout_sec
    while not has_policy_state_fn():
        client.update()
        if time.perf_counter() > deadline:
            raise TimeoutError("timed out waiting for initial learner policy weights")
        time.sleep(sleep_sec)


def _learner_should_stop(
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



if __name__ == "__main__":
    main()
