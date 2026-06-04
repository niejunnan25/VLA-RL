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

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.libero.pld.config import (
    create_env,
    create_pld_agent,
    create_pld_obs_builder,
    create_reference_policy,
    load_config,
    predict_base_actions,
    validate_pld_cfg,
)
from vla_rl.algorithms.pld import build_pld_obs, load_pld_offline_replay
from vla_rl.data import MixedReplaySampler, ReplayBuffer, Transition
from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer

from vla_rl.runtime.agentlace import json_sanitize, make_agentlace_replay_store
from vla_rl.runtime.checkpoint import CheckpointManager
from vla_rl.runtime.wandb import make_wandb_logger
from vla_rl.runtime.timer import Timer
from vla_rl.runtime.run_utils import (
    apply_actor_summary_file as read_actor_summary_file,
    make_jsonl_metric_writer,
    run_dir_from_runtime,
    save_checkpoint,
    send_actor_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIBERO PLD actor or learner.")
    parser.add_argument("--config", required=True, help="Path to a PLD YAML config.")
    parser.add_argument("--role", required=True, choices=("actor", "learner"))
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="OmegaConf dotlist overrides after --.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    cfg.runtime.role = args.role
    validate_pld_cfg(cfg)
    summary = run_learner(cfg) if args.role == "learner" else run_actor(cfg)
    print(json.dumps(json_sanitize(summary), sort_keys=True))


def run_learner(cfg: DictConfig) -> dict[str, Any]:
    runtime = cfg.runtime
    agent = create_pld_agent(cfg)
    run_dir = run_dir_from_runtime(runtime)
    checkpoints = CheckpointManager(run_dir) if run_dir is not None else None
    replay = ReplayBuffer(capacity=int(runtime.replay_capacity), seed=int(runtime.replay_seed))
    config_snapshot = OmegaConf.to_container(cfg, resolve=True)
    offline_load_start = time.perf_counter()
    offline_replay, offline_stats = _load_offline_replay(runtime)
    offline_load_time_sec = time.perf_counter() - offline_load_start
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

    def refresh_actor_summary_file() -> None:
        nonlocal actor_done, actor_done_env_steps
        actor_done, actor_done_env_steps = read_actor_summary_file(run_dir, actor_done, actor_done_env_steps)

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
    sampler = MixedReplaySampler(
        replay,
        offline_replay,
        offline_ratio=0.0 if offline_replay is None else float(runtime.offline_ratio),
    )
    start_time = time.perf_counter()
    active_update_time_sec = 0.0
    last_wait_metric_time = 0.0
    last_wait_publish_time = time.perf_counter()
    last_update: dict[str, Any] = {}
    calql_steps_done = 0

    write_metric(
        {
            "role": "learner",
            "algorithm": "pld",
            "event": "offline_replay_loaded",
            "offline_replay_size": 0 if offline_replay is None else len(offline_replay),
            "offline_load_time_sec": offline_load_time_sec,
            "offline_stats": offline_stats,
        }
    )

    completed_loop = False
    try:
        while calql_steps_done < int(runtime.calql_pretrain_steps) and update_steps < int(runtime.max_update_steps):
            if offline_replay is None or len(offline_replay) == 0:
                break
            step_timer = Timer()
            with step_timer.context("offline_sample"):
                batch = offline_replay.sample(int(runtime.batch_size))
            with step_timer.context("calql_update"):
                last_update = agent.update_critics_calql(
                    batch,
                    calql_alpha=float(runtime.calql_alpha),
                    calql_n_actions=int(runtime.calql_n_actions),
                    calql_temperature=float(runtime.calql_temperature),
                )
            timing = step_timer.get_average_times(reset=False, prefix="time/", suffix="_sec")
            active_update_time_sec += sum(step_timer.get_total_times(reset=False).values())
            update_steps += 1
            calql_steps_done += 1
            wall_time_sec = time.perf_counter() - start_time
            write_metric(
                {
                    "role": "learner",
                    "phase": "calql_pretrain",
                    "update_steps": update_steps,
                    "calql_pretrain_steps": calql_steps_done,
                    "offline_replay_size": 0 if offline_replay is None else len(offline_replay),
                    "update_time_sec": timing.get("time/offline_sample_sec", 0.0)
                    + timing.get("time/calql_update_sec", 0.0),
                    "speed/learner_wall_updates_per_sec": update_steps / max(wall_time_sec, 1e-9),
                    "speed/learner_active_updates_per_sec": update_steps / max(active_update_time_sec, 1e-9),
                    "wall_time_sec": wall_time_sec,
                    **timing,
                    **{f"train/{key}": value for key, value in last_update.items()},
                }
            )

        publish_start = time.perf_counter()
        server.publish_network(agent.policy_state_dict())
        initial_publish_time_sec = time.perf_counter() - publish_start
        write_metric(
            {
                "role": "learner",
                "algorithm": "pld",
                "event": "initial_policy_published",
                "time/publish_network_sec": initial_publish_time_sec,
                "update_steps": update_steps,
            }
        )
        steps_per_update = int(runtime.steps_per_update)
        checkpoint_period = int(runtime.checkpoint_period)

        while True:
            refresh_actor_summary_file()
            env_steps = current_env_steps(env_steps)
            if _learner_should_stop(update_steps, env_steps, int(runtime.max_update_steps), int(runtime.max_env_steps), actor_done):
                break
            online_updates = max(0, update_steps - calql_steps_done)
            min_online = max(int(runtime.training_starts), 1)
            if len(replay) < min_online or online_updates >= len(replay):
                now = time.perf_counter()
                if actor_done:
                    write_metric({
                        "role": "learner",
                        "event": "stopping_online_replay_exhausted",
                        "replay_size": len(replay),
                        "training_starts": int(runtime.training_starts),
                        "update_steps": update_steps,
                        "online_update_steps": online_updates,
                        "env_steps": env_steps,
                        "wall_time_sec": now - start_time,
                    })
                    break
                if now - last_wait_publish_time >= 1.0:
                    server.publish_network(agent.policy_state_dict())
                    last_wait_publish_time = now
                if now - last_wait_metric_time >= 1.0:
                    write_metric({
                        "role": "learner",
                        "event": "waiting_for_online_replay",
                        "replay_size": len(replay),
                        "training_starts": int(runtime.training_starts),
                        "update_steps": update_steps,
                        "online_update_steps": online_updates,
                        "env_steps": env_steps,
                        "actor_done": actor_done,
                        "wall_time_sec": now - start_time,
                    })
                    last_wait_metric_time = now
                time.sleep(float(runtime.get("update_sleep_sec", 0.05)))
                continue
            if update_steps >= int(runtime.max_update_steps):
                time.sleep(float(runtime.get("update_sleep_sec", 0.05)))
                continue
            step_timer = Timer()
            with step_timer.context("mixed_sample"):
                mixed = sampler.sample(int(runtime.batch_size))
            with step_timer.context("algorithm_update"):
                last_update = agent.update(mixed.batch)
            timing = step_timer.get_average_times(reset=False, prefix="time/", suffix="_sec")
            active_update_time_sec += sum(step_timer.get_total_times(reset=False).values())
            update_steps += 1
            env_steps = current_env_steps(env_steps)

            publish_time_sec = 0.0
            checkpoint_time_sec = 0.0
            if steps_per_update > 0 and update_steps > 0 and update_steps % steps_per_update == 0:
                publish_start = time.perf_counter()
                server.publish_network(agent.policy_state_dict())
                publish_time_sec += time.perf_counter() - publish_start
            if checkpoints is not None and checkpoint_period > 0 and update_steps > 0 and update_steps % checkpoint_period == 0:
                ckpt_start = time.perf_counter()
                save_checkpoint(checkpoints, agent, env_steps, update_steps, episodes, total_reward, config_snapshot, tag=f"update_{update_steps}.pt")
                checkpoint_time_sec += time.perf_counter() - ckpt_start

            wall_time_sec = time.perf_counter() - start_time
            write_metric(
                {
                    "role": "learner",
                    "phase": "online",
                    "env_steps": env_steps,
                    "update_steps": update_steps,
                    "online_update_steps": max(0, update_steps - calql_steps_done),
                    "calql_pretrain_steps": calql_steps_done,
                    "replay_size": len(replay),
                    "offline_replay_size": 0 if offline_replay is None else len(offline_replay),
                    "batch_mix": mixed.mix,
                    "update_time_sec": timing.get("time/mixed_sample_sec", 0.0)
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
        server.stop()
        if not completed_loop:
            finish_wandb_logger()

    env_steps = current_env_steps(env_steps)
    summary = {
        "role": "learner",
        "algorithm": "pld",
        "env_steps": env_steps,
        "update_steps": update_steps,
        "online_update_steps": max(0, update_steps - calql_steps_done),
        "calql_pretrain_steps": calql_steps_done,
        "replay_size": len(replay),
        "offline_replay_size": 0 if offline_replay is None else len(offline_replay),
        "offline_stats": offline_stats,
        "last_algorithm_updates": last_update.get("updates", 0),
    }
    try:
        if checkpoints is not None:
            save_checkpoint(checkpoints, agent, env_steps, update_steps, episodes, total_reward, config_snapshot, tag="final.pt")
        if run_dir is not None:
            (run_dir / "summary.json").write_text(json.dumps(json_sanitize(summary), indent=2, sort_keys=True) + "\n")
        write_metric({"summary": summary})
    finally:
        finish_wandb_logger()
    return summary


def run_actor(cfg: DictConfig) -> dict[str, Any]:
    runtime = cfg.runtime
    env = create_env(cfg)
    reference_policy = create_reference_policy(cfg)
    pld_obs_builder = create_pld_obs_builder(cfg)
    agent = create_pld_agent(cfg)
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
    deadline = time.perf_counter() + float(runtime.get("initial_weight_timeout_sec", 300.0))
    while not has_policy_state:
        client.update()
        if time.perf_counter() > deadline:
            raise TimeoutError("timed out waiting for initial learner policy weights")
        time.sleep(float(runtime.get("client_update_sleep_sec", 0.1)))

    try:
        obs = env.reset()
        env_steps = 0
        episodes = 0
        successes = 0
        total_reward = 0.0
        episode_return = 0.0
        steps_per_update = int(runtime.steps_per_update)
        log_period = int(runtime.log_period)
        start_time = time.perf_counter()
        active_rollout_time_sec = 0.0
        episode_success = False
        episode_steps = 0
        while env_steps < int(runtime.max_env_steps):
            chunk_start = time.perf_counter()
            chunk_timer = Timer()
            with chunk_timer.context("reference_policy"):
                base_actions = predict_base_actions(
                    reference_policy,
                    obs,
                    horizon=int(runtime.execute_horizon),
                    action_dim=int(cfg.algorithm.action_dim),
                )
            with chunk_timer.context("build_pld_obs"):
                pld_obs = build_pld_obs(obs, base_actions, builder=pld_obs_builder)
            base_warmup = episodes < int(runtime.base_warmup_episodes)
            if base_warmup:
                final_actions = base_actions
            else:
                with chunk_timer.context("sample_action"):
                    final_actions = agent.sample_action(pld_obs)
            final_actions = np.asarray(final_actions, dtype=np.float32)
            execute_actions = final_actions
            with chunk_timer.context("env_step_chunk"):
                next_obs, reward, done, truncated, info = env.step_chunk(execute_actions)
            executed_steps = int(info.get("executed_steps", len(execute_actions)))
            env_steps += executed_steps
            episode_steps += executed_steps
            total_reward += float(reward)
            episode_return += float(reward)
            terminal = bool(done or truncated)
            episode_success = bool(episode_success or info.get("success", False) or info.get("env_done", False))
            next_pld_obs = None
            if not terminal:
                with chunk_timer.context("next_reference_policy"):
                    next_base_actions = predict_base_actions(
                        reference_policy,
                        next_obs,
                        horizon=int(runtime.execute_horizon),
                        action_dim=int(cfg.algorithm.action_dim),
                    )
                with chunk_timer.context("next_build_pld_obs"):
                    next_pld_obs = build_pld_obs(next_obs, next_base_actions, builder=pld_obs_builder)
            transition = Transition(
                obs=pld_obs,
                next_obs=next_pld_obs,
                action=np.asarray(execute_actions, dtype=np.float32).reshape(-1),
                reward=float(reward),
                done=bool(done),
                truncated=bool(truncated),
                discount=0.0 if terminal else float(runtime.gamma) ** executed_steps,
                executed_steps=executed_steps,
                env_steps=env_steps,
                info={**info, "base_warmup": bool(base_warmup)},
            )
            with chunk_timer.context("send_transition"):
                data_store.insert(transition.to_payload())
                client.update()
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
                                "length": episode_steps,
                                "success": bool(episode_success),
                                "env_steps": env_steps,
                            }
                        }
                    },
                )
                episodes += 1
                successes += int(episode_success)
                episode_success = False
                episode_return = 0.0
                episode_steps = 0
                reset_start = time.perf_counter()
                obs = env.reset()
                reset_time_sec = time.perf_counter() - reset_start
            else:
                obs = next_obs
            timing = chunk_timer.get_average_times(reset=False, prefix="time/", suffix="_sec")
            wall_time_sec = time.perf_counter() - start_time
            metric = {
                "role": "actor",
                "algorithm": "pld",
                "env_steps": env_steps,
                "episode": metric_episode,
                "reward": float(reward),
                "done": bool(done),
                "truncated": bool(truncated),
                "executed_steps": executed_steps,
                "base_warmup": bool(base_warmup),
                "wall_time_sec": wall_time_sec,
                "chunk_time_sec": chunk_time_sec,
                "time/reset_env_sec": reset_time_sec,
                "speed/actor_wall_env_steps_per_sec": env_steps / max(wall_time_sec, 1e-9),
                "speed/actor_active_env_steps_per_sec": env_steps / max(active_rollout_time_sec, 1e-9),
                **timing,
            }
            write_actor_metric(metric)
            if log_period > 0 and env_steps % log_period == 0:
                client.request(str(runtime.request_type), {"timer": chunk_timer.get_average_times()})
            if steps_per_update > 0 and env_steps % steps_per_update == 0:
                client.update()
        summary = {
            "role": "actor",
            "algorithm": "pld",
            "env_steps": env_steps,
            "episodes": episodes,
            "successes": successes,
            "total_reward": total_reward,
            "received_policy_state": has_policy_state,
        }
        summary = send_actor_summary(client, str(runtime.request_type), summary, run_dir, write_actor_metric)
        write_actor_metric({"summary": summary})
        return summary
    finally:
        client.stop()
        env.close()
        reference_policy.close()


def _load_offline_replay(runtime: DictConfig) -> tuple[ReplayBuffer | None, dict[str, Any]]:
    path = runtime.get("offline_replay_path", None)
    if not path:
        if bool(runtime.require_offline):
            raise RuntimeError("PLD requires offline replay; set runtime.offline_replay_path")
        return None, {"episodes_loaded": 0, "transitions_loaded": 0}
    replay, stats = load_pld_offline_replay(
        path,
        capacity=int(runtime.offline_capacity),
        seed=0,
        max_episodes=runtime.get("offline_max_episodes", None),
        max_transitions=runtime.get("offline_max_transitions", None),
    )
    if bool(runtime.require_offline) and len(replay) == 0:
        raise RuntimeError(f"PLD offline replay is empty: {path}")
    return replay, stats


def _learner_should_stop(update_steps: int, env_steps: int, max_update_steps: int, max_env_steps: int, actor_done: bool) -> bool:
    if update_steps < max_update_steps:
        return False
    if max_env_steps <= 0:
        return True
    return env_steps >= max_env_steps or actor_done



if __name__ == "__main__":
    main()
