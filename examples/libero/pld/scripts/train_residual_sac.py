#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
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

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer

from examples.libero.common.async_eval import start_async_eval_worker
from examples.libero.pld.metrics import actor_episode_metric, residual_sac_learner_metric_aliases
from examples.libero.pld.config import (
    create_env,
    create_pld_agent,
    create_pld_obs_builder,
    create_reference_policy,
    load_config,
    predict_base_actions,
    reference_action_policy_horizon,
    validate_pld_cfg,
)
from vla_rl.algorithms.pld import build_pld_obs
from vla_rl.data import MemoryEfficientReplayBuffer, Transition
from vla_rl.rewards.processor import PendingRewardTransition, build_reward_processor
from vla_rl.runtime.agentlace import json_sanitize, make_agentlace_replay_store
from vla_rl.runtime.async_eval import (
    append_async_eval_request,
    append_async_eval_stop,
    check_async_eval_worker,
    load_new_async_eval_results,
    wait_for_async_eval_worker,
)
from vla_rl.runtime.checkpoint import CheckpointManager
from vla_rl.runtime.run_utils import (
    apply_actor_summary_file as read_actor_summary_file,
    make_jsonl_metric_writer,
    run_dir_from_runtime,
    save_checkpoint,
    send_actor_summary,
)
from vla_rl.runtime.timer import Timer
from vla_rl.runtime.wandb import make_wandb_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIBERO online residual SAC actor or learner.")
    parser.add_argument("--config", required=True, help="Path to a residual SAC YAML config.")
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
    _assert_online_only(runtime)
    agent = create_pld_agent(cfg)
    run_dir = run_dir_from_runtime(runtime)
    checkpoints = CheckpointManager(run_dir) if run_dir is not None else None
    replay = MemoryEfficientReplayBuffer(capacity=int(runtime.replay_capacity), seed=int(runtime.replay_seed))
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

    async_eval = start_async_eval_worker(runtime, run_dir=run_dir, algorithm="pld")
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
    latest_completed_episode_id = int(episodes)
    completed_episode_env_steps: dict[int, int] = {}
    last_queued_async_eval_episode = 0

    def current_env_steps(value: int) -> int:
        return max(int(value), int(replay.latest_env_steps), int(actor_done_env_steps))

    def write_metric(metric: dict[str, Any]) -> None:
        if run_dir is None:
            wandb_logger.log(metric, step=update_steps)
            return
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_sanitize(metric), sort_keys=True) + "\n")
        wandb_logger.log(metric, step=update_steps)

    def refresh_actor_summary_file() -> None:
        nonlocal actor_done, actor_done_env_steps
        actor_done, actor_done_env_steps = read_actor_summary_file(run_dir, actor_done, actor_done_env_steps)

    def request_callback(request_type: str, payload: Any) -> Any:
        nonlocal actor_done, actor_done_env_steps, latest_completed_episode_id, episodes, total_reward
        if request_type != str(runtime.request_type):
            return {"ok": False, "error": f"unsupported request type: {request_type}"}
        stat = dict(payload or {})
        stat.setdefault("role", "actor")
        if stat.get("event") == "actor_summary":
            actor_done = True
            actor_done_env_steps = max(actor_done_env_steps, int(stat.get("env_steps", 0)))
        episode_id = int(stat.get("rollout/episode_id", 0) or 0)
        if episode_id > 0:
            latest_completed_episode_id = max(latest_completed_episode_id, episode_id)
            episodes = max(episodes, episode_id)
            episode_env_steps = stat.get("env_steps", None)
            if episode_env_steps is None:
                episode_env_steps = stat.get("environment", {}).get("episode", {}).get("env_steps", 0)
            completed_episode_env_steps[episode_id] = max(0, int(episode_env_steps or 0))
            episode_return = stat.get("environment", {}).get("episode", {}).get("return", None)
            if episode_return is not None:
                total_reward += float(episode_return)
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

    steps_per_update = int(runtime.steps_per_update)
    critic_actor_ratio = max(1, int(cfg.algorithm.get("critic_actor_ratio", 1)))
    checkpoint_period = int(runtime.checkpoint_period)
    start_time = time.perf_counter()
    active_update_time_sec = 0.0
    last_wait_metric_time = 0.0
    last_wait_publish_time = time.perf_counter()
    last_update: dict[str, Any] = {}

    def queue_async_eval(
        *,
        eval_tag: str,
        train_episode_id: int,
        train_env_steps: int,
        force_zero_residual: bool = False,
    ) -> None:
        if async_eval.eval_checkpoint_dir is None or run_dir is None:
            return
        async_cfg = runtime.async_eval
        eval_checkpoint_path = async_eval.eval_checkpoint_dir / f"{eval_tag}.pt"
        torch.save(
            {
                "version": 1,
                "env_steps": int(train_env_steps),
                "update_steps": int(update_steps),
                "episodes": int(train_episode_id),
                "total_reward": float(total_reward),
                "algorithm_state": agent.state_dict(),
                "config": config_snapshot,
            },
            eval_checkpoint_path,
        )
        append_async_eval_request(
            async_eval,
            {
                "eval_index": int(async_eval.triggered_count),
                "train_episode_id": int(train_episode_id),
                "train_env_steps": int(train_env_steps),
                "train_update_steps": int(update_steps),
                "checkpoint_path": str(eval_checkpoint_path),
                "output_dir": str(run_dir / "eval_runs" / eval_tag),
                "episodes": int(async_cfg.get("episodes", 50)),
                "max_env_steps_per_episode": int(async_cfg.get("max_env_steps_per_episode", 0) or 0),
                "save_videos": bool(async_cfg.get("save_videos", False)),
                "env_url": None if async_cfg.get("env_url", None) is None else str(async_cfg.get("env_url")),
                "policy_url": str(async_cfg.get("policy_url")),
                "force_zero_residual": bool(force_zero_residual),
            },
        )
        write_metric(
            {
                "role": "learner",
                "event": "async_eval_queued",
                "eval_index": int(async_eval.triggered_count - 1),
                "train_episode_id": int(train_episode_id),
                "env_steps": int(train_env_steps),
                "update_steps": int(update_steps),
                "force_zero_residual": bool(force_zero_residual),
            }
        )

    def maybe_queue_initial_async_eval() -> None:
        if not async_eval.enabled or async_eval.eval_checkpoint_dir is None or run_dir is None:
            return
        if update_steps != 0 or latest_completed_episode_id != 0:
            return
        queue_async_eval(
            eval_tag="episode_0",
            train_episode_id=0,
            train_env_steps=0,
            force_zero_residual=True,
        )

    def maybe_queue_async_eval() -> None:
        nonlocal last_queued_async_eval_episode
        if not async_eval.enabled or async_eval.eval_checkpoint_dir is None or run_dir is None:
            return
        if update_steps <= 0:
            return
        every_episodes = int(async_eval.every_episodes)
        while latest_completed_episode_id >= last_queued_async_eval_episode + every_episodes:
            target_episode = last_queued_async_eval_episode + every_episodes
            queue_async_eval(
                eval_tag=f"episode_{target_episode}",
                train_episode_id=target_episode,
                train_env_steps=int(completed_episode_env_steps.get(target_episode, env_steps)),
            )
            last_queued_async_eval_episode = target_episode

    completed_loop = False
    try:
        maybe_queue_initial_async_eval()
        while True:
            refresh_actor_summary_file()
            env_steps = current_env_steps(env_steps)
            maybe_queue_async_eval()

            min_replay_size = max(int(runtime.training_starts), int(runtime.batch_size))
            if len(replay) < min_replay_size:
                check_async_eval_worker(async_eval)
                for eval_result in load_new_async_eval_results(async_eval):
                    write_metric(eval_result)
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
                            "env_steps": current_env_steps(env_steps),
                            "wall_time_sec": now - start_time,
                            **residual_sac_learner_metric_aliases(
                                {},
                                update_steps=update_steps,
                                env_steps=current_env_steps(env_steps),
                                replay_size=len(replay),
                            ),
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
                            "env_steps": current_env_steps(env_steps),
                            "wall_time_sec": now - start_time,
                            **residual_sac_learner_metric_aliases(
                                {},
                                update_steps=update_steps,
                                env_steps=current_env_steps(env_steps),
                                replay_size=len(replay),
                            ),
                        }
                    )
                    last_wait_metric_time = now
                time.sleep(float(runtime.get("update_sleep_sec", 0.05)))
                continue

            step_timer = Timer()
            critic_only_infos: list[dict[str, Any]] = []
            for _ in range(max(0, critic_actor_ratio - 1)):
                with step_timer.context("critic_only_update"):
                    critic_batch = replay.sample(int(runtime.batch_size))
                    critic_only_infos.append(agent.update_critics(critic_batch))
            with step_timer.context("sample_replay"):
                batch = replay.sample(int(runtime.batch_size))
            with step_timer.context("algorithm_update"):
                last_update = agent.update_high_utd(batch)
            if critic_only_infos:
                last_update = {
                    **last_update,
                    "extra_critic_updates": float(len(critic_only_infos)),
                }
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
            if int(runtime.log_period) > 0 and update_steps > 0 and update_steps % int(runtime.log_period) == 0:
                check_async_eval_worker(async_eval)
                for eval_result in load_new_async_eval_results(async_eval):
                    write_metric(eval_result)
            if (
                checkpoints is not None
                and checkpoint_period > 0
                and update_steps > 0
                and update_steps % checkpoint_period == 0
            ):
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

            wall_time_sec = time.perf_counter() - start_time
            write_metric(
                {
                    "role": "learner",
                    "algorithm": "residual_sac",
                    "env_steps": env_steps,
                    "update_steps": update_steps,
                    "episodes": episodes,
                    "replay_size": len(replay),
                    "update_time_sec": sum(float(value) for value in timing.values()),
                    "time/publish_network_sec": publish_time_sec,
                    "time/save_checkpoint_sec": checkpoint_time_sec,
                    "speed/learner_wall_updates_per_sec": update_steps / max(wall_time_sec, 1e-9),
                    "speed/learner_active_updates_per_sec": update_steps / max(active_update_time_sec, 1e-9),
                    "wall_time_sec": wall_time_sec,
                    **timing,
                    **{f"train/{key}": value for key, value in last_update.items()},
                    **residual_sac_learner_metric_aliases(
                        last_update,
                        update_steps=update_steps,
                        env_steps=env_steps,
                        replay_size=len(replay),
                    ),
                }
            )
            if actor_done:
                maybe_queue_async_eval()
                break
        completed_loop = True
    finally:
        try:
            maybe_queue_async_eval()
            append_async_eval_stop(async_eval)
            wait_for_async_eval_worker(
                async_eval,
                poll_interval_sec=float(runtime.get("async_eval", {}).get("poll_interval_sec", 5.0))
                if runtime.get("async_eval", None)
                else 5.0,
            )
            for eval_result in load_new_async_eval_results(async_eval):
                write_metric(eval_result)
        except Exception as exc:
            write_metric({"role": "learner", "event": "async_eval_shutdown_failed", "error": str(exc)})
        stop = getattr(server, "stop", None)
        if callable(stop):
            stop()
        if not completed_loop:
            finish_wandb_logger()

    env_steps = current_env_steps(env_steps)
    summary = {
        "role": "learner",
        "algorithm": "residual_sac",
        "env_steps": env_steps,
        "update_steps": update_steps,
        "episodes": episodes,
        "replay_size": len(replay),
        "last_algorithm_updates": last_update.get("updates", 0),
    }
    try:
        if checkpoints is not None:
            save_checkpoint(
                checkpoints,
                agent,
                env_steps,
                update_steps,
                episodes,
                total_reward,
                config_snapshot,
                tag="final.pt",
            )
        if run_dir is not None:
            (run_dir / "summary.json").write_text(json.dumps(json_sanitize(summary), indent=2, sort_keys=True) + "\n")
        write_metric({"summary": summary})
    finally:
        finish_wandb_logger()
    return summary


def run_actor(cfg: DictConfig) -> dict[str, Any]:
    runtime = cfg.runtime
    _assert_online_only(runtime)
    env = create_env(cfg)
    reference_policy = create_reference_policy(cfg)
    pld_obs_builder = create_pld_obs_builder(cfg)
    agent = create_pld_agent(cfg)
    run_dir = run_dir_from_runtime(runtime)
    write_actor_metric = make_jsonl_metric_writer(run_dir, "actor_metrics.jsonl")
    write_progress_event = _make_progress_event_writer(cfg, run_dir)

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
    reward_processor = None

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
        episode_steps = 0
        active_rollout_time_sec = 0.0
        start_time = time.perf_counter()
        timer = Timer()
        execute_horizon = int(runtime.execute_horizon)
        policy_horizon = reference_action_policy_horizon(cfg)
        steps_per_update = int(runtime.steps_per_update)
        log_period = int(runtime.log_period)
        recent_successes: deque[int] = deque(maxlen=50)
        actor_chunk_index = 0
        last_reward_committed = 0
        cached_pld_obs: dict[str, np.ndarray] | None = None
        cached_base_actions: np.ndarray | None = None

        reward_processor = build_reward_processor(
            cfg,
            data_store=data_store,
            gamma=float(runtime.gamma),
            metric_writer=write_actor_metric,
            progress_event_writer=write_progress_event,
        )

        while env_steps < int(runtime.max_env_steps):
            timer.tick("total")
            chunk_start = time.perf_counter()
            chunk_start_env_steps = env_steps

            if cached_pld_obs is None:
                with timer.context("reference_policy"):
                    base_actions = predict_base_actions(
                        reference_policy,
                        obs,
                        horizon=execute_horizon,
                        action_dim=int(cfg.algorithm.action_dim),
                        policy_horizon=policy_horizon,
                    )
                with timer.context("build_pld_obs"):
                    pld_obs = build_pld_obs(obs, base_actions, builder=pld_obs_builder)
            else:
                pld_obs = cached_pld_obs
                base_actions = cached_base_actions
                cached_pld_obs = None
                cached_base_actions = None

            base_warmup = episodes < int(runtime.get("base_warmup_episodes", 0))
            if base_warmup:
                final_actions = np.asarray(base_actions, dtype=np.float32)
            else:
                with timer.context("sample_action"):
                    final_actions = agent.sample_action(pld_obs)
            final_actions = np.asarray(final_actions, dtype=np.float32)

            remaining_env_steps = int(runtime.max_env_steps) - env_steps
            actions_to_execute = final_actions[: min(execute_horizon, remaining_env_steps)]
            with timer.context("env_step_chunk"):
                next_obs, reward, done, truncated, info = env.step_chunk(actions_to_execute, return_steps=True)

            info = _normalize_chunk_info(
                info,
                fallback_reward=float(reward),
                fallback_steps=len(actions_to_execute),
                fallback_done=bool(done),
                fallback_truncated=bool(truncated),
            )
            executed_steps = int(info["executed_steps"])
            env_reward = float(info["chunk_reward"])
            done = bool(info["done"])
            truncated = bool(info["truncated"])
            chunk_success = bool(info["success"])
            critic_terminal = chunk_success
            terminal = bool(done or truncated)
            env_steps += executed_steps
            episode_steps += executed_steps
            episode_return += env_reward
            total_reward += env_reward

            next_pld_obs = None
            next_base_actions = None
            if not critic_terminal:
                with timer.context("next_reference_policy"):
                    next_base_actions = predict_base_actions(
                        reference_policy,
                        next_obs,
                        horizon=execute_horizon,
                        action_dim=int(cfg.algorithm.action_dim),
                        policy_horizon=policy_horizon,
                    )
                with timer.context("next_build_pld_obs"):
                    next_pld_obs = build_pld_obs(next_obs, next_base_actions, builder=pld_obs_builder)

            transition = Transition(
                obs=pld_obs,
                next_obs=next_pld_obs,
                action=np.asarray(final_actions, dtype=np.float32).reshape(-1),
                reward=env_reward,
                done=done,
                truncated=truncated,
                discount=0.0 if critic_terminal else float(runtime.gamma) ** executed_steps,
                executed_steps=executed_steps,
                env_steps=env_steps,
                info={
                    **dict(info.get("last_step_info", {})),
                    "algorithm": "residual_sac",
                    "base_warmup": bool(base_warmup),
                    "critic_terminal": bool(critic_terminal),
                    "chunk_start_env_steps": int(chunk_start_env_steps),
                    "chunk_reward": float(env_reward),
                    "chunk_success": bool(chunk_success),
                    "replan_steps": execute_horizon,
                },
            )

            with timer.context("send_transition"):
                submitted_transitions = reward_processor.submit(
                    PendingRewardTransition(
                        episode_id=int(episodes),
                        chunk_index=int(actor_chunk_index),
                        start_observation=obs,
                        end_observation=next_obs,
                        transition=transition,
                        env_reward=env_reward,
                        executed_steps=executed_steps,
                        task=next_obs.task or obs.task,
                    )
                )
                actor_chunk_index += 1
                client.update()

            metric_episode = episodes
            chunk_time_sec = time.perf_counter() - chunk_start
            active_rollout_time_sec += chunk_time_sec
            reset_time_sec = 0.0
            if terminal or env_steps >= int(runtime.max_env_steps):
                episode_success = chunk_success
                recent_successes.append(int(episode_success))
                successes += int(episode_success)
                completed_episode_id = episodes + 1
                recent_success_rate_50 = float(sum(recent_successes)) / max(1, len(recent_successes))
                client.request(
                    str(runtime.request_type),
                    {
                        **actor_episode_metric(
                            episode_id=completed_episode_id,
                            episode_return=episode_return,
                            episode_steps=episode_steps,
                            success=episode_success,
                            env_steps=env_steps,
                            recent_success_rate_50=recent_success_rate_50,
                        ),
                        "env_steps": env_steps,
                    },
                )
                client.update()
                episodes += 1
                episode_return = 0.0
                episode_steps = 0
                cached_pld_obs = None
                cached_base_actions = None
                if env_steps < int(runtime.max_env_steps):
                    reset_start = time.perf_counter()
                    with timer.context("reset_env"):
                        obs = env.reset()
                    reset_time_sec = time.perf_counter() - reset_start
            else:
                obs = next_obs
                cached_pld_obs = next_pld_obs
                cached_base_actions = next_base_actions

            wall_time_sec = time.perf_counter() - start_time
            timer.tock("total")
            reward_stats = reward_processor.stats()
            reward_committed = int(reward_stats.get("committed", 0))
            inserted_transitions = max(0, reward_committed - last_reward_committed)
            last_reward_committed = reward_committed
            timing = timer.get_average_times(reset=False, prefix="time/", suffix="_sec")
            metric = {
                "role": "actor",
                "algorithm": "residual_sac",
                "env_steps": env_steps,
                "episode": metric_episode,
                "reward": env_reward,
                "done": done,
                "truncated": truncated,
                "success": chunk_success,
                "executed_steps": executed_steps,
                "chunk_size": execute_horizon,
                "replay_transitions": inserted_transitions,
                "submitted_transitions": submitted_transitions,
                "base_warmup": bool(base_warmup),
                "wall_time_sec": wall_time_sec,
                "chunk_time_sec": chunk_time_sec,
                "time/reset_env_sec": reset_time_sec,
                "speed/actor_wall_env_steps_per_sec": env_steps / max(wall_time_sec, 1e-9),
                "speed/actor_active_env_steps_per_sec": env_steps / max(active_rollout_time_sec, 1e-9),
                "reward_processor": reward_stats,
                **timing,
            }
            write_actor_metric(metric)

            if log_period > 0 and env_steps % log_period == 0:
                client.request(
                    str(runtime.request_type),
                    {
                        "timer": timer.get_average_times(),
                        "actor": {
                            "env_steps": env_steps,
                            "episodes": episodes,
                            "total_reward": total_reward,
                            "successes": successes,
                            "speed_wall_env_steps_per_sec": env_steps / max(wall_time_sec, 1e-9),
                            "speed_active_env_steps_per_sec": env_steps / max(active_rollout_time_sec, 1e-9),
                            "reward_processor": reward_stats,
                        },
                    },
                )

            if steps_per_update > 0 and env_steps % steps_per_update == 0:
                client.update()

        reward_summary = reward_processor.close(drain=True)
        client.update()
        summary = {
            "role": "actor",
            "algorithm": "residual_sac",
            "env_steps": env_steps,
            "episodes": episodes,
            "successes": successes,
            "total_reward": total_reward,
            "received_policy_state": has_policy_state,
            "reward": reward_summary,
        }
        summary = send_actor_summary(client, str(runtime.request_type), summary, run_dir, write_actor_metric)
        write_actor_metric({"summary": summary})
        return summary
    finally:
        try:
            if reward_processor is not None:
                reward_processor.close(drain=True)
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


def _assert_online_only(runtime: DictConfig) -> None:
    offline_cfg = runtime.get("offline", {})
    if bool(offline_cfg.get("enabled", False)):
        raise NotImplementedError(
            "residual_sac currently implements online-only chunk replay; "
            "set runtime.offline.enabled=false"
        )


def _make_progress_event_writer(cfg: DictConfig, run_dir: Path | None):
    reward_cfg = cfg.get("reward", None)
    if reward_cfg is None:
        return None
    reward_type = str(reward_cfg.get("type", "sparse"))
    reward_source = str(
        reward_cfg.get(
            "source",
            "env" if reward_type in {"sparse", "env", "none"} else "remote_progress",
        )
    )
    if reward_source == "remote_progress":
        return make_jsonl_metric_writer(run_dir, "progress_events.jsonl")
    return None


def _normalize_chunk_info(
    info: dict[str, Any],
    *,
    fallback_reward: float,
    fallback_steps: int,
    fallback_done: bool,
    fallback_truncated: bool,
) -> dict[str, Any]:
    payload = dict(info or {})
    if "num_steps" in payload:
        executed_steps = int(payload.get("num_steps", fallback_steps))
    else:
        executed_steps = int(payload.get("executed_steps", fallback_steps))
    rewards = [float(value) for value in payload.get("rewards", [])]
    dones = [bool(value) for value in payload.get("dones", [])]
    truncateds = [bool(value) for value in payload.get("truncateds", [])]
    infos = [dict(value or {}) for value in payload.get("infos", [])]

    if rewards:
        chunk_reward = float(sum(rewards[:executed_steps]))
    else:
        chunk_reward = float(fallback_reward)
    last_info = infos[min(executed_steps, len(infos)) - 1] if infos and executed_steps > 0 else payload
    done = (
        bool(dones[min(executed_steps, len(dones)) - 1])
        if dones and executed_steps > 0
        else bool(payload.get("done", fallback_done))
    )
    truncated = (
        bool(truncateds[min(executed_steps, len(truncateds)) - 1])
        if truncateds and executed_steps > 0
        else bool(payload.get("truncated", fallback_truncated))
    )
    success = bool(_step_info_success(payload) or _step_info_success(last_info))
    if infos:
        success = bool(success or any(_step_info_success(step_info) for step_info in infos[:executed_steps]))
    return {
        **payload,
        "executed_steps": executed_steps,
        "chunk_reward": chunk_reward,
        "done": done,
        "truncated": truncated,
        "success": success,
        "last_step_info": last_info,
    }


def _step_info_success(info: dict[str, Any]) -> bool:
    return bool(info.get("success", False) or info.get("env_done", False) or info.get("is_success", False))


if __name__ == "__main__":
    main()
