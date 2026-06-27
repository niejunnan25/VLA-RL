#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer

from examples.libero.rlpd.async_eval import start_async_eval_worker
from examples.libero.rlpd.config import (
    create_env,
    create_rlpd_agent,
    create_rlpd_obs_builder,
    load_config,
    validate_rlpd_cfg,
)
from examples.libero.rlpd.metrics import (
    actor_episode_metric,
    actor_speed_stats,
    rlpd_learner_metric_aliases,
)
from examples.libero.rlpd.reward import PendingRLPDRewardTransition, build_rlpd_reward_processor
from examples.libero.rlpd.rollout import (
    assert_single_executed_step,
    assert_single_step_actions,
    single_step_action_summary,
    single_step_discount,
    step_info_success,
)
from vla_rl.algorithms.rlpd import build_rlpd_obs, load_offline_replay as load_rlpd_offline_replay
from vla_rl.data import MemoryEfficientReplayBuffer, MixedReplaySampler, Transition
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


MetricWriter = Callable[[dict[str, Any]], None]
EXPECTED_OFFLINE_IMAGE_PREPROCESS = "libero"


##############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIBERO single-step RLPD actor or learner.")
    parser.add_argument("--config", required=True, help="Path to an RLPD YAML config.")
    parser.add_argument("--role", required=True, choices=("actor", "learner"))
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="OmegaConf dotlist overrides after --.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    validate_rlpd_cfg(cfg)

    if args.role == "learner":
        summary = run_learner(cfg)
    else:
        summary = run_actor(cfg)
    print(json.dumps(json_sanitize(summary), sort_keys=True))


##############################################################################


def run_learner(cfg: DictConfig) -> dict[str, Any]:
    """Learner loop, SERL-style: receive actor data and update the SAC agent."""

    runtime = cfg.runtime
    agent = create_rlpd_agent(cfg)
    run_dir = run_dir_from_runtime(runtime)
    checkpoints = CheckpointManager(run_dir) if run_dir is not None else None
    replay = MemoryEfficientReplayBuffer(capacity=int(runtime.replay_capacity), seed=int(runtime.replay_seed))
    config_snapshot = OmegaConf.to_container(cfg, resolve=True)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(config_snapshot), run_dir / "config.yaml")
        if runtime.get("resume_from", None):
            (run_dir / "metrics.jsonl").touch(exist_ok=True)
        else:
            (run_dir / "metrics.jsonl").write_text("")

    offline_load_started = time.perf_counter()
    offline_replay, offline_stats = _load_offline_replay(cfg)
    offline_load_time_sec = time.perf_counter() - offline_load_started
    sampler = MixedReplaySampler(
        replay,
        offline_replay,
        offline_ratio=0.0 if offline_replay is None else float(runtime.offline_ratio),
    )

    update_steps = 0
    env_steps = 0
    episodes = 0
    total_reward = 0.0
    actor_done = False
    actor_done_env_steps = 0
    latest_completed_episode_id = 0
    completed_episode_env_steps: dict[int, int] = {}
    last_queued_async_eval_episode = 0
    active_update_time_sec = 0.0
    last_wait_metric_time = 0.0
    last_wait_publish_time = time.perf_counter()
    last_update: dict[str, Any] = {}

    if runtime.get("resume_from", None):
        payload = checkpoints.load(runtime.resume_from, agent) if checkpoints is not None else {}
        update_steps = int(payload.get("update_steps", 0))
        env_steps = int(payload.get("env_steps", 0))
        episodes = int(payload.get("episodes", 0))
        total_reward = float(payload.get("total_reward", 0.0))

    wandb_logger = make_wandb_logger(cfg.get("wandb", None), variant=config_snapshot, run_dir=run_dir)
    wandb_finished = False

    def finish_wandb_logger() -> None:
        nonlocal wandb_finished
        if not wandb_finished:
            wandb_logger.finish()
            wandb_finished = True

    def write_metric(metric: dict[str, Any]) -> None:
        sanitized = json_sanitize(metric)
        if run_dir is not None:
            with (run_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(sanitized, sort_keys=True) + "\n")
        wandb_logger.log(sanitized, step=update_steps)

    async_eval = start_async_eval_worker(runtime, run_dir=run_dir)

    def current_env_steps(value: int) -> int:
        return max(int(value), int(replay.latest_env_steps), int(actor_done_env_steps))

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

    def queue_async_eval(*, eval_tag: str, train_episode_id: int, train_env_steps: int) -> None:
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
            },
        )
        write_metric(
            {
                "role": "learner",
                "algorithm": "rlpd",
                "event": "async_eval_queued",
                "eval_index": int(async_eval.triggered_count - 1),
                "train_episode_id": int(train_episode_id),
                "env_steps": int(train_env_steps),
                "update_steps": int(update_steps),
            }
        )

    trainer_config = TrainerConfig(
        port_number=int(runtime.trainer_port),
        broadcast_port=int(runtime.broadcast_port),
        request_types=[str(runtime.request_type)],
    )

    def request_callback(request_type: str, payload: Any) -> dict[str, Any]:
        nonlocal actor_done, actor_done_env_steps, episodes, total_reward, latest_completed_episode_id
        if request_type != str(runtime.request_type):
            return {"ok": False, "error": f"unsupported request type: {request_type}"}

        stat = dict(payload or {})
        stat.setdefault("role", "actor")
        if stat.get("event") == "actor_summary":
            actor_done = True
            actor_done_env_steps = max(actor_done_env_steps, int(stat.get("env_steps", 0)))

        environment = stat.get("environment", {})
        episode_payload = environment.get("episode", {}) if isinstance(environment, dict) else {}
        episode_id = int(stat.get("rollout/episode_id", 0) or episode_payload.get("episode_id", 0) or 0)
        if episode_payload:
            episodes = max(episodes, int(episode_payload.get("episode_id", episode_id or episodes)))
            total_reward = max(total_reward, float(stat.get("total_reward", total_reward)))
        if episode_id > 0:
            latest_completed_episode_id = max(latest_completed_episode_id, episode_id)
            episode_env_steps = stat.get("env_steps", None)
            if episode_env_steps is None:
                episode_env_steps = episode_payload.get("env_steps", 0)
            completed_episode_env_steps[episode_id] = max(0, int(episode_env_steps or 0))

        write_metric(stat)
        return {"ok": True}

    server = TrainerServer(trainer_config, request_callback=request_callback)
    server.register_data_store(str(runtime.store_name), make_agentlace_replay_store(replay))
    server.start(threaded=True)

    start_time = time.perf_counter()
    write_metric(
        {
            "role": "learner",
            "algorithm": "rlpd",
            "event": "offline_replay_loaded",
            "offline_replay_size": 0 if offline_replay is None else len(offline_replay),
            "offline_load_time_sec": offline_load_time_sec,
            "offline_stats": offline_stats,
        }
    )
    server.publish_network(agent.policy_state_dict())

    completed_loop = False
    try:
        while True:
            actor_done, actor_done_env_steps = read_actor_summary_file(run_dir, actor_done, actor_done_env_steps)
            env_steps = current_env_steps(env_steps)
            maybe_queue_async_eval()

            if _learner_should_stop(actor_done=actor_done):
                break

            min_replay_size = max(
                int(runtime.training_starts),
                int(runtime.batch_size),
            )
            if len(replay) < min_replay_size:
                check_async_eval_worker(async_eval)
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
                            "env_steps": env_steps,
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
                            "env_steps": env_steps,
                            "actor_done": actor_done,
                            "wall_time_sec": now - start_time,
                        }
                    )
                    last_wait_metric_time = now
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
            if int(runtime.steps_per_update) > 0 and update_steps % int(runtime.steps_per_update) == 0:
                publish_started = time.perf_counter()
                server.publish_network(agent.policy_state_dict())
                last_wait_publish_time = time.perf_counter()
                publish_time_sec = last_wait_publish_time - publish_started
            log_period = int(runtime.log_period)
            should_log_update = log_period > 0 and update_steps % log_period == 0
            if should_log_update:
                check_async_eval_worker(async_eval)
                for eval_result in load_new_async_eval_results(async_eval):
                    write_metric(eval_result)

            checkpoint_time_sec = 0.0
            if (
                checkpoints is not None
                and int(runtime.checkpoint_period) > 0
                and update_steps % int(runtime.checkpoint_period) == 0
            ):
                checkpoint_started = time.perf_counter()
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
                checkpoint_time_sec = time.perf_counter() - checkpoint_started

            wall_time_sec = time.perf_counter() - start_time
            if should_log_update:
                write_metric(
                    {
                        "role": "learner",
                        "algorithm": "rlpd",
                        "phase": "online",
                        "env_steps": env_steps,
                        "update_steps": update_steps,
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
                        **rlpd_learner_metric_aliases(
                            last_update,
                            update_steps=update_steps,
                            env_steps=env_steps,
                            replay_size=len(replay),
                            offline_replay_size=0 if offline_replay is None else len(offline_replay),
                            batch_mix=mixed.mix,
                        ),
                    }
                )
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
            write_metric({"role": "learner", "algorithm": "rlpd", "event": "async_eval_shutdown_failed", "error": str(exc)})
        _stop_if_available(server)
        if not completed_loop:
            finish_wandb_logger()

    env_steps = current_env_steps(env_steps)
    summary = {
        "role": "learner",
        "algorithm": "rlpd",
        "env_steps": env_steps,
        "update_steps": update_steps,
        "replay_size": len(replay),
        "offline_replay_size": 0 if offline_replay is None else len(offline_replay),
        "offline_stats": offline_stats,
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


##############################################################################


def run_actor(cfg: DictConfig) -> dict[str, Any]:
    """Actor loop: collect single-step LIBERO transitions and send them to learner."""

    runtime = cfg.runtime
    env = create_env(cfg)
    obs_builder = create_rlpd_obs_builder(cfg)
    agent = create_rlpd_agent(cfg)
    agent.set_compile_enabled(False)
    run_dir = run_dir_from_runtime(runtime)

    write_actor_metric = make_jsonl_metric_writer(run_dir, "actor_metrics.jsonl")
    write_progress_event = _make_progress_metric_writer(cfg, run_dir)
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
        if payload is None:
            return
        agent.load_policy_state_dict(payload)
        has_policy_state = True

    client.recv_network_callback(update_actor)
    deadline = time.perf_counter() + float(runtime.get("initial_weight_timeout_sec", 300.0))
    while not has_policy_state:
        client.update()
        if time.perf_counter() > deadline:
            raise TimeoutError("timed out waiting for initial learner policy weights")
        time.sleep(float(runtime.get("client_update_sleep_sec", 0.1)))

    reward_processor = None
    try:
        obs = env.reset()
        reward_processor = build_rlpd_reward_processor(
            cfg,
            data_store=data_store,
            gamma=float(runtime.gamma),
            metric_writer=write_actor_metric,
            progress_event_writer=write_progress_event,
        )

        timer = Timer()
        env_steps = 0
        episodes = 0
        successes = 0
        total_reward = 0.0
        episode_return = 0.0
        episode_success = False
        episode_steps = 0
        recent_successes: deque[int] = deque(maxlen=50)
        transition_index = 0
        last_reward_committed = 0
        active_rollout_time_sec = 0.0
        start_time = time.perf_counter()

        while env_steps < int(runtime.max_env_steps):
            step_started = time.perf_counter()
            timer.tick("total")
            step_timer = Timer()

            with step_timer.context("build_rlpd_obs"):
                rlpd_obs = build_rlpd_obs(obs, builder=obs_builder)

            with step_timer.context("sample_action"):
                actions = agent.sample_action(rlpd_obs)
            actions = assert_single_step_actions(actions, context="actor final_actions")

            with step_timer.context("env_step_chunk"):
                next_obs, env_reward, done, truncated, info = env.step_chunk(actions)

            executed_steps = assert_single_executed_step(info, context="env.step_chunk")
            next_env_steps = int(env_steps) + int(executed_steps)
            terminal = bool(done or truncated)
            step_success = step_info_success(info)
            critic_terminal = bool(terminal or step_success)

            next_rlpd_obs = None
            if not terminal:
                with step_timer.context("next_build_rlpd_obs"):
                    next_rlpd_obs = build_rlpd_obs(next_obs, builder=obs_builder)

            transition = Transition(
                obs=rlpd_obs,
                next_obs=next_rlpd_obs,
                action=actions.reshape(-1),
                reward=float(env_reward),
                done=bool(done),
                truncated=bool(truncated),
                discount=single_step_discount(gamma=float(runtime.gamma), terminal=critic_terminal),
                executed_steps=int(executed_steps),
                env_steps=int(next_env_steps),
                info={
                    **dict(info),
                    "algorithm": "rlpd",
                    "chunk_start_env_steps": int(env_steps),
                    "critic_terminal": bool(critic_terminal),
                    "terminal": bool(terminal),
                },
            )

            send_timer = Timer()
            with send_timer.context("send_transition"):
                reward_processor.submit(
                    PendingRLPDRewardTransition(
                        episode_id=int(episodes),
                        chunk_index=int(transition_index),
                        start_observation=obs,
                        end_observation=next_obs,
                        transition=transition,
                        env_reward=float(env_reward),
                        executed_steps=int(executed_steps),
                        task=next_obs.task or obs.task,
                    )
                )
            transition_index += 1

            env_steps = next_env_steps
            episode_steps += int(executed_steps)
            total_reward += float(env_reward)
            episode_return += float(env_reward)
            episode_success = bool(episode_success or step_success)

            metric_episode = episodes
            reset_time_sec = 0.0
            if terminal:
                completed_episode_id = episodes + 1
                recent_successes.append(int(episode_success))
                recent_success_rate_50 = float(sum(recent_successes)) / max(1, len(recent_successes))
                client.request(
                    str(runtime.request_type),
                    {
                        "role": "actor",
                        "algorithm": "rlpd",
                        "total_reward": total_reward,
                        **actor_episode_metric(
                            episode_id=completed_episode_id,
                            episode_return=episode_return,
                            episode_steps=episode_steps,
                            success=bool(episode_success),
                            env_steps=env_steps,
                            recent_success_rate_50=recent_success_rate_50,
                        ),
                    },
                )

                episodes += 1
                successes += int(episode_success)
                episode_return = 0.0
                episode_success = False
                episode_steps = 0
                transition_index = 0

                reset_started = time.perf_counter()
                obs = env.reset()
                reset_time_sec = time.perf_counter() - reset_started
            else:
                obs = next_obs

            step_time_sec = time.perf_counter() - step_started
            active_rollout_time_sec += step_time_sec
            timer.tock("total")

            reward_stats = reward_processor.stats()
            reward_committed = int(reward_stats.get("committed", 0))
            inserted_transitions = max(0, reward_committed - last_reward_committed)
            last_reward_committed = reward_committed
            wall_time_sec = time.perf_counter() - start_time
            timings = step_timer.get_average_times(reset=False, prefix="time/", suffix="_sec")
            timings.update(send_timer.get_average_times(reset=False, prefix="time/", suffix="_sec"))

            write_actor_metric(
                {
                    "role": "actor",
                    "algorithm": "rlpd",
                    "env_steps": env_steps,
                    "episode": metric_episode,
                    "reward": float(env_reward),
                    "done": bool(done),
                    "truncated": bool(truncated),
                    "executed_steps": int(executed_steps),
                    "replay_transitions": inserted_transitions,
                    "submitted_transitions": 1,
                    "wall_time_sec": wall_time_sec,
                    "step_time_sec": step_time_sec,
                    "time/reset_env_sec": reset_time_sec,
                    "speed/actor_wall_env_steps_per_sec": env_steps / max(wall_time_sec, 1e-9),
                    "speed/actor_active_env_steps_per_sec": env_steps / max(active_rollout_time_sec, 1e-9),
                    "reward_processor": reward_stats,
                    **{f"reward/{key}": value for key, value in reward_stats.items()},
                    **single_step_action_summary(actions),
                    **timings,
                }
            )

            if int(runtime.log_period) > 0 and env_steps % int(runtime.log_period) == 0:
                client.request(
                    str(runtime.request_type),
                    {
                        "timer": timer.get_average_times(),
                        "actor": {
                            "role": "actor",
                            "algorithm": "rlpd",
                            "last_reward": float(env_reward),
                            "speed/actor_wall_env_steps_per_sec": env_steps / max(wall_time_sec, 1e-9),
                            "speed/actor_active_env_steps_per_sec": env_steps / max(active_rollout_time_sec, 1e-9),
                            **actor_speed_stats(
                                env_steps=env_steps,
                                episodes=episodes,
                                successes=successes,
                                total_reward=total_reward,
                                wall_time_sec=wall_time_sec,
                                active_rollout_time_sec=active_rollout_time_sec,
                            ),
                        },
                    },
                )

            if int(runtime.steps_per_update) > 0 and env_steps % int(runtime.steps_per_update) == 0:
                client.update()

        reward_summary = reward_processor.close(drain=True)
        client.update()
        summary = {
            "role": "actor",
            "algorithm": "rlpd",
            "env_steps": env_steps,
            "episodes": episodes,
            "successes": successes,
            "total_reward": total_reward,
            "received_policy_state": bool(has_policy_state),
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
            _stop_if_available(client)
            _close_if_available(env)


##############################################################################


def _make_progress_metric_writer(cfg: DictConfig, run_dir: Path | None) -> MetricWriter | None:
    reward_cfg = cfg.get("reward", None)
    if reward_cfg is None:
        return None

    reward_type = str(reward_cfg.get("type", "sparse"))
    default_source = "env" if reward_type in {"sparse", "env", "none"} else "remote_progress"
    reward_source = str(reward_cfg.get("source", default_source))
    if reward_source != "remote_progress":
        return None
    return make_jsonl_metric_writer(run_dir, "progress_events.jsonl")


def _load_offline_replay(cfg: DictConfig) -> tuple[MemoryEfficientReplayBuffer | None, dict[str, Any]]:
    runtime = cfg.runtime
    path = runtime.get("offline_replay_path", None)
    if not path:
        raise RuntimeError("RLPD requires offline replay; set runtime.offline_replay_path")

    _assert_offline_replay_manifest(path)
    replay, stats = load_rlpd_offline_replay(
        path,
        capacity=int(runtime.offline_capacity),
        seed=0,
        max_episodes=runtime.get("offline_max_episodes", None),
        max_transitions=runtime.get("offline_max_transitions", None),
    )
    if len(replay) == 0:
        raise RuntimeError(f"RLPD offline replay is empty: {path}")

    _assert_standard_replay(
        replay,
        action_dim=int(cfg.algorithm.action_dim),
        image_keys=tuple(str(key) for key in cfg.rlpd_observation.image_keys),
        path=str(path),
    )
    return replay, stats


def _assert_offline_replay_manifest(path: str | Path) -> None:
    manifest_path = Path(path) / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"RLPD offline replay missing manifest.json: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    stats = payload.get("stats", {})
    image_preprocess = stats.get("image_preprocess", None)
    if image_preprocess != EXPECTED_OFFLINE_IMAGE_PREPROCESS:
        raise RuntimeError(
            f"RLPD offline replay {path} was converted with image_preprocess={image_preprocess!r}; "
            f"expected {EXPECTED_OFFLINE_IMAGE_PREPROCESS!r}. "
            "Regenerate the offline replay with the LIBERO RLPD converter so offline images match online observations."
        )


def _assert_standard_replay(replay: Any, *, action_dim: int, image_keys: tuple[str, ...], path: str) -> None:
    iter_transitions = getattr(replay, "iter_transitions", None)
    transitions = iter_transitions() if callable(iter_transitions) else iter(getattr(replay, "_items", ()))
    for index, transition in enumerate(transitions):
        if int(transition.executed_steps) != 1:
            raise AssertionError(
                f"RLPD offline replay must be single-step: "
                f"{path} transition {index} has executed_steps={transition.executed_steps}"
            )

        action = np.asarray(transition.action, dtype=np.float32).reshape(-1)
        if action.shape[0] != int(action_dim):
            raise AssertionError(
                f"RLPD offline replay must store one action per transition: "
                f"{path} transition {index} action shape={action.shape}, expected {(int(action_dim),)}"
            )

        _assert_standard_obs(transition.obs, image_keys=image_keys, path=path, index=index, name="obs")
        if transition.next_obs is not None:
            _assert_standard_obs(transition.next_obs, image_keys=image_keys, path=path, index=index, name="next_obs")


def _assert_standard_obs(obs: Any, *, image_keys: tuple[str, ...], path: str, index: int, name: str) -> None:
    if not isinstance(obs, dict):
        raise AssertionError(f"RLPD offline replay {path} transition {index} {name} must be an observation dict")
    if "base_action_chunk" in obs or "alpha" in obs:
        raise AssertionError(f"RLPD offline replay must not contain residual fields: {path} transition {index} {name}")
    if "proprio" not in obs:
        raise AssertionError(f"RLPD offline replay {path} transition {index} {name} missing proprio")

    for key in image_keys:
        image_key = f"image_{key}"
        if image_key not in obs:
            raise AssertionError(f"RLPD offline replay {path} transition {index} {name} missing {image_key}")


def _learner_should_stop(*, actor_done: bool) -> bool:
    return bool(actor_done)


def _stop_if_available(obj: Any) -> None:
    stop = getattr(obj, "stop", None)
    if callable(stop):
        stop()


def _close_if_available(obj: Any) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
