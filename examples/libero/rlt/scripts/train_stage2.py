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

from vla_rl.algorithms.rlt.features import encode_rlt_obs
from vla_rl.data import ReplayBuffer, Transition
from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer

from vla_rl.runtime.agentlace import json_sanitize, make_agentlace_replay_store
from vla_rl.runtime.async_eval import (
    AsyncEvalRuntime,
    append_async_eval_request,
    append_async_eval_stop,
    check_async_eval_worker,
    count_jsonl_lines,
    launch_async_eval_worker,
    load_new_async_eval_results,
    resolve_async_eval_path,
    wait_for_async_eval_worker,
)
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
from examples.libero.rlt.config import (
    build_reference_policy,
    create_env,
    create_rlt_agent,
    load_config,
    feature_cfg,
    load_rl_token_encoder,
    resolve_online_feature_source,
    rlt_cfg,
    validate_rlt_cfg,
)


def _subsample_observation_steps(action_steps: int, chunk_size: int, subsample_stride: int) -> list[int]:
    if subsample_stride <= 1:
        return []
    return [
        step
        for step in range(subsample_stride, action_steps + 1, subsample_stride)
        if step < chunk_size
    ]


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

    async_eval = start_async_eval_worker(runtime, run_dir=run_dir)
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
    latest_completed_episode_id = 0
    completed_episode_env_steps: dict[int, int] = {}
    last_queued_async_eval_episode = 0

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
        nonlocal actor_done, actor_done_env_steps, latest_completed_episode_id
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
            episode_env_steps = stat.get("env_steps", None)
            if episode_env_steps is None:
                episode_env_steps = stat.get("environment", {}).get("episode", {}).get("env_steps", 0)
            completed_episode_env_steps[episode_id] = max(0, int(episode_env_steps or 0))
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
    checkpoint_period = int(runtime.checkpoint_period)
    start_time = time.perf_counter()
    active_update_time_sec = 0.0
    last_wait_metric_time = 0.0
    last_wait_publish_time = time.perf_counter()
    last_update: dict[str, Any] = {}

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
                "env_url": str(async_cfg.get("env_url")),
                "policy_url": str(async_cfg.get("policy_url")),
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
            }
        )

    completed_loop = False
    try:
        while True:
            refresh_actor_summary_file()
            env_steps = current_env_steps(env_steps)
            maybe_queue_async_eval()
            if _learner_should_stop(
                update_steps=update_steps,
                env_steps=env_steps,
                max_update_steps=int(runtime.max_update_steps),
                max_env_steps=int(runtime.max_env_steps),
                actor_done=actor_done,
            ):
                break

            if update_steps >= int(runtime.max_update_steps):
                check_async_eval_worker(async_eval)
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
                time.sleep(float(runtime.get("update_sleep_sec", 0.05)))
                continue

            min_replay_size = max(int(runtime.training_starts), int(runtime.batch_size))
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
                time.sleep(float(runtime.get("update_sleep_sec", 0.05)))
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
            if steps_per_update > 0 and update_steps > 0 and update_steps % steps_per_update == 0:
                publish_start = time.perf_counter()
                server.publish_network(agent.policy_state_dict())
                publish_time_sec += time.perf_counter() - publish_start
            if int(runtime.log_period) > 0 and update_steps > 0 and update_steps % int(runtime.log_period) == 0:
                check_async_eval_worker(async_eval)
                for eval_result in load_new_async_eval_results(async_eval):
                    write_metric(eval_result)
            if checkpoints is not None and checkpoint_period > 0 and update_steps > 0 and update_steps % checkpoint_period == 0:
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
                    **rlt_learner_metric_aliases(
                        last_update,
                        update_steps=update_steps,
                        env_steps=env_steps,
                        replay_size=len(replay),
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
            write_metric({"role": "learner", "event": "async_eval_shutdown_failed", "error": str(exc)})
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
    feature = feature_cfg(cfg)
    online_feature_source = resolve_online_feature_source(feature)
    reference_policy_num_steps = int(feature.get("num_steps", 10))
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
        total_reward = 0.0
        episode_return = 0.0
        active_rollout_time_sec = 0.0
        steps_per_update = int(runtime.steps_per_update)
        log_period = int(runtime.log_period)
        start_time = time.perf_counter()
        timer = Timer()
        rlt = rlt_cfg(cfg)
        chunk_size = int(rlt.chunk_size)
        replan_steps = int(rlt.get("replan_steps", 5))
        subsample_stride = int(rlt.get("subsample_stride", 0) or 0)
        window_replay_enabled = subsample_stride > 1
        pending_chunk: dict[str, Any] | None = None
        cached_rlt_obs: dict[str, np.ndarray] | None = None
        cached_base_actions: np.ndarray | None = None
        episode_step = 0
        recent_successes: deque[int] = deque(maxlen=50)

        while env_steps < int(runtime.max_env_steps):
            timer.tick("total")
            chunk_start = time.perf_counter()
            chunk_start_env_steps = env_steps

            if cached_rlt_obs is None:
                with timer.context("reference_policy"):
                    base_actions, prefix_tokens, proprio = reference_policy.predict_actions_and_prefix(
                        obs,
                        feature_source=online_feature_source,
                        num_steps=reference_policy_num_steps,
                    )
                    base_actions = np.asarray(base_actions, dtype=np.float32)[:chunk_size]

                with timer.context("encode_rlt_obs"):
                    rlt_obs = encode_rlt_obs(
                        prefix_tokens,
                        base_actions,
                        proprio,
                        rl_token_encoder=rl_token_encoder,
                    )
            else:
                rlt_obs = cached_rlt_obs
                base_actions = cached_base_actions
                cached_rlt_obs = None
                cached_base_actions = None

            with timer.context("sample_actions"):
                if env_steps < int(runtime.warmup_steps):
                    actions = base_actions
                else:
                    actions = agent.sample_action(rlt_obs)

            actions = np.asarray(actions, dtype=np.float32)

            chunk_reward = 0.0
            done = False
            truncated = False
            info: dict[str, Any] = {}
            next_obs = obs
            executed_steps = 0
            chunk_success = False
            window_start_rlt_obs: list[dict[str, np.ndarray]] = []
            remaining_env_steps = int(runtime.max_env_steps) - env_steps
            actions_to_execute = actions[: min(replan_steps, remaining_env_steps)]
            subsample_observation_steps = _subsample_observation_steps(
                action_steps=len(actions_to_execute),
                chunk_size=chunk_size,
                subsample_stride=subsample_stride,
            )

            with timer.context("step_env"):
                next_obs, _, done, truncated, info = env.step_chunk(
                    actions_to_execute,
                    return_steps=True,
                    observation_indices=subsample_observation_steps,
                )
                subsample_obs_by_step = dict(zip(info["observation_indices"], info["observations"]))

                rewards = list(info["rewards"])
                dones = list(info["dones"])
                truncateds = list(info["truncateds"])
                infos = list(info["infos"])

                for step_idx in range(int(info["num_steps"])):
                    step_reward = float(rewards[step_idx])
                    done = bool(dones[step_idx])
                    truncated = bool(truncateds[step_idx])
                    info = dict(infos[step_idx])
                    chunk_reward += float(step_reward)
                    executed_steps += 1
                    env_steps += 1
                    episode_step += 1
                    episode_return += float(step_reward)
                    chunk_success = chunk_success or bool(
                        info.get("success", False) or info.get("env_done", False) or info.get("is_success", False)
                    )

                    if (
                        window_replay_enabled
                        and not bool(done or truncated)
                        and executed_steps % subsample_stride == 0
                        and executed_steps < chunk_size
                    ):
                        with timer.context("subsample_vla_inference"):
                            sub_obs = subsample_obs_by_step[executed_steps]
                            sub_base_actions, sub_prefix_tokens, sub_proprio = reference_policy.predict_actions_and_prefix(
                                sub_obs,
                                feature_source=online_feature_source,
                                num_steps=reference_policy_num_steps,
                            )
                            sub_base_actions = np.asarray(sub_base_actions, dtype=np.float32)[:chunk_size]
                            window_start_rlt_obs.append(
                                encode_rlt_obs(
                                    sub_prefix_tokens,
                                    sub_base_actions,
                                    sub_proprio,
                                    rl_token_encoder=rl_token_encoder,
                                )
                            )

                    if bool(done or truncated) or env_steps >= int(runtime.max_env_steps):
                        break

            reward = float(chunk_reward)
            total_reward += reward
            terminal = bool(done or truncated)

            with timer.context("next_reference_policy"):
                next_base_actions, next_prefix_tokens, next_proprio = reference_policy.predict_actions_and_prefix(
                    next_obs,
                    feature_source=online_feature_source,
                    num_steps=reference_policy_num_steps,
                )
                next_base_actions = np.asarray(next_base_actions, dtype=np.float32)[:chunk_size]
            with timer.context("next_encode_rlt_obs"):
                next_rlt_state = encode_rlt_obs(
                    next_prefix_tokens,
                    next_base_actions,
                    next_proprio,
                    rl_token_encoder=rl_token_encoder,
                )

            inserted_transitions = 0
            with timer.context("send_transition"):
                if window_replay_enabled:
                    if pending_chunk is not None:
                        pending_chunk["next_actions"] = actions.copy()
                        inserted_transitions += _insert_window_replay_transitions(
                            pending_chunk,
                            next_rlt_obs=rlt_obs,
                            next_window_start_rlt_obs=window_start_rlt_obs,
                            data_store=data_store,
                            subsample_stride=subsample_stride,
                            chunk_size=chunk_size,
                            gamma=float(runtime.gamma),
                        )
                    pending_chunk = {
                        "rlt_obs": rlt_obs,
                        "window_start_rlt_obs": window_start_rlt_obs,
                        "actions": actions.copy(),
                        "next_actions": None,
                        "reward": reward,
                        "done": bool(done),
                        "truncated": bool(truncated),
                        "terminal": terminal,
                        "executed_steps": executed_steps,
                        "env_steps": env_steps,
                        "info": dict(info),
                        "chunk_start_env_steps": chunk_start_env_steps,
                        "replan_steps": replan_steps,
                    }
                    if terminal or env_steps >= int(runtime.max_env_steps):
                        inserted_transitions += _insert_window_replay_transitions(
                            pending_chunk,
                            next_rlt_obs=next_rlt_state,
                            next_window_start_rlt_obs=[],
                            data_store=data_store,
                            subsample_stride=subsample_stride,
                            chunk_size=chunk_size,
                            gamma=float(runtime.gamma),
                        )
                        pending_chunk = None
                else:
                    transition = Transition(
                        obs=rlt_obs,
                        next_obs=next_rlt_state,
                        action=actions.reshape(-1),
                        reward=reward,
                        done=bool(done),
                        truncated=bool(truncated),
                        discount=0.0 if terminal else float(runtime.gamma) ** chunk_size,
                        executed_steps=executed_steps,
                        env_steps=env_steps,
                        info={**info, "chunk_start_env_steps": chunk_start_env_steps, "replan_steps": replan_steps},
                    )
                    data_store.insert(transition.to_payload())
                    inserted_transitions = 1
                if inserted_transitions > 0:
                    client.update()

            metric_episode = episodes
            chunk_time_sec = time.perf_counter() - chunk_start
            active_rollout_time_sec += chunk_time_sec
            reset_time_sec = 0.0
            if terminal:
                cached_rlt_obs = None
                cached_base_actions = None
                episode_success = bool(chunk_success)
                recent_successes.append(int(episode_success))
                recent_success_rate_50 = float(sum(recent_successes)) / max(1, len(recent_successes))
                completed_episode_id = episodes + 1
                client.request(
                    str(runtime.request_type),
                    {
                        "environment": {
                            "episode": {
                                "return": episode_return,
                                "length": episode_step,
                                "success": episode_success,
                                "env_steps": env_steps,
                            }
                        },
                        **rlt_rollout_metric_aliases(
                            episode_id=completed_episode_id,
                            episode_return=episode_return,
                            episode_steps=episode_step,
                            success=episode_success,
                            recent_success_rate_50=recent_success_rate_50,
                        ),
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
                cached_rlt_obs = next_rlt_state
                cached_base_actions = next_base_actions

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
                "replay_transitions": inserted_transitions,
                "wall_time_sec": wall_time_sec,
                "chunk_time_sec": chunk_time_sec,
                "time/reset_env_sec": reset_time_sec,
            }

            write_actor_metric(metric)
            if log_period > 0 and env_steps % log_period == 0:
                actor_stats = {
                    "env_steps": env_steps,
                    "episodes": episodes,
                    "total_reward": total_reward,
                    "wall_time_sec": wall_time_sec,
                    "active_env_steps_per_sec": env_steps / max(active_rollout_time_sec, 1e-9),
                    "wall_env_steps_per_sec": env_steps / max(wall_time_sec, 1e-9),
                }
                client.request(str(runtime.request_type), {"timer": timer.get_average_times(), "actor": actor_stats})
            if steps_per_update > 0 and env_steps % steps_per_update == 0:
                client.update()

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


def _insert_window_replay_transitions(
    pending_chunk: dict[str, Any],
    *,
    next_rlt_obs: dict[str, np.ndarray],
    next_window_start_rlt_obs: list[dict[str, np.ndarray]],
    data_store: Any,
    subsample_stride: int,
    chunk_size: int,
    gamma: float,
) -> int:
    """Insert yixin-style RLT replay windows for one completed chunk.

    Position 0 stores the original action chunk. Later positions store
    action_chunk[p:] + next_action_chunk[:p], so the critic sees the same
    fixed-length action window from multiple starts inside the chunk.
    """

    action_chunk = np.asarray(pending_chunk["actions"], dtype=np.float32)
    next_action_chunk = pending_chunk.get("next_actions")
    inserted = 0
    for pos_idx, position in enumerate(range(0, chunk_size, subsample_stride)):
        if position == 0:
            obs = pending_chunk["rlt_obs"]
            next_obs = next_rlt_obs
            action = action_chunk
        else:
            sub_idx = pos_idx - 1
            if sub_idx >= len(pending_chunk["window_start_rlt_obs"]) or sub_idx >= len(next_window_start_rlt_obs):
                break
            obs = pending_chunk["window_start_rlt_obs"][sub_idx]
            next_obs = next_window_start_rlt_obs[sub_idx]
            # Window replay uses the remaining current actions plus the next chunk prefix.
            tail = action_chunk[position:]
            if next_action_chunk is None:
                head = np.zeros((position, action_chunk.shape[1]), dtype=np.float32)
            else:
                head = np.asarray(next_action_chunk, dtype=np.float32)[:position]
            action = np.concatenate([tail, head], axis=0)

        transition = Transition(
            obs=obs,
            next_obs=next_obs,
            action=action.reshape(-1),
            reward=float(pending_chunk["reward"]),
            done=bool(pending_chunk["done"]),
            truncated=bool(pending_chunk["truncated"]),
            discount=0.0 if bool(pending_chunk["terminal"]) else float(gamma) ** int(chunk_size),
            executed_steps=int(pending_chunk["executed_steps"]),
            env_steps=int(pending_chunk["env_steps"]),
            info={
                **dict(pending_chunk["info"]),
                "chunk_start_env_steps": int(pending_chunk["chunk_start_env_steps"]),
                "subsample_stride": int(subsample_stride),
                "subsample_position": int(position),
            },
        )
        data_store.insert(transition.to_payload())
        inserted += 1
    return inserted


def start_async_eval_worker(runtime: DictConfig, *, run_dir: Path | None) -> AsyncEvalRuntime:
    async_cfg = runtime.get("async_eval", None)
    if async_cfg is None or not bool(async_cfg.get("enabled", False)):
        return AsyncEvalRuntime()
    if run_dir is None:
        raise ValueError("runtime.run_dir is required when runtime.async_eval.enabled=true")

    every_episodes = int(async_cfg.get("every_episodes", 50))
    if every_episodes <= 0:
        raise ValueError("runtime.async_eval.every_episodes must be positive")
    if not async_cfg.get("env_url", None) or not async_cfg.get("policy_url", None):
        raise ValueError("runtime.async_eval.env_url and policy_url are required when async eval is enabled")

    queue_path = resolve_async_eval_path(async_cfg.get("queue_file", "eval_queue.jsonl"), run_dir=run_dir)
    summary_path = resolve_async_eval_path(async_cfg.get("summary_jsonl", "eval_summary.jsonl"), run_dir=run_dir)
    worker_log_path = resolve_async_eval_path(async_cfg.get("worker_log_file", "eval_worker.log"), run_dir=run_dir)
    eval_checkpoint_dir = resolve_async_eval_path(async_cfg.get("checkpoint_dir", "eval_checkpoints"), run_dir=run_dir)
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text("")
    summary_path.touch(exist_ok=True)

    worker_script = Path(__file__).resolve().parent / "process_eval_queue.py"
    cmd = [
        sys.executable,
        str(worker_script),
        "--train-config",
        str(run_dir / "config.yaml"),
        "--queue-file",
        str(queue_path),
        "--summary-jsonl",
        str(summary_path),
        "--poll-interval-sec",
        str(float(async_cfg.get("poll_interval_sec", 5.0))),
    ]
    worker_proc, worker_log_fp = launch_async_eval_worker(cmd=cmd, worker_log_path=worker_log_path)
    return AsyncEvalRuntime(
        enabled=True,
        every_episodes=every_episodes,
        queue_path=queue_path,
        summary_jsonl_path=summary_path,
        worker_log_path=worker_log_path,
        worker_proc=worker_proc,
        worker_log_fp=worker_log_fp,
        eval_checkpoint_dir=eval_checkpoint_dir,
        processed_summary_lines=count_jsonl_lines(summary_path),
    )



def rlt_rollout_metric_aliases(
    *,
    episode_id: int,
    episode_return: float,
    episode_steps: int,
    success: bool,
    recent_success_rate_50: float,
) -> dict[str, Any]:
    return {
        "rollout/episode_id": int(episode_id),
        "rollout/episode_return": float(episode_return),
        "rollout/episode_steps": int(episode_steps),
        "rollout/success": int(success),
        "rollout/recent_success_rate_50": float(recent_success_rate_50),
    }


def rlt_learner_metric_aliases(
    update_info: dict[str, Any],
    *,
    update_steps: int,
    env_steps: int,
    replay_size: int,
) -> dict[str, Any]:
    metric = {
        "learner/update_steps": int(update_steps),
        "learner/env_steps": int(env_steps),
        "learner/replay_size": int(replay_size),
    }
    for key in (
        "loss_critic",
        "target_q_mean",
        "predicted_q_mean",
        "loss_actor",
        "bc_loss",
        "q_value_mean",
    ):
        if key in update_info:
            metric[f"learner/{key}"] = update_info[key]
    return metric


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
