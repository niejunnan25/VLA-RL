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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vla_rl.algorithms.pld import load_pld_offline_replay
from vla_rl.config import instantiate
from vla_rl.data import ActionChunk, CompactReplayBuffer, CompactTransition, MixedReplaySampler
from vla_rl.runtime.agentlace import import_agentlace, json_sanitize, make_agentlace_replay_store, make_trainer_config
from vla_rl.runtime.checkpoint import CheckpointManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIBERO PLD actor or learner.")
    parser.add_argument("--config", required=True, help="Path to a PLD YAML config.")
    parser.add_argument("--role", required=True, choices=("actor", "learner"))
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="OmegaConf dotlist overrides after --.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config, args.overrides)
    cfg.runtime.role = args.role
    _validate_pld_cfg(cfg)
    summary = run_learner(cfg) if args.role == "learner" else run_actor(cfg)
    print(json.dumps(json_sanitize(summary), sort_keys=True))


def run_learner(cfg: DictConfig) -> dict[str, Any]:
    runtime = cfg.runtime
    algorithm = instantiate(cfg.algorithm)
    agentlace = import_agentlace()
    run_dir = _run_dir(runtime)
    checkpoints = CheckpointManager(run_dir) if run_dir is not None else None
    replay = CompactReplayBuffer(capacity=int(runtime.replay_capacity), seed=int(runtime.replay_seed))
    config_snapshot = OmegaConf.to_container(cfg, resolve=True)
    offline_replay, offline_stats = _load_offline_replay(runtime)

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
        payload = checkpoints.load(runtime.resume_from, algorithm) if checkpoints is not None else {}
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
            return
        with (run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(json_sanitize(metric), sort_keys=True) + "\n")

    def apply_actor_summary_file() -> None:
        nonlocal actor_done, actor_done_env_steps
        actor_done, actor_done_env_steps = _apply_actor_summary_file(run_dir, actor_done, actor_done_env_steps)

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

    server = agentlace.TrainerServer(
        make_trainer_config(agentlace, int(runtime.trainer_port), int(runtime.broadcast_port), [str(runtime.request_type)]),
        request_callback=request_callback,
    )
    server.register_data_store(str(runtime.store_name), make_agentlace_replay_store(agentlace, replay))
    server.start(threaded=True)
    sampler = MixedReplaySampler(
        replay,
        offline_replay,
        offline_ratio=0.0 if offline_replay is None else float(runtime.offline_ratio),
    )
    start_time = time.perf_counter()
    last_wait_metric_time = 0.0
    last_wait_publish_time = time.perf_counter()
    last_update: dict[str, Any] = {}
    calql_steps_done = 0

    try:
        while calql_steps_done < int(runtime.calql_pretrain_steps) and update_steps < int(runtime.max_update_steps):
            if offline_replay is None or len(offline_replay) == 0:
                break
            last_update = algorithm.update_critics_calql(
                offline_replay.sample(int(runtime.batch_size)),
                calql_alpha=float(runtime.calql_alpha),
                calql_n_actions=int(runtime.calql_n_actions),
                calql_temperature=float(runtime.calql_temperature),
            )
            update_steps += 1
            calql_steps_done += 1
            write_metric({
                "role": "learner",
                "phase": "calql_pretrain",
                "update_steps": update_steps,
                "calql_pretrain_steps": calql_steps_done,
                "offline_replay_size": 0 if offline_replay is None else len(offline_replay),
                **{f"train/{key}": value for key, value in last_update.items()},
            })

        server.publish_network(algorithm.policy_state_dict())
        next_publish_at = _next_interval(update_steps, int(runtime.publish_interval_updates))
        next_ckpt_env_at = _next_interval(env_steps, int(runtime.checkpoint_interval_env_steps))
        next_ckpt_update_at = _next_interval(update_steps, int(runtime.checkpoint_interval_updates))

        while True:
            apply_actor_summary_file()
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
                    server.publish_network(algorithm.policy_state_dict())
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
                time.sleep(_runtime_float(runtime, "update_sleep_sec", 0.05))
                continue
            if update_steps >= int(runtime.max_update_steps):
                time.sleep(_runtime_float(runtime, "update_sleep_sec", 0.05))
                continue
            step_start = time.perf_counter()
            mixed = sampler.sample(int(runtime.batch_size))
            last_update = algorithm.update(mixed.batch)
            update_steps += 1
            env_steps = current_env_steps(env_steps)
            write_metric({
                "role": "learner",
                "phase": "online",
                "env_steps": env_steps,
                "update_steps": update_steps,
                "online_update_steps": max(0, update_steps - calql_steps_done),
                "calql_pretrain_steps": calql_steps_done,
                "replay_size": len(replay),
                "offline_replay_size": 0 if offline_replay is None else len(offline_replay),
                "batch_mix": mixed.mix,
                "update_time_sec": time.perf_counter() - step_start,
                "wall_time_sec": time.perf_counter() - start_time,
                **{f"train/{key}": value for key, value in last_update.items()},
            })
            if int(runtime.publish_interval_updates) > 0 and update_steps >= next_publish_at:
                server.publish_network(algorithm.policy_state_dict())
                next_publish_at += int(runtime.publish_interval_updates)
            if checkpoints is not None and int(runtime.checkpoint_interval_env_steps) > 0 and env_steps >= next_ckpt_env_at:
                _save_checkpoint(checkpoints, algorithm, env_steps, update_steps, episodes, total_reward, config_snapshot)
                next_ckpt_env_at += int(runtime.checkpoint_interval_env_steps)
            if checkpoints is not None and int(runtime.checkpoint_interval_updates) > 0 and update_steps >= next_ckpt_update_at:
                _save_checkpoint(checkpoints, algorithm, env_steps, update_steps, episodes, total_reward, config_snapshot, tag=f"update_{update_steps}.pt")
                next_ckpt_update_at += int(runtime.checkpoint_interval_updates)
    finally:
        stop = getattr(server, "stop", None)
        if callable(stop):
            stop()

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
    if checkpoints is not None:
        _save_checkpoint(checkpoints, algorithm, env_steps, update_steps, episodes, total_reward, config_snapshot, tag="final.pt")
    if run_dir is not None:
        (run_dir / "summary.json").write_text(json.dumps(json_sanitize(summary), indent=2, sort_keys=True) + "\n")
    write_metric({"summary": summary})
    return summary


def run_actor(cfg: DictConfig) -> dict[str, Any]:
    runtime = cfg.runtime
    env = instantiate(cfg.env)
    policy = instantiate(cfg.policy)
    feature_processor = instantiate(cfg.feature)
    algorithm = instantiate(cfg.algorithm)
    agentlace = import_agentlace()
    run_dir = _run_dir(runtime)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "actor_metrics.jsonl").write_text("")

    def write_actor_metric(metric: dict[str, Any]) -> None:
        if run_dir is None:
            return
        with (run_dir / "actor_metrics.jsonl").open("a") as f:
            f.write(json.dumps(json_sanitize(metric), sort_keys=True) + "\n")

    data_store = agentlace.QueuedDataStore(int(runtime.actor_queue_capacity))
    client = agentlace.TrainerClient(
        str(runtime.actor_name),
        str(runtime.trainer_ip),
        make_trainer_config(agentlace, int(runtime.trainer_port), int(runtime.broadcast_port), [str(runtime.request_type)]),
        data_store,
        wait_for_server=True,
    )
    has_policy_state = False

    def network_callback(payload: dict[str, Any]) -> None:
        nonlocal has_policy_state
        if payload is not None:
            algorithm.load_policy_state_dict(payload)
            has_policy_state = True

    client.recv_network_callback(network_callback)
    _wait_for_initial_weights(
        client,
        has_policy_state_fn=lambda: has_policy_state,
        timeout_sec=_runtime_float(runtime, "initial_weight_timeout_sec", 300.0),
        sleep_sec=_runtime_float(runtime, "client_update_sleep_sec", 0.1),
    )

    try:
        obs = env.reset()
        env_steps = 0
        episodes = 0
        successes = 0
        total_reward = 0.0
        next_weight_update_at = _next_interval(env_steps, int(runtime.weight_update_interval_steps))
        next_stats_at = _next_interval(env_steps, int(runtime.stats_interval_env_steps))
        start_time = time.perf_counter()
        episode_success = False
        while env_steps < int(runtime.max_env_steps):
            chunk_start = time.perf_counter()
            features = policy.extract_features(obs)
            reference = _reference_chunk_from_features(features)
            agent_obs = feature_processor.process(obs, features)
            base_warmup = episodes < int(runtime.base_warmup_episodes)
            action_chunk = reference if base_warmup else algorithm.act(obs, features=features, agent_obs=agent_obs)
            action_chunk.validate()
            if action_chunk.actions.shape[0] < int(runtime.execute_horizon):
                raise ValueError(
                    f"action chunk length {action_chunk.actions.shape[0]} is shorter than execute_horizon={int(runtime.execute_horizon)}"
                )
            execute_actions = action_chunk.actions[: int(runtime.execute_horizon)]
            next_obs, reward, done, truncated, info = env.step_chunk(execute_actions)
            executed_steps = int(info.get("executed_steps", len(execute_actions)))
            env_steps += executed_steps
            total_reward += float(reward)
            terminal = bool(done or truncated)
            episode_success = bool(episode_success or info.get("success", False) or info.get("env_done", False))
            next_agent_obs = None
            if not terminal:
                next_features = policy.extract_features(next_obs)
                next_agent_obs = feature_processor.process(next_obs, next_features)
            transition = CompactTransition(
                agent_obs=agent_obs,
                next_agent_obs=next_agent_obs,
                action=np.asarray(execute_actions, dtype=np.float32).reshape(-1),
                reward=float(reward),
                done=bool(done),
                truncated=bool(truncated),
                discount=0.0 if terminal else float(runtime.gamma) ** executed_steps,
                executed_steps=executed_steps,
                env_steps=env_steps,
                info={**info, "base_warmup": bool(base_warmup)},
            )
            data_store.insert(transition.to_payload())
            client.update()
            metric = {
                "role": "actor",
                "algorithm": "pld",
                "env_steps": env_steps,
                "episode": episodes,
                "reward": float(reward),
                "done": bool(done),
                "truncated": bool(truncated),
                "executed_steps": executed_steps,
                "base_warmup": bool(base_warmup),
                "wall_time_sec": time.perf_counter() - start_time,
                "chunk_time_sec": time.perf_counter() - chunk_start,
            }
            write_actor_metric(metric)
            if int(runtime.stats_interval_env_steps) > 0 and env_steps >= next_stats_at:
                client.request(str(runtime.request_type), metric)
                next_stats_at += int(runtime.stats_interval_env_steps)
            if int(runtime.weight_update_interval_steps) > 0 and env_steps >= next_weight_update_at:
                client.update()
                next_weight_update_at += int(runtime.weight_update_interval_steps)
            if terminal:
                episodes += 1
                successes += int(episode_success)
                episode_success = False
                obs = env.reset()
            else:
                obs = next_obs
        summary = {
            "role": "actor",
            "algorithm": "pld",
            "env_steps": env_steps,
            "episodes": episodes,
            "successes": successes,
            "total_reward": total_reward,
            "received_policy_state": has_policy_state,
        }
        summary = _send_actor_summary(client, str(runtime.request_type), summary, run_dir, write_actor_metric)
        write_actor_metric({"summary": summary})
        return summary
    finally:
        stop = getattr(client, "stop", None)
        if callable(stop):
            stop()
        close = getattr(env, "close", None)
        if callable(close):
            close()
        close = getattr(policy, "close", None)
        if callable(close):
            close()


def _load_config(path: str, overrides: list[str]) -> DictConfig:
    dotlist = list(overrides)
    if dotlist and dotlist[0] == "--":
        dotlist = dotlist[1:]
    cfg = OmegaConf.load(Path(path))
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


def _validate_pld_cfg(cfg: DictConfig) -> None:
    if int(cfg.algorithm.chunk_horizon) != int(cfg.runtime.execute_horizon):
        raise ValueError("algorithm.chunk_horizon must match runtime.execute_horizon")
    if int(cfg.runtime.execute_horizon) <= 0:
        raise ValueError("runtime.execute_horizon must be positive")


def _load_offline_replay(runtime: DictConfig) -> tuple[CompactReplayBuffer | None, dict[str, Any]]:
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


def _reference_chunk_from_features(features) -> ActionChunk:
    if features.reference_actions is None:
        raise ValueError("PLD actor requires PolicyFeatures.reference_actions")
    reference = ActionChunk(
        actions=np.asarray(features.reference_actions, dtype=np.float32),
        horizon=int(features.reference_actions.shape[0]),
        metadata={"source": "policy_features.reference_actions"},
    )
    reference.validate()
    return reference


def _runtime_float(runtime: DictConfig, key: str, default: float) -> float:
    return float(runtime.get(key, default))


def _run_dir(runtime: DictConfig) -> Path | None:
    value = runtime.get("run_dir", None)
    return Path(value) if value else None



def _write_actor_summary(run_dir: Path | None, summary: dict[str, Any]) -> None:
    if run_dir is None:
        return
    (run_dir / "actor_summary.json").write_text(json.dumps(json_sanitize(summary), indent=2, sort_keys=True) + "\n")


def _read_actor_summary(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    path = run_dir / "actor_summary.json"
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _apply_actor_summary_file(run_dir: Path | None, actor_done: bool, actor_done_env_steps: int) -> tuple[bool, int]:
    summary = _read_actor_summary(run_dir)
    if summary is None:
        return actor_done, actor_done_env_steps
    return True, max(int(actor_done_env_steps), int(summary.get("env_steps", 0)))


def _send_actor_summary(client: Any, request_type: str, summary: dict[str, Any], run_dir: Path | None, write_metric) -> dict[str, Any]:
    final_summary = dict(summary)
    final_summary["actor_summary_notified"] = False
    _write_actor_summary(run_dir, final_summary)
    try:
        client.request(str(request_type), {"event": "actor_summary", **final_summary})
    except Exception as exc:
        metric = {
            "role": final_summary.get("role", "actor"),
            "event": "actor_summary_send_failed",
            "env_steps": int(final_summary.get("env_steps", 0)),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if "algorithm" in final_summary:
            metric["algorithm"] = final_summary["algorithm"]
        write_metric(metric)
    else:
        final_summary["actor_summary_notified"] = True
        metric = {
            "role": final_summary.get("role", "actor"),
            "event": "actor_summary_sent",
            "env_steps": int(final_summary.get("env_steps", 0)),
        }
        if "algorithm" in final_summary:
            metric["algorithm"] = final_summary["algorithm"]
        write_metric(metric)
    _write_actor_summary(run_dir, final_summary)
    return final_summary

def _save_checkpoint(
    checkpoints: CheckpointManager,
    algorithm,
    env_steps: int,
    update_steps: int,
    episodes: int,
    total_reward: float,
    config: dict[str, Any],
    tag: str | None = None,
) -> None:
    checkpoints.save(
        algorithm,
        env_steps=env_steps,
        update_steps=update_steps,
        episodes=episodes,
        total_reward=total_reward,
        config=config,
        tag=tag,
    )


def _wait_for_initial_weights(client: Any, *, has_policy_state_fn, timeout_sec: float, sleep_sec: float) -> None:
    deadline = time.perf_counter() + timeout_sec
    while not has_policy_state_fn():
        client.update()
        if time.perf_counter() > deadline:
            raise TimeoutError("timed out waiting for initial learner policy weights")
        time.sleep(sleep_sec)


def _learner_should_stop(update_steps: int, env_steps: int, max_update_steps: int, max_env_steps: int, actor_done: bool) -> bool:
    if update_steps < max_update_steps:
        return False
    if max_env_steps <= 0:
        return True
    return env_steps >= max_env_steps or actor_done


def _next_interval(current: int, interval: int) -> int:
    if interval <= 0:
        return 0
    return ((int(current) // int(interval)) + 1) * int(interval)


if __name__ == "__main__":
    main()
