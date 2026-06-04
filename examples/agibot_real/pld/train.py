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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.agibot_real.common.env import create_agibot_env
from examples.agibot_real.common.io import load_config, make_jsonl_writer, run_dir, write_json
from examples.agibot_real.common.reference_policy import create_reference_policy
from vla_rl.algorithms.pld import PLDObservationBuilder, PLDSACAgent, build_pld_obs, load_pld_offline_replay
from vla_rl.data import MixedReplaySampler, ReplayBuffer, Transition
from vla_rl.runtime.agentlace import import_agentlace, json_sanitize, make_agentlace_replay_store, make_trainer_config
from vla_rl.runtime.checkpoint import CheckpointManager
from vla_rl.runtime.wandb import make_wandb_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgiBot PLD/residual actor or learner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", required=True, choices=("actor", "learner"))
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    cfg.runtime.role = args.role
    validate_cfg(cfg)
    summary = run_learner(cfg) if args.role == "learner" else run_actor(cfg)
    print(json.dumps(json_sanitize(summary), sort_keys=True))


def run_learner(cfg: DictConfig) -> dict[str, Any]:
    runtime = cfg.runtime
    agent = create_pld_agent(cfg)
    agentlace = import_agentlace()
    out_dir = run_dir(runtime)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out_dir / "config.yaml")
    metrics = make_jsonl_writer(None if out_dir is None else out_dir / "metrics.jsonl")
    checkpoints = CheckpointManager(out_dir) if out_dir is not None else None
    online = ReplayBuffer(capacity=int(runtime.replay_capacity), seed=int(runtime.replay_seed))
    offline, offline_stats = load_optional_offline(runtime)
    sampler = MixedReplaySampler(online, offline, offline_ratio=0.0 if offline is None else float(runtime.offline_ratio))
    wandb = make_wandb_logger(cfg.get("wandb", None), variant=OmegaConf.to_container(cfg, resolve=True), run_dir=out_dir)
    actor_done = False
    actor_env_steps = 0

    def write(metric: dict[str, Any], step: int) -> None:
        metrics(metric)
        wandb.log(metric, step=step)

    def request_callback(request_type: str, payload: Any) -> Any:
        nonlocal actor_done, actor_env_steps
        if request_type != str(runtime.request_type):
            return {"ok": False, "error": request_type}
        stat = dict(payload or {})
        if stat.get("event") == "actor_summary":
            actor_done = True
            actor_env_steps = max(actor_env_steps, int(stat.get("env_steps", 0)))
        write({"role": "actor", **stat}, step=agent.update_count)
        return {"ok": True}

    server = agentlace.TrainerServer(
        make_trainer_config(agentlace, int(runtime.trainer_port), int(runtime.broadcast_port), [str(runtime.request_type)]),
        request_callback=request_callback,
    )
    server.register_data_store(str(runtime.store_name), make_agentlace_replay_store(agentlace, online))
    server.start(threaded=True)
    write({"role": "learner", "event": "offline_replay_loaded", "offline_stats": offline_stats}, step=0)
    calql_done = 0
    while calql_done < int(runtime.calql_pretrain_steps) and offline is not None and len(offline) > 0:
        info = agent.update_critics_calql(offline.sample(int(runtime.batch_size)), calql_alpha=float(runtime.calql_alpha))
        calql_done += 1
        write({"role": "learner", "phase": "calql_pretrain", "calql_steps": calql_done, **{f"train/{k}": v for k, v in info.items()}}, step=calql_done)
    server.publish_network(agent.policy_state_dict())

    update_steps = calql_done
    start = time.perf_counter()
    try:
        while update_steps < int(runtime.max_update_steps):
            if len(online) < max(int(runtime.training_starts), 1):
                if actor_done:
                    break
                time.sleep(float(runtime.get("update_sleep_sec", 0.05)))
                continue
            mixed = sampler.sample(int(runtime.batch_size))
            info = agent.update(mixed.batch)
            update_steps += 1
            if int(runtime.publish_interval_updates) > 0 and update_steps % int(runtime.publish_interval_updates) == 0:
                server.publish_network(agent.policy_state_dict())
            env_steps = max(online.latest_env_steps, actor_env_steps)
            write(
                {
                    "role": "learner",
                    "update_steps": update_steps,
                    "env_steps": env_steps,
                    "replay_size": len(online),
                    "offline_replay_size": 0 if offline is None else len(offline),
                    "wall_time_sec": time.perf_counter() - start,
                    **{f"train/{k}": v for k, v in info.items()},
                },
                step=update_steps,
            )
            if checkpoints is not None and int(runtime.checkpoint_interval_updates) > 0 and update_steps % int(runtime.checkpoint_interval_updates) == 0:
                checkpoints.save(agent, env_steps=env_steps, update_steps=update_steps, episodes=0, total_reward=0.0, config=OmegaConf.to_container(cfg, resolve=True))
        env_steps = max(online.latest_env_steps, actor_env_steps)
        summary = {"role": "learner", "update_steps": update_steps, "env_steps": env_steps, "replay_size": len(online)}
        if checkpoints is not None:
            checkpoints.save(agent, env_steps=env_steps, update_steps=update_steps, episodes=0, total_reward=0.0, config=OmegaConf.to_container(cfg, resolve=True), tag="final.pt")
            write_json(out_dir / "summary.json", summary)
        return summary
    finally:
        server.stop()
        wandb.finish()


def run_actor(cfg: DictConfig) -> dict[str, Any]:
    runtime = cfg.runtime
    env = create_agibot_env(cfg)
    reference_policy = create_reference_policy(cfg)
    obs_builder = create_pld_obs_builder(cfg)
    agent = create_pld_agent(cfg)
    agentlace = import_agentlace()
    out_dir = run_dir(runtime)
    actor_metrics = make_jsonl_writer(None if out_dir is None else out_dir / "actor_metrics.jsonl")
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
        agent.load_policy_state_dict(payload)
        has_policy_state = True

    client.recv_network_callback(network_callback)
    while not has_policy_state:
        client.update()
        time.sleep(float(runtime.get("client_update_sleep_sec", 0.1)))

    obs = env.reset()
    env_steps = 0
    episodes = 0
    total_reward = 0.0
    start = time.perf_counter()
    try:
        while env_steps < int(runtime.max_env_steps):
            base_actions = predict_base_actions(reference_policy, obs, horizon=int(runtime.execute_horizon))
            pld_obs = build_pld_obs(obs, base_actions, builder=obs_builder)
            if env_steps < int(runtime.base_warmup_steps):
                final_actions = base_actions
            else:
                final_actions = agent.sample_action(pld_obs)
            final_actions = np.asarray(final_actions, dtype=np.float32)[: int(runtime.execute_horizon)]
            next_obs, reward, done, truncated, info = env.step_chunk(final_actions)
            info = dict(info)
            executed_steps = int(info.get("executed_steps", len(final_actions)))
            terminal = bool(done or truncated)
            next_pld_obs = None
            if not terminal:
                next_base_actions = predict_base_actions(reference_policy, next_obs, horizon=int(runtime.execute_horizon))
                next_pld_obs = build_pld_obs(next_obs, next_base_actions, builder=obs_builder)
            transition = Transition(
                obs=pld_obs,
                next_obs=next_pld_obs,
                action=np.asarray(final_actions, dtype=np.float32).reshape(-1),
                reward=float(reward),
                done=bool(done),
                truncated=bool(truncated),
                discount=0.0 if terminal else float(runtime.gamma) ** executed_steps,
                executed_steps=executed_steps,
                env_steps=env_steps + executed_steps,
                info=info,
            )
            data_store.insert(transition.to_payload())
            client.update()
            env_steps += executed_steps
            total_reward += float(reward)
            actor_metrics({"role": "actor", "env_steps": env_steps, "episode": episodes, "reward": float(reward), "done": bool(done), "truncated": bool(truncated), "wall_time_sec": time.perf_counter() - start})
            if terminal:
                episodes += 1
                obs = env.reset()
            else:
                obs = next_obs
            if int(runtime.weight_update_interval_steps) > 0 and env_steps % int(runtime.weight_update_interval_steps) == 0:
                client.update()
        summary = {"event": "actor_summary", "role": "actor", "env_steps": env_steps, "episodes": episodes, "total_reward": total_reward, "received_policy_state": has_policy_state}
        client.request(str(runtime.request_type), summary)
        if out_dir is not None:
            write_json(out_dir / "actor_summary.json", summary)
        return summary
    finally:
        client.stop()
        env.close()
        reference_policy.close()


def predict_base_actions(reference_policy, obs, *, horizon: int) -> np.ndarray:
    return np.asarray(reference_policy.sample_actions(obs), dtype=np.float32)[:horizon]


def create_pld_obs_builder(cfg: DictConfig) -> PLDObservationBuilder:
    return PLDObservationBuilder(**OmegaConf.to_container(cfg.pld_observation, resolve=True))


def create_pld_agent(cfg: DictConfig) -> PLDSACAgent:
    return PLDSACAgent(**OmegaConf.to_container(cfg.algorithm, resolve=True))


def load_optional_offline(runtime) -> tuple[ReplayBuffer | None, dict[str, Any]]:
    path = runtime.get("offline_replay_path", None)
    if path is None or str(path) == "":
        return None, {"enabled": False}
    replay, stats = load_pld_offline_replay(path, capacity=int(runtime.offline_capacity), seed=int(runtime.replay_seed))
    return replay, stats


def validate_cfg(cfg: DictConfig) -> None:
    if int(cfg.runtime.execute_horizon) != int(cfg.algorithm.chunk_horizon):
        raise ValueError("runtime.execute_horizon must match algorithm.chunk_horizon")
    if int(cfg.env.action_dim) != int(cfg.algorithm.action_dim):
        raise ValueError("env.action_dim must match algorithm.action_dim")


if __name__ == "__main__":
    main()
