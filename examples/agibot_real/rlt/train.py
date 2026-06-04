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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.agibot_real.common.env import create_agibot_env
from examples.agibot_real.common.io import load_config, make_jsonl_writer, run_dir, write_json
from examples.agibot_real.common.reference_policy import create_reference_policy
from examples.agibot_real.rlt.handover import HandoverClassifier, HandoverLabelCollector, load_handover_classifier, predict_handover_prob
from vla_rl.algorithms.rlt import RLTAgent, RLTokenEncoder
from vla_rl.algorithms.rlt.features import load_frozen_rlt_encoder
from vla_rl.data import ReplayBuffer, Transition
from vla_rl.runtime.agentlace import import_agentlace, json_sanitize, make_agentlace_replay_store, make_trainer_config
from vla_rl.runtime.checkpoint import CheckpointManager
from vla_rl.runtime.wandb import make_wandb_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgiBot RLT actor or learner.")
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
    agent = create_rlt_agent(cfg)
    agentlace = import_agentlace()
    out_dir = run_dir(runtime)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out_dir / "config.yaml")
    metrics = make_jsonl_writer(None if out_dir is None else out_dir / "metrics.jsonl")
    checkpoints = CheckpointManager(out_dir) if out_dir is not None else None
    replay = ReplayBuffer(capacity=int(runtime.replay_capacity), seed=int(runtime.replay_seed))
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
    server.register_data_store(str(runtime.store_name), make_agentlace_replay_store(agentlace, replay))
    server.start(threaded=True)
    server.publish_network(agent.policy_state_dict())

    start = time.perf_counter()
    update_steps = 0
    next_publish = int(runtime.publish_interval_updates)
    try:
        while update_steps < int(runtime.max_update_steps):
            if len(replay) < max(int(runtime.training_starts), int(runtime.batch_size)):
                if actor_done:
                    break
                time.sleep(float(runtime.get("update_sleep_sec", 0.05)))
                continue
            batch = replay.sample(int(runtime.batch_size))
            info = agent.update(batch)
            update_steps += 1
            if update_steps >= next_publish:
                server.publish_network(agent.policy_state_dict())
                next_publish += int(runtime.publish_interval_updates)
            env_steps = max(replay.latest_env_steps, actor_env_steps)
            metric = {
                "role": "learner",
                "update_steps": update_steps,
                "env_steps": env_steps,
                "replay_size": len(replay),
                "wall_time_sec": time.perf_counter() - start,
                **{f"train/{k}": v for k, v in info.items()},
            }
            write(metric, step=update_steps)
            if checkpoints is not None and int(runtime.checkpoint_interval_updates) > 0 and update_steps % int(runtime.checkpoint_interval_updates) == 0:
                checkpoints.save(agent, env_steps=env_steps, update_steps=update_steps, episodes=0, total_reward=0.0, config=OmegaConf.to_container(cfg, resolve=True))
        env_steps = max(replay.latest_env_steps, actor_env_steps)
        summary = {"role": "learner", "update_steps": update_steps, "env_steps": env_steps, "replay_size": len(replay)}
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
    rl_token_encoder = load_rl_token_encoder(cfg)
    agent = create_rlt_agent(cfg)
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

    collector = create_handover_collector(cfg)
    handover_classifier = create_handover_classifier(cfg)
    obs = env.reset()
    env_steps = 0
    critical_env_steps = 0
    episodes = 0
    episode_step = 0
    total_reward = 0.0
    start = time.perf_counter()
    try:
        episode_critical = not bool(cfg.critical_phase.enabled)
        while env_steps < int(runtime.max_env_steps):
            features = reference_policy.extract_features(obs)
            base_actions = np.asarray(features.reference_actions, dtype=np.float32)[: int(runtime.execute_horizon)]
            rlt_obs = build_rlt_obs(
                features,
                rl_token_encoder=rl_token_encoder,
                action_dim=int(cfg.algorithm.action_dim),
                chunk_size=int(cfg.algorithm.chunk_size),
            )
            prefix_mean = mean_prefix_embedding(features)
            handover_now = (not episode_critical) and should_start_critical_phase(
                cfg,
                env,
                handover_classifier=handover_classifier,
                prefix_mean=prefix_mean,
            )
            if collector is not None and not episode_critical:
                collector.submit_hidden(episodes, episode_step, prefix_mean)
            if handover_now:
                episode_critical = True
                if collector is not None:
                    collector.mark_handover(episodes)

            if episode_critical and critical_env_steps >= int(runtime.rlt_warmup_steps):
                action_chunk = agent.sample_action(rlt_obs)
            else:
                action_chunk = base_actions
            action_chunk = np.asarray(action_chunk, dtype=np.float32)[: int(runtime.execute_horizon)]

            next_obs, reward, done, truncated, info = env.step_chunk(action_chunk)
            info = dict(info)
            executed_steps = int(info.get("executed_steps", len(action_chunk)))
            terminal = bool(done or truncated)
            next_rlt_obs = None
            if episode_critical and not terminal:
                next_features = reference_policy.extract_features(next_obs)
                next_rlt_obs = build_rlt_obs(
                    next_features,
                    rl_token_encoder=rl_token_encoder,
                    action_dim=int(cfg.algorithm.action_dim),
                    chunk_size=int(cfg.algorithm.chunk_size),
                )

            if episode_critical:
                mask = np.zeros((int(cfg.algorithm.chunk_size), int(cfg.algorithm.action_dim)), dtype=np.float32)
                mask[:executed_steps] = 1.0
                rlt_obs["action_mask"] = mask.reshape(-1)
                transition = Transition(
                    obs=rlt_obs,
                    next_obs=next_rlt_obs,
                    action=pad_action_chunk(action_chunk, chunk_size=int(cfg.algorithm.chunk_size), action_dim=int(cfg.algorithm.action_dim)).reshape(-1),
                    reward=float(reward),
                    done=bool(done),
                    truncated=bool(truncated),
                    discount=0.0 if terminal else float(runtime.gamma) ** executed_steps,
                    executed_steps=executed_steps,
                    env_steps=critical_env_steps + executed_steps,
                    info={**info, "critical_phase": True, "wall_env_steps": env_steps + executed_steps},
                )
                data_store.insert(transition.to_payload())
                client.update()
                critical_env_steps += executed_steps

            env_steps += executed_steps
            episode_step += executed_steps
            total_reward += float(reward)
            actor_metrics({
                "role": "actor",
                "env_steps": env_steps,
                "critical_env_steps": critical_env_steps,
                "episode": episodes,
                "critical_phase": bool(episode_critical),
                "reward": float(reward),
                "done": bool(done),
                "truncated": bool(truncated),
                "wall_time_sec": time.perf_counter() - start,
            })
            if terminal:
                if collector is not None:
                    collector.end_episode(episodes)
                episodes += 1
                episode_step = 0
                episode_critical = not bool(cfg.critical_phase.enabled)
                obs = env.reset()
            else:
                obs = next_obs
            if int(runtime.weight_update_interval_steps) > 0 and env_steps % int(runtime.weight_update_interval_steps) == 0:
                client.update()

        summary = {
            "event": "actor_summary",
            "role": "actor",
            "env_steps": env_steps,
            "critical_env_steps": critical_env_steps,
            "episodes": episodes,
            "total_reward": total_reward,
            "received_policy_state": has_policy_state,
        }
        client.request(str(runtime.request_type), summary)
        if out_dir is not None:
            write_json(out_dir / "actor_summary.json", summary)
        return summary
    finally:
        if collector is not None:
            collector.close()
        client.stop()
        env.close()
        reference_policy.close()


def build_rlt_obs(
    features,
    *,
    rl_token_encoder: RLTokenEncoder,
    action_dim: int,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    ref = np.asarray(features.reference_actions, dtype=np.float32)
    if ref.shape[0] < chunk_size:
        raise ValueError(f"reference policy returned {ref.shape[0]} actions, need {chunk_size}")
    prefix_tokens = np.asarray(features.embeddings["prefix"], dtype=np.float32)
    z_vla = torch.as_tensor(prefix_tokens, dtype=torch.float32, device=next(rl_token_encoder.parameters()).device)
    if z_vla.dim() == 2:
        z_vla = z_vla.unsqueeze(0)
    if rl_token_encoder.max_tokens is not None:
        z_vla = z_vla[:, : int(rl_token_encoder.max_tokens), :]
    with torch.no_grad():
        z_rl = rl_token_encoder(z_vla).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return {
        "z_rl": z_rl,
        "reference_action": ref[:chunk_size, :action_dim].reshape(-1),
        "proprio": np.asarray(features.proprio, dtype=np.float32).reshape(-1),
    }


def pad_action_chunk(actions: np.ndarray, *, chunk_size: int, action_dim: int) -> np.ndarray:
    result = np.zeros((chunk_size, action_dim), dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, action_dim)
    result[: min(chunk_size, actions.shape[0])] = actions[:chunk_size]
    return result


def mean_prefix_embedding(features) -> np.ndarray:
    prefix = np.asarray(features.embeddings["prefix"], dtype=np.float32)
    if prefix.ndim == 3:
        prefix = prefix[0]
    return prefix.mean(axis=0).astype(np.float32)


def should_start_critical_phase(
    cfg,
    env,
    *,
    handover_classifier: HandoverClassifier | None,
    prefix_mean: np.ndarray,
) -> bool:
    critical_cfg = cfg.get("critical_phase", {})
    if not bool(critical_cfg.get("enabled", True)):
        return True
    meta = env.controller_meta()
    if bool(meta.get("critical_phase_active", False)):
        return True
    if handover_classifier is None:
        return False
    prob = predict_handover_prob(handover_classifier, torch.as_tensor(prefix_mean, dtype=torch.float32))
    return prob >= float(critical_cfg.handover_threshold)


def create_handover_collector(cfg) -> HandoverLabelCollector | None:
    critical_cfg = cfg.get("critical_phase", {})
    output = critical_cfg.get("handover_labels_path", None)
    if output is None:
        return None
    return HandoverLabelCollector(output, lookahead=int(critical_cfg.handover_label_lookahead))


def create_handover_classifier(cfg) -> HandoverClassifier | None:
    path = cfg.critical_phase.get("classifier_path", None)
    if path is None:
        return None
    return load_handover_classifier(path, device=str(cfg.critical_phase.classifier_device))


def load_rl_token_encoder(cfg: DictConfig) -> RLTokenEncoder:
    feature_cfg = cfg.feature
    encoder = load_frozen_rlt_encoder(
        str(feature_cfg.encoder_path),
        device=str(feature_cfg.device),
        input_dim=int(feature_cfg.input_dim),
        rl_token_dim=int(feature_cfg.rl_token_dim),
        num_encoder_layers=int(feature_cfg.num_encoder_layers),
        num_heads=int(feature_cfg.num_heads),
        ff_dim=int(feature_cfg.ff_dim),
        dropout=float(feature_cfg.dropout),
        max_tokens=None if feature_cfg.max_tokens is None else int(feature_cfg.max_tokens),
    )
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def create_rlt_agent(cfg: DictConfig) -> RLTAgent:
    kwargs = OmegaConf.to_container(cfg.algorithm, resolve=True)
    return RLTAgent(**kwargs)


def validate_cfg(cfg: DictConfig) -> None:
    if int(cfg.runtime.execute_horizon) > int(cfg.algorithm.chunk_size):
        raise ValueError("runtime.execute_horizon must be <= algorithm.chunk_size")
    if int(cfg.env.action_dim) != int(cfg.algorithm.action_dim):
        raise ValueError("AgiBot RLT currently expects env.action_dim == algorithm.action_dim")


if __name__ == "__main__":
    main()
