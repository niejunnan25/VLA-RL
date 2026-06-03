from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from vla_rl.algorithms import Algorithm
from vla_rl.data import ActionChunk, CompactReplayBuffer, CompactTransition
from vla_rl.envs import EnvBackend
from vla_rl.features import FeatureProcessor
from vla_rl.policies import PolicyBackend
from vla_rl.runtime.base import Runner
from vla_rl.runtime.checkpoint import CheckpointManager


class AgentlaceLearnerRuntime(Runner):
    """Agentlace learner for RLT-style compact actor transitions."""

    def __init__(
        self,
        algorithm: Algorithm,
        max_update_steps: int = 1000,
        batch_size: int = 1,
        replay_capacity: int = 100_000,
        replay_seed: int = 0,
        training_starts: int = 1,
        trainer_port: int = 5488,
        broadcast_port: int = 5489,
        publish_interval_updates: int = 100,
        run_dir: str | None = None,
        checkpoint_interval_env_steps: int = 0,
        checkpoint_interval_updates: int = 0,
        resume_from: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
        update_sleep_sec: float = 0.05,
        store_name: str = "actor_env",
        request_type: str = "send-stats",
    ) -> None:
        self.algorithm = algorithm
        self.max_update_steps = int(max_update_steps)
        self.batch_size = int(batch_size)
        self.training_starts = int(training_starts)
        self.trainer_port = int(trainer_port)
        self.broadcast_port = int(broadcast_port)
        self.publish_interval_updates = int(publish_interval_updates)
        self.run_dir = Path(run_dir) if run_dir else None
        self.checkpoints = CheckpointManager(self.run_dir) if self.run_dir else None
        self.checkpoint_interval_env_steps = int(checkpoint_interval_env_steps)
        self.checkpoint_interval_updates = int(checkpoint_interval_updates)
        self.resume_from = resume_from
        self.config_snapshot = config_snapshot or {}
        self.update_sleep_sec = float(update_sleep_sec)
        self.store_name = str(store_name)
        self.request_type = str(request_type)
        self.replay = CompactReplayBuffer(capacity=replay_capacity, seed=replay_seed)
        self._actor_stats: list[dict[str, Any]] = []

    def run(self) -> dict:
        agentlace = _import_agentlace()
        self._prepare_outputs()
        update_steps = 0
        env_steps = 0
        episodes = 0
        total_reward = 0.0
        if self.resume_from:
            payload = self._load_checkpoint(self.resume_from)
            update_steps = int(payload.get("update_steps", 0))
            env_steps = int(payload.get("env_steps", 0))
            episodes = int(payload.get("episodes", 0))
            total_reward = float(payload.get("total_reward", 0.0))

        server = agentlace.TrainerServer(
            _make_trainer_config(agentlace, self.trainer_port, self.broadcast_port, [self.request_type]),
            request_callback=self._request_callback,
        )
        store = _make_agentlace_replay_store(agentlace, self.replay)
        server.register_data_store(self.store_name, store)
        server.start(threaded=True)
        server.publish_network(self.algorithm.policy_state_dict())

        next_publish_at = self._next_interval(update_steps, self.publish_interval_updates)
        next_ckpt_env_at = self._next_interval(env_steps, self.checkpoint_interval_env_steps)
        next_ckpt_update_at = self._next_interval(update_steps, self.checkpoint_interval_updates)
        start_time = time.perf_counter()
        last_wait_metric_time = 0.0
        last_wait_publish_time = time.perf_counter()
        last_update: dict[str, Any] = {}
        try:
            while update_steps < self.max_update_steps:
                if len(self.replay) < max(self.batch_size, self.training_starts):
                    now = time.perf_counter()
                    if now - last_wait_publish_time >= 1.0:
                        server.publish_network(self.algorithm.policy_state_dict())
                        last_wait_publish_time = now
                    if now - last_wait_metric_time >= 1.0:
                        self._write_metric(
                            {
                                "role": "learner",
                                "event": "waiting_for_replay",
                                "replay_size": len(self.replay),
                                "training_starts": self.training_starts,
                                "update_steps": update_steps,
                                "env_steps": max(env_steps, self.replay.latest_env_steps),
                                "wall_time_sec": now - start_time,
                            }
                        )
                        last_wait_metric_time = now
                    time.sleep(self.update_sleep_sec)
                    continue

                step_start = time.perf_counter()
                last_update = self.algorithm.update(self.replay.sample(self.batch_size))
                update_steps += 1
                env_steps = max(env_steps, self.replay.latest_env_steps)
                metric = {
                    "role": "learner",
                    "env_steps": env_steps,
                    "update_steps": update_steps,
                    "replay_size": len(self.replay),
                    "update_time_sec": time.perf_counter() - step_start,
                    "wall_time_sec": time.perf_counter() - start_time,
                    **{f"train/{key}": value for key, value in last_update.items()},
                }
                self._write_metric(metric)

                if self.publish_interval_updates > 0 and update_steps >= next_publish_at:
                    server.publish_network(self.algorithm.policy_state_dict())
                    next_publish_at += self.publish_interval_updates

                if (
                    self.checkpoints is not None
                    and self.checkpoint_interval_env_steps > 0
                    and env_steps >= next_ckpt_env_at
                ):
                    self._save_checkpoint(env_steps, update_steps, episodes, total_reward)
                    next_ckpt_env_at += self.checkpoint_interval_env_steps
                if (
                    self.checkpoints is not None
                    and self.checkpoint_interval_updates > 0
                    and update_steps >= next_ckpt_update_at
                ):
                    self._save_checkpoint(env_steps, update_steps, episodes, total_reward, tag=f"update_{update_steps}.pt")
                    next_ckpt_update_at += self.checkpoint_interval_updates
        finally:
            stop = getattr(server, "stop", None)
            if callable(stop):
                stop()

        summary = {
            "role": "learner",
            "env_steps": env_steps,
            "update_steps": update_steps,
            "replay_size": len(self.replay),
            "last_algorithm_updates": last_update.get("updates", 0),
        }
        if self.checkpoints is not None:
            self._save_checkpoint(env_steps, update_steps, episodes, total_reward, tag="final.pt")
        self._write_summary(summary)
        self._write_metric({"summary": summary})
        return summary

    def _request_callback(self, request_type: str, payload: Any) -> Any:
        if request_type == self.request_type:
            stat = dict(payload or {})
            stat.setdefault("role", "actor")
            self._actor_stats.append(stat)
            self._write_metric(stat)
            return {"ok": True}
        return {"ok": False, "error": f"unsupported request type: {request_type}"}

    def _prepare_outputs(self) -> None:
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(self.config_snapshot), self.run_dir / "config.yaml")
        if not self.resume_from:
            (self.run_dir / "metrics.jsonl").write_text("")
        else:
            (self.run_dir / "metrics.jsonl").touch(exist_ok=True)

    def _write_metric(self, metric: dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(_json_sanitize(metric), sort_keys=True) + "\n")

    def _write_summary(self, summary: dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        (self.run_dir / "summary.json").write_text(json.dumps(_json_sanitize(summary), indent=2, sort_keys=True) + "\n")

    def _save_checkpoint(
        self,
        env_steps: int,
        update_steps: int,
        episodes: int,
        total_reward: float,
        tag: str | None = None,
    ) -> None:
        if self.checkpoints is None:
            return
        self.checkpoints.save(
            self.algorithm,
            env_steps=env_steps,
            update_steps=update_steps,
            episodes=episodes,
            total_reward=total_reward,
            config=self.config_snapshot,
            tag=tag,
        )

    def _load_checkpoint(self, path: str) -> dict:
        if self.checkpoints is None:
            self.checkpoints = CheckpointManager(Path(path).parent.parent)
        return self.checkpoints.load(path, self.algorithm)

    @staticmethod
    def _next_interval(current: int, interval: int) -> int:
        if interval <= 0:
            return 0
        return ((int(current) // int(interval)) + 1) * int(interval)


class AgentlaceActorRuntime(Runner):
    """Agentlace actor that rolls out env chunks and streams compact transitions."""

    def __init__(
        self,
        env: EnvBackend,
        policy: PolicyBackend,
        algorithm: Algorithm,
        feature_processor: FeatureProcessor,
        max_env_steps: int = 1000,
        warmup_steps: int = 0,
        execute_horizon: int = 1,
        gamma: float = 0.99,
        trainer_ip: str = "127.0.0.1",
        trainer_port: int = 5488,
        broadcast_port: int = 5489,
        actor_name: str = "actor_env",
        actor_queue_capacity: int = 2000,
        weight_update_interval_steps: int = 1,
        stats_interval_env_steps: int = 50,
        initial_weight_timeout_sec: float = 300.0,
        client_update_sleep_sec: float = 0.1,
        run_dir: str | None = None,
        request_type: str = "send-stats",
    ) -> None:
        if execute_horizon <= 0:
            raise ValueError(f"execute_horizon must be positive, got {execute_horizon}")
        self.env = env
        self.policy = policy
        self.algorithm = algorithm
        self.feature_processor = feature_processor
        self.max_env_steps = int(max_env_steps)
        self.warmup_steps = int(warmup_steps)
        self.execute_horizon = int(execute_horizon)
        self.gamma = float(gamma)
        self.trainer_ip = str(trainer_ip)
        self.trainer_port = int(trainer_port)
        self.broadcast_port = int(broadcast_port)
        self.actor_name = str(actor_name)
        self.actor_queue_capacity = int(actor_queue_capacity)
        self.weight_update_interval_steps = int(weight_update_interval_steps)
        self.stats_interval_env_steps = int(stats_interval_env_steps)
        self.initial_weight_timeout_sec = float(initial_weight_timeout_sec)
        self.client_update_sleep_sec = float(client_update_sleep_sec)
        self.run_dir = Path(run_dir) if run_dir else None
        self.request_type = str(request_type)
        self._has_policy_state = False

    def run(self) -> dict:
        agentlace = _import_agentlace()
        self._prepare_outputs()
        data_store = agentlace.QueuedDataStore(self.actor_queue_capacity)
        client = agentlace.TrainerClient(
            self.actor_name,
            self.trainer_ip,
            _make_trainer_config(agentlace, self.trainer_port, self.broadcast_port, [self.request_type]),
            data_store,
            wait_for_server=True,
        )
        client.recv_network_callback(self._network_callback)
        self._wait_for_initial_weights(client)

        try:
            obs = self.env.reset()
            env_steps = 0
            episodes = 0
            total_reward = 0.0
            next_weight_update_at = self._next_interval(env_steps, self.weight_update_interval_steps)
            next_stats_at = self._next_interval(env_steps, self.stats_interval_env_steps)
            start_time = time.perf_counter()

            while env_steps < self.max_env_steps:
                chunk_start = time.perf_counter()
                features = self.policy.extract_features(obs)
                reference = _reference_chunk_from_features(features)
                agent_obs = self.feature_processor.process(obs, features)
                if env_steps < self.warmup_steps:
                    action_chunk = reference
                else:
                    action_chunk = self.algorithm.act(obs, features=features, agent_obs=agent_obs)

                execute_actions = action_chunk.actions[: self.execute_horizon]
                next_obs, reward, done, truncated, info = self.env.step_chunk(execute_actions)
                executed_steps = int(info.get("executed_steps", len(execute_actions)))
                env_steps += executed_steps
                total_reward += float(reward)
                terminal = bool(done or truncated)
                next_agent_obs = None
                if not terminal:
                    next_features = self.policy.extract_features(next_obs)
                    next_agent_obs = self.feature_processor.process(next_obs, next_features)

                transition = CompactTransition(
                    agent_obs=agent_obs,
                    next_agent_obs=next_agent_obs,
                    action=np.asarray(action_chunk.actions, dtype=np.float32).reshape(-1),
                    reward=float(reward),
                    done=bool(done),
                    truncated=bool(truncated),
                    discount=0.0 if terminal else self.gamma**executed_steps,
                    executed_steps=executed_steps,
                    env_steps=env_steps,
                    info=info,
                )
                data_store.insert(transition.to_payload())
                client.update()

                metric = {
                    "role": "actor",
                    "env_steps": env_steps,
                    "episode": episodes,
                    "reward": float(reward),
                    "done": bool(done),
                    "truncated": bool(truncated),
                    "executed_steps": executed_steps,
                    "wall_time_sec": time.perf_counter() - start_time,
                    "chunk_time_sec": time.perf_counter() - chunk_start,
                }
                self._write_actor_metric(metric)
                if self.stats_interval_env_steps > 0 and env_steps >= next_stats_at:
                    client.request(self.request_type, metric)
                    next_stats_at += self.stats_interval_env_steps

                if self.weight_update_interval_steps > 0 and env_steps >= next_weight_update_at:
                    client.update()
                    next_weight_update_at += self.weight_update_interval_steps

                if terminal:
                    episodes += 1
                    obs = self.env.reset()
                else:
                    obs = next_obs

            summary = {
                "role": "actor",
                "env_steps": env_steps,
                "episodes": episodes,
                "total_reward": total_reward,
                "received_policy_state": self._has_policy_state,
            }
            self._write_actor_metric({"summary": summary})
            if self.run_dir is not None:
                (self.run_dir / "actor_summary.json").write_text(
                    json.dumps(_json_sanitize(summary), indent=2, sort_keys=True) + "\n"
                )
            return summary
        finally:
            stop = getattr(client, "stop", None)
            if callable(stop):
                stop()

    def _network_callback(self, payload: dict[str, Any]) -> None:
        if payload is not None:
            self.algorithm.load_policy_state_dict(payload)
            self._has_policy_state = True

    def _wait_for_initial_weights(self, client: Any) -> None:
        deadline = time.perf_counter() + self.initial_weight_timeout_sec
        while not self._has_policy_state:
            client.update()
            if time.perf_counter() > deadline:
                raise TimeoutError("timed out waiting for initial learner policy weights")
            time.sleep(self.client_update_sleep_sec)

    def _prepare_outputs(self) -> None:
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "actor_metrics.jsonl").write_text("")

    def _write_actor_metric(self, metric: dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        with (self.run_dir / "actor_metrics.jsonl").open("a") as f:
            f.write(json.dumps(_json_sanitize(metric), sort_keys=True) + "\n")

    @staticmethod
    def _next_interval(current: int, interval: int) -> int:
        if interval <= 0:
            return 0
        return ((int(current) // int(interval)) + 1) * int(interval)


def _reference_chunk_from_features(features) -> ActionChunk:
    if features.reference_actions is None:
        raise ValueError("Agentlace actor requires PolicyFeatures.reference_actions")
    reference = ActionChunk(
        actions=np.asarray(features.reference_actions, dtype=np.float32),
        horizon=int(features.reference_actions.shape[0]),
        metadata={"source": "policy_features.reference_actions"},
    )
    reference.validate()
    return reference


def _make_agentlace_replay_store(agentlace: Any, replay: CompactReplayBuffer) -> Any:
    class AgentlaceCompactReplayStore(agentlace.DataStoreBase):
        def __init__(self) -> None:
            self._latest_data_id = 0

        def insert(self, data: Any) -> None:
            replay.add(data)
            self._latest_data_id += 1

        def latest_data_id(self) -> int:
            return int(self._latest_data_id)

        def get_latest_data(self, from_id: int) -> list[Any]:
            del from_id
            return []

        def __len__(self) -> int:
            return len(replay)

    return AgentlaceCompactReplayStore()


def _make_trainer_config(agentlace: Any, trainer_port: int, broadcast_port: int, request_types: list[str]) -> Any:
    return agentlace.TrainerConfig(
        port_number=int(trainer_port),
        broadcast_port=int(broadcast_port),
        request_types=list(request_types),
    )


def _import_agentlace() -> Any:
    try:
        from agentlace.data.data_store import DataStoreBase, QueuedDataStore
        from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer
    except ImportError as exc:
        raise ImportError(
            "Agentlace runtime requires the 'agentlace' package. Install VLA-RL dependencies before running "
            "scripts/train_agentlace.py."
        ) from exc

    class _AgentlaceModule:
        pass

    module = _AgentlaceModule()
    module.DataStoreBase = DataStoreBase
    module.QueuedDataStore = QueuedDataStore
    module.TrainerClient = TrainerClient
    module.TrainerConfig = TrainerConfig
    module.TrainerServer = TrainerServer
    return module


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
