#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
import uuid
from typing import Any, Sequence

import numpy as np
from omegaconf import DictConfig
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.libero.rlpd.config import create_rlpd_obs_builder, load_config, validate_rlpd_cfg
from examples.libero.rlpd.scripts.convert_lerobot_expert_replay import (
    OFFLINE_IMAGE_PREPROCESS,
    attach_mc_returns,
    get_libero_task_prompt,
    make_observation,
    resolve_lerobot_task_repo,
)
from vla_rl.algorithms.rlpd import build_rlpd_obs, write_offline_episode
from vla_rl.data import Observation, Transition
from vla_rl.rewards.processor import observation_to_reward_payload
from vla_rl.rewards.progress import (
    RemoteProgressClient,
    compute_potential_discount,
    compute_progress_reward,
)


@dataclass(slots=True)
class ProgressStep:
    previous_progress: float | None
    progress: float
    trajectory_indices: list[int]
    query_indices: list[int]
    absolute_query_indices: list[int]
    progress_values: list[float]
    latency_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relabel task-matched LIBERO LeRobot expert demos with a remote absolute-progress reward model."
    )
    parser.add_argument("--config", required=True, help="RLPD YAML config with reward.source=remote_progress.")
    parser.add_argument("--lerobot-root", default="/vla/users/niejunnan/datasets/libero_lerobot")
    parser.add_argument("--output-dir", default=None, help="Output replay dir. Defaults to runtime.offline_replay_path.")
    parser.add_argument("--task-prompt", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-transitions-per-episode", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None, help="Discount. Defaults to runtime.gamma.")
    parser.add_argument("--step-reward", type=float, default=0.0)
    parser.add_argument("--terminal-reward", type=float, default=1.0)
    parser.add_argument("--reward-type", default=None, help="Defaults to reward.type from the YAML.")
    parser.add_argument("--reward-scale", type=float, default=None, help="Defaults to reward.scale from the YAML.")
    parser.add_argument("--initial-progress", choices=("query_start", "zero"), default=None)
    parser.add_argument("--remote-url", default=None, help="Defaults to reward.remote.url from the YAML.")
    parser.add_argument("--remote-method", default=None, help="Defaults to reward.remote.method from the YAML.")
    parser.add_argument("--remote-timeout", type=float, default=None)
    parser.add_argument("--remote-retries", type=int, default=None)
    parser.add_argument("--remote-retry-sleep", type=float, default=None)
    parser.add_argument("--image-keys", nargs="+", default=None, help="Defaults to reward.trajectory.image_keys.")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, list(args.overrides))
    validate_rlpd_cfg(cfg)
    reward_cfg = _require_remote_reward_cfg(cfg)

    suite_name = str(cfg.env.task_suite_name)
    task_id = int(cfg.env.task_id)
    action_dim = int(cfg.algorithm.action_dim)
    image_size = int(cfg.env.get("image_size", 224))
    gamma = float(cfg.runtime.gamma if args.gamma is None else args.gamma)
    reward_type = str(args.reward_type or reward_cfg.get("type", "env_plus_potential_delta"))
    if args.max_transitions_per_episode is not None and int(args.max_transitions_per_episode) <= 0:
        raise ValueError("--max-transitions-per-episode must be positive when set")
    reward_scale = float(args.reward_scale if args.reward_scale is not None else reward_cfg.get("scale", 1.0))
    initial_progress = str(args.initial_progress or reward_cfg.get("initial_progress", "query_start"))
    output_dir = Path(args.output_dir or cfg.runtime.offline_replay_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = f"offline_relabel:{uuid.uuid4().hex}"

    remote_cfg = reward_cfg.get("remote", {})
    remote_url = str(args.remote_url or remote_cfg.get("url", ""))
    if not remote_url:
        raise ValueError("remote reward relabel requires reward.remote.url or --remote-url")
    remote_method = str(args.remote_method or remote_cfg.get("method", "predict_progress"))
    remote_timeout = float(args.remote_timeout if args.remote_timeout is not None else remote_cfg.get("timeout", 120.0))
    remote_retries = int(args.remote_retries if args.remote_retries is not None else remote_cfg.get("retries", 1))
    remote_retry_sleep = float(
        args.remote_retry_sleep if args.remote_retry_sleep is not None else remote_cfg.get("retry_sleep", 0.5)
    )
    image_keys = _resolve_image_keys(cfg, args.image_keys)

    task_prompt = args.task_prompt or get_libero_task_prompt(
        suite_name=suite_name,
        task_id=task_id,
        libero_root=cfg.env.get("libero_root", None) or cfg.env.get("benchmark_root", None),
    )
    repo_dir, repo_task = resolve_lerobot_task_repo(Path(args.lerobot_root), suite_name, task_prompt)
    obs_builder = create_rlpd_obs_builder(cfg)
    client = RemoteProgressClient(
        remote_url,
        method=remote_method,
        timeout=remote_timeout,
        retries=remote_retries,
        retry_sleep=remote_retry_sleep,
    )

    parquet_files = sorted((repo_dir / "data").rglob("episode_*.parquet"))
    if args.max_episodes is not None:
        parquet_files = parquet_files[: int(args.max_episodes)]
    if not parquet_files:
        raise FileNotFoundError(f"no episode parquet files found under {repo_dir / 'data'}")

    progress_events_path = output_dir / "progress_events.jsonl"
    progress_events_path.write_text("")
    start_time = time.perf_counter()
    episodes_written = 0
    transitions_written = 0
    skipped_short_episodes = 0
    try:
        for parquet_path in parquet_files:
            relabelled = relabel_episode(
                parquet_path,
                obs_builder=obs_builder,
                client=client,
                episode_id=episodes_written,
                task_prompt=task_prompt,
                suite_name=suite_name,
                task_id=task_id,
                action_dim=action_dim,
                image_size=image_size,
                gamma=gamma,
                step_reward=float(args.step_reward),
                terminal_reward=float(args.terminal_reward),
                reward_type=reward_type,
                reward_scale=reward_scale,
                initial_progress=initial_progress,
                image_keys=image_keys,
                max_transitions_per_episode=args.max_transitions_per_episode,
                source_repo=repo_dir,
                session_id=session_id,
            )
            if not relabelled:
                skipped_short_episodes += 1
                continue
            transitions = [transition for transition, _progress in relabelled]
            attach_mc_returns(transitions, gamma=gamma)
            transitions_written += len(transitions)
            write_offline_episode(
                output_dir,
                episodes_written,
                transitions,
                manifest_stats={
                    "format_note": "single-step RLPD expert replay relabelled with remote absolute progress",
                    "suite_name": suite_name,
                    "task_id": task_id,
                    "task_prompt": task_prompt,
                    "lerobot_root": str(Path(args.lerobot_root).expanduser()),
                    "matched_repo": str(repo_dir),
                    "matched_repo_task": repo_task,
                    "image_preprocess": OFFLINE_IMAGE_PREPROCESS,
                    "image_size": image_size,
                    "step_reward": float(args.step_reward),
                    "terminal_reward": float(args.terminal_reward),
                    "gamma": gamma,
                    "reward_type": reward_type,
                    "reward_scale": reward_scale,
                    "initial_progress": initial_progress,
                    "reward_source": "remote_progress",
                    "reward_remote_url": remote_url,
                    "max_transitions_per_episode": args.max_transitions_per_episode,
                    "reward_remote_method": remote_method,
                    "reward_session_id": session_id,
                    "image_keys": list(image_keys),
                    "episodes_written": episodes_written + 1,
                    "transitions_written": transitions_written,
                    "skipped_short_episodes": skipped_short_episodes,
                },
            )
            _append_progress_events(
                progress_events_path,
                episode_id=episodes_written,
                task_prompt=task_prompt,
                relabelled=relabelled,
                reward_type=reward_type,
                reward_scale=reward_scale,
                reward_remote_url=remote_url,
            )
            episodes_written += 1
            print(
                json.dumps(
                    {
                        "episode": episodes_written,
                        "source": str(parquet_path),
                        "transitions": len(transitions),
                        "total_transitions": transitions_written,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        client.close()

    summary = {
        "output_dir": str(output_dir),
        "suite_name": suite_name,
        "task_id": task_id,
        "task_prompt": task_prompt,
        "matched_repo": str(repo_dir),
        "matched_repo_task": repo_task,
        "episodes_written": episodes_written,
        "transitions_written": transitions_written,
        "skipped_short_episodes": skipped_short_episodes,
        "reward_type": reward_type,
        "reward_scale": reward_scale,
        "reward_remote_url": remote_url,
        "reward_session_id": session_id,
        "max_transitions_per_episode": args.max_transitions_per_episode,
        "elapsed_sec": time.perf_counter() - start_time,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


def relabel_episode(
    parquet_path: Path,
    *,
    obs_builder: Any,
    client: RemoteProgressClient,
    episode_id: int,
    task_prompt: str,
    suite_name: str,
    task_id: int,
    action_dim: int,
    image_size: int,
    gamma: float,
    step_reward: float,
    terminal_reward: float,
    reward_type: str,
    reward_scale: float,
    initial_progress: str,
    image_keys: tuple[str, ...],
    source_repo: Path,
    max_transitions_per_episode: int | None = None,
    session_id: str = "offline_relabel",
) -> list[tuple[Transition, ProgressStep]]:
    table = pq.read_table(
        parquet_path,
        columns=["image", "wrist_image", "state", "actions", "frame_index", "episode_index", "task_index"],
    )
    data = table.to_pydict()
    n_rows = len(data["actions"])
    if n_rows < 2:
        return []
    if max_transitions_per_episode is not None:
        n_rows = min(n_rows, int(max_transitions_per_episode) + 1)
    observations = [
        make_observation(
            image_cell=data["image"][idx],
            wrist_cell=data["wrist_image"][idx],
            state=data["state"][idx],
            task_prompt=task_prompt,
            image_size=image_size,
            source_dir=source_repo,
        )
        for idx in range(n_rows)
    ]
    progress_steps = predict_episode_progress_steps(
        client,
        observations,
        episode_id=episode_id,
        task_prompt=task_prompt,
        task_id=task_id,
        initial_progress=initial_progress,
        image_keys=image_keys,
        session_id=session_id,
    )
    source_episode_index = int(data["episode_index"][0]) if data.get("episode_index") else -1
    relabelled: list[tuple[Transition, ProgressStep]] = []
    for idx in range(n_rows - 1):
        progress_step = progress_steps[idx]
        done = idx == n_rows - 2
        env_reward = terminal_reward if done else step_reward
        discount = 0.0 if done else float(gamma)
        potential_discount = compute_potential_discount(
            gamma=float(gamma),
            executed_steps=1,
            discount=float(discount),
            terminal=bool(done),
        )
        reward = compute_progress_reward(
            reward_type,
            env_reward=float(env_reward),
            progress=float(progress_step.progress),
            previous_progress=progress_step.previous_progress,
            gamma=float(gamma),
            executed_steps=1,
            scale=float(reward_scale),
            discount=float(discount),
            terminal=bool(done),
        )
        action = np.asarray(data["actions"][idx], dtype=np.float32).reshape(-1)
        if action.shape != (action_dim,):
            raise ValueError(f"{parquet_path} row {idx} action shape {action.shape}, expected {(int(action_dim),)}")
        frame_index = int(data["frame_index"][idx])
        next_frame_index = int(data["frame_index"][idx + 1])
        transition = Transition(
            obs=build_rlpd_obs(observations[idx], builder=obs_builder),
            next_obs=build_rlpd_obs(observations[idx + 1], builder=obs_builder),
            action=action,
            reward=float(reward),
            done=bool(done),
            truncated=False,
            discount=float(discount),
            executed_steps=1,
            env_steps=idx + 1,
            info={
                "source": "libero_lerobot_expert",
                "source_parquet": str(parquet_path),
                "source_episode_index": source_episode_index,
                "source_frame_index": frame_index,
                "source_next_frame_index": next_frame_index,
                "source_task_index": int(data["task_index"][idx]) if data.get("task_index") else 0,
                "suite_name": suite_name,
                "task_id": int(task_id),
                "task_prompt": task_prompt,
                "expert_terminal": bool(done),
                "critic_terminal": bool(done),
                "reward_type": str(reward_type),
                "env_reward": float(env_reward),
                "reward_model_progress": float(progress_step.progress),
                "reward_model_previous_progress": None
                if progress_step.previous_progress is None
                else float(progress_step.previous_progress),
                "reward_model_latency_sec": float(progress_step.latency_sec),
                "reward_potential_discount": float(potential_discount),
                "reward_model_progress_values": [float(value) for value in progress_step.progress_values],
                "reward_model_trajectory_indices": [int(value) for value in progress_step.trajectory_indices],
                "reward_model_query_indices": [int(value) for value in progress_step.query_indices],
                "reward_model_absolute_query_indices": [int(value) for value in progress_step.absolute_query_indices],
            },
        )
        relabelled.append((transition, progress_step))
    return relabelled


def predict_episode_progress_steps(
    client: RemoteProgressClient,
    observations: Sequence[Observation],
    *,
    episode_id: int,
    task_prompt: str,
    task_id: int,
    initial_progress: str,
    image_keys: tuple[str, ...],
    session_id: str = "offline_relabel",
) -> list[ProgressStep]:
    if initial_progress not in {"query_start", "zero"}:
        raise ValueError("initial_progress must be query_start or zero")
    if len(observations) < 2:
        return []
    steps: list[ProgressStep] = []
    last_progress: float | None = None
    for idx in range(len(observations) - 1):
        current_index = idx + 1
        needs_start_query = idx == 0 and initial_progress == "query_start"
        trajectory = [observations[0], observations[current_index]]
        trajectory_indices = [0, current_index]
        if needs_start_query:
            query_indices = [0, 1]
            absolute_query_indices = [0, current_index]
            expected_count = 2
        else:
            query_indices = [1]
            absolute_query_indices = [current_index]
            expected_count = 1

        request = build_progress_request(
            episode_id=episode_id,
            chunk_index=idx,
            task_prompt=task_prompt,
            task_id=task_id,
            trajectory=trajectory,
            trajectory_indices=trajectory_indices,
            query_indices=query_indices,
            absolute_query_indices=absolute_query_indices,
            image_keys=image_keys,
            session_id=session_id,
            done=current_index == len(observations) - 1,
        )
        start = time.perf_counter()
        values = client.predict_progress(request, expected_count=expected_count)
        latency = time.perf_counter() - start
        if needs_start_query:
            previous_progress = float(values[0])
            progress = float(values[-1])
        else:
            previous_progress = last_progress if last_progress is not None else 0.0
            progress = float(values[-1])
        last_progress = progress
        steps.append(
            ProgressStep(
                previous_progress=None if previous_progress is None else float(previous_progress),
                progress=float(progress),
                trajectory_indices=list(trajectory_indices),
                query_indices=list(query_indices),
                absolute_query_indices=list(absolute_query_indices),
                progress_values=[float(value) for value in values],
                latency_sec=float(latency),
            )
        )
    return steps


def build_progress_request(
    *,
    episode_id: int,
    chunk_index: int,
    task_prompt: str,
    task_id: int,
    trajectory: Sequence[Observation],
    trajectory_indices: Sequence[int],
    query_indices: Sequence[int],
    absolute_query_indices: Sequence[int],
    image_keys: tuple[str, ...],
    session_id: str,
    done: bool,
) -> dict[str, Any]:
    return {
        "episode_id": int(episode_id),
        "chunk_index": int(chunk_index),
        "session_id": str(session_id),
        "task": str(task_prompt),
        "trajectory": [observation_to_reward_payload(obs, image_keys=image_keys) for obs in trajectory],
        "trajectory_indices": [int(value) for value in trajectory_indices],
        "query_indices": [int(value) for value in query_indices],
        "absolute_query_indices": [int(value) for value in absolute_query_indices],
        "image_keys": tuple(image_keys),
        "metadata": {
            "session_id": str(session_id),
            "task_id": int(task_id),
            "done": bool(done),
            "truncated": False,
            "executed_steps": 1,
        },
    }


def _append_progress_events(
    path: Path,
    *,
    episode_id: int,
    task_prompt: str,
    relabelled: Sequence[tuple[Transition, ProgressStep]],
    reward_type: str,
    reward_scale: float,
    reward_remote_url: str,
) -> None:
    with path.open("a") as fp:
        for chunk_index, (transition, progress_step) in enumerate(relabelled):
            fp.write(
                json.dumps(
                    {
                        "role": "reward_progress",
                        "source": "offline_relabel",
                        "status": "ok",
                        "episode_id": int(episode_id),
                        "chunk_index": int(chunk_index),
                        "task": task_prompt,
                        "trajectory_indices": progress_step.trajectory_indices,
                        "query_indices": progress_step.query_indices,
                        "absolute_query_indices": progress_step.absolute_query_indices,
                        "progress_values": progress_step.progress_values,
                        "progress": float(progress_step.progress),
                        "previous_progress": None
                        if progress_step.previous_progress is None
                        else float(progress_step.previous_progress),
                        "env_reward": float(transition.info.get("env_reward", 0.0)),
                        "computed_reward": float(transition.reward),
                        "reward_type": reward_type,
                        "scale": float(reward_scale),
                        "discount": float(transition.discount),
                        "potential_discount": float(
                            transition.info.get("reward_potential_discount", transition.discount)
                        ),
                        "executed_steps": int(transition.executed_steps),
                        "env_steps": int(transition.env_steps),
                        "done": bool(transition.done),
                        "truncated": bool(transition.truncated),
                        "latency_sec": float(progress_step.latency_sec),
                        "reward_remote_url": reward_remote_url,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _require_remote_reward_cfg(cfg: DictConfig) -> DictConfig:
    reward_cfg = cfg.get("reward", None)
    if reward_cfg is None:
        raise ValueError("offline reward relabel requires a reward section")
    if "terminal_potential" in reward_cfg:
        raise ValueError(
            "reward.terminal_potential is no longer supported; terminal potential is always zeroed"
        )
    if str(reward_cfg.get("source", "remote_progress")) != "remote_progress":
        raise ValueError("offline reward relabel requires reward.source=remote_progress")
    return reward_cfg


def _resolve_image_keys(cfg: DictConfig, explicit: Sequence[str] | None) -> tuple[str, ...]:
    if explicit:
        return tuple(str(key) for key in explicit)
    reward_cfg = cfg.get("reward", {})
    trajectory_cfg = reward_cfg.get("trajectory", {}) if reward_cfg is not None else {}
    image_keys = trajectory_cfg.get("image_keys", None)
    if image_keys:
        return tuple(str(key) for key in image_keys)
    return tuple(str(key) for key in cfg.rlpd_observation.image_keys)


if __name__ == "__main__":
    main()
