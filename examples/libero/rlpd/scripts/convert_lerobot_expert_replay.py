#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.libero.rlpd.config import load_config, validate_rlpd_cfg, create_rlpd_obs_builder
from vla_rl.algorithms.rlpd import build_rlpd_obs, write_offline_episode
from vla_rl.data import Observation, Transition
from vla_rl.envs.libero.observation import normalize_image, resize_nearest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert task-matched LIBERO LeRobot expert demos into single-step RLPD replay."
    )
    parser.add_argument("--config", required=True, help="RLPD YAML config.")
    parser.add_argument(
        "--lerobot-root",
        default="/vla/users/niejunnan/datasets/libero_lerobot",
        help="Root containing suite/task LeRobot repos, e.g. /.../libero_lerobot.",
    )
    parser.add_argument("--output-dir", default=None, help="Output RLPD replay dir. Defaults to runtime.offline_replay_path.")
    parser.add_argument("--task-prompt", default=None, help="Optional explicit task prompt override. Defaults to LIBERO benchmark task_id metadata.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None, help="Discount. Defaults to runtime.gamma.")
    parser.add_argument("--step-reward", type=float, default=0.0)
    parser.add_argument("--terminal-reward", type=float, default=1.0)
    parser.add_argument("--image-preprocess", choices=("none", "libero"), default="none")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, list(args.overrides))
    validate_rlpd_cfg(cfg)
    suite_name = str(cfg.env.task_suite_name)
    task_id = int(cfg.env.task_id)
    action_dim = int(cfg.algorithm.action_dim)
    image_size = int(cfg.env.get("image_size", 224))
    gamma = float(cfg.runtime.gamma if args.gamma is None else args.gamma)
    output_dir = Path(args.output_dir or cfg.runtime.offline_replay_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_prompt = args.task_prompt or get_libero_task_prompt(
        suite_name=suite_name,
        task_id=task_id,
        libero_root=cfg.env.get("libero_root", None) or cfg.env.get("benchmark_root", None),
    )
    repo_dir, repo_task = resolve_lerobot_task_repo(Path(args.lerobot_root), suite_name, task_prompt)
    obs_builder = create_rlpd_obs_builder(cfg)

    parquet_files = sorted((repo_dir / "data").rglob("episode_*.parquet"))
    if args.max_episodes is not None:
        parquet_files = parquet_files[: int(args.max_episodes)]
    if not parquet_files:
        raise FileNotFoundError(f"no episode parquet files found under {repo_dir / 'data'}")

    start_time = time.perf_counter()
    episodes_written = 0
    transitions_written = 0
    skipped_short_episodes = 0
    for parquet_path in parquet_files:
        transitions = convert_episode(
            parquet_path,
            obs_builder=obs_builder,
            task_prompt=task_prompt,
            suite_name=suite_name,
            task_id=task_id,
            action_dim=action_dim,
            image_size=image_size,
            image_preprocess=str(args.image_preprocess),
            gamma=gamma,
            step_reward=float(args.step_reward),
            terminal_reward=float(args.terminal_reward),
            source_repo=repo_dir,
        )
        if not transitions:
            skipped_short_episodes += 1
            continue
        attach_mc_returns(transitions, gamma=gamma)
        transitions_written += len(transitions)
        write_offline_episode(
            output_dir,
            episodes_written,
            transitions,
            manifest_stats={
                "format_note": "single-step RLPD expert replay converted from LeRobot LIBERO demos",
                "suite_name": suite_name,
                "task_id": task_id,
                "task_prompt": task_prompt,
                "lerobot_root": str(Path(args.lerobot_root).expanduser()),
                "matched_repo": str(repo_dir),
                "matched_repo_task": repo_task,
                "image_preprocess": str(args.image_preprocess),
                "image_size": image_size,
                "step_reward": float(args.step_reward),
                "terminal_reward": float(args.terminal_reward),
                "gamma": gamma,
                "episodes_written": episodes_written + 1,
                "transitions_written": transitions_written,
                "skipped_short_episodes": skipped_short_episodes,
            },
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
        "elapsed_sec": time.perf_counter() - start_time,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


def get_libero_task_prompt(*, suite_name: str, task_id: int, libero_root: str | None) -> str:
    candidates: list[Path] = []
    if libero_root:
        candidates.append(Path(str(libero_root)).expanduser())
    candidates.extend(
        [
            Path("/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO"),
            Path("/mnt/workspace/users/niejunnan/codebase/serl_torch/third_party/LIBERO"),
        ]
    )
    last_error: Exception | None = None
    for root in candidates:
        if not root.exists():
            continue
        root_str = str(root.resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            with redirect_stdout(StringIO()):
                from libero.libero.benchmark import get_benchmark
                benchmark = get_benchmark(suite_name)()
            task = benchmark.get_task(int(task_id))
            prompt = str(getattr(task, "language", "") or "").strip()
            if not prompt:
                raise RuntimeError(f"LIBERO task {suite_name}/{task_id} has empty language")
            return prompt
        except Exception as exc:  # pragma: no cover - path fallback
            last_error = exc
    raise RuntimeError(f"failed to resolve LIBERO prompt for {suite_name}/task{task_id}: {last_error}")


def resolve_lerobot_task_repo(lerobot_root: Path, suite_name: str, task_prompt: str) -> tuple[Path, str]:
    suite_dir = lerobot_root.expanduser() / suite_name
    if not suite_dir.exists():
        raise FileNotFoundError(f"LeRobot suite directory does not exist: {suite_dir}")
    target = normalize_prompt(task_prompt)
    matches: list[tuple[Path, str]] = []
    seen: list[tuple[str, str]] = []
    for repo_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
        tasks_path = repo_dir / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            continue
        for line in tasks_path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            task = str(payload.get("task", "")).strip()
            seen.append((repo_dir.name, task))
            if normalize_prompt(task) == target:
                matches.append((repo_dir, task))
    if len(matches) != 1:
        preview = "; ".join(f"{name}: {task}" for name, task in seen[:20])
        raise RuntimeError(
            f"expected exactly one LeRobot repo for prompt {task_prompt!r} under {suite_dir}, "
            f"found {len(matches)}. Seen: {preview}"
        )
    return matches[0]


def convert_episode(
    parquet_path: Path,
    *,
    obs_builder: Any,
    task_prompt: str,
    suite_name: str,
    task_id: int,
    action_dim: int,
    image_size: int,
    image_preprocess: str,
    gamma: float,
    step_reward: float,
    terminal_reward: float,
    source_repo: Path,
) -> list[Transition]:
    table = pq.read_table(parquet_path, columns=["image", "wrist_image", "state", "actions", "frame_index", "episode_index", "task_index"])
    data = table.to_pydict()
    n_rows = len(data["actions"])
    if n_rows < 2:
        return []
    observations = [
        make_observation(
            image_cell=data["image"][idx],
            wrist_cell=data["wrist_image"][idx],
            state=data["state"][idx],
            task_prompt=task_prompt,
            image_size=image_size,
            image_preprocess=image_preprocess,
            source_dir=source_repo,
        )
        for idx in range(n_rows)
    ]
    transitions: list[Transition] = []
    source_episode_index = int(data["episode_index"][0]) if data.get("episode_index") else -1
    for idx in range(n_rows - 1):
        done = idx == n_rows - 2
        action = np.asarray(data["actions"][idx], dtype=np.float32).reshape(-1)
        if action.shape != (action_dim,):
            raise ValueError(f"{parquet_path} row {idx} action shape {action.shape}, expected {(action_dim,)}")
        reward = terminal_reward if done else step_reward
        obs = build_rlpd_obs(observations[idx], builder=obs_builder)
        next_obs = build_rlpd_obs(observations[idx + 1], builder=obs_builder)
        frame_index = int(data["frame_index"][idx])
        next_frame_index = int(data["frame_index"][idx + 1])
        transitions.append(
            Transition(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=float(reward),
                done=bool(done),
                truncated=False,
                discount=0.0 if done else float(gamma),
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
                },
            )
        )
    return transitions


def make_observation(
    *,
    image_cell: Any,
    wrist_cell: Any,
    state: Any,
    task_prompt: str,
    image_size: int,
    image_preprocess: str,
    source_dir: Path,
) -> Observation:
    image = decode_lerobot_image(image_cell, source_dir=source_dir)
    wrist = decode_lerobot_image(wrist_cell, source_dir=source_dir)
    image = preprocess_image(image, image_size=image_size, mode=image_preprocess)
    wrist = preprocess_image(wrist, image_size=image_size, mode=image_preprocess)
    obs = Observation(
        images={
            "image_rgb_0": image,
            "image_rgb_1": wrist,
            "image_rgb_2": np.zeros((image_size, image_size, 3), dtype=np.uint8),
        },
        proprio=np.asarray(state, dtype=np.float32).reshape(-1),
        task=task_prompt,
        raw={"task": task_prompt},
    )
    obs.validate()
    return obs


def decode_lerobot_image(cell: Any, *, source_dir: Path) -> np.ndarray:
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            with Image.open(BytesIO(cell["bytes"])) as image:
                return np.asarray(image.convert("RGB"))
        if cell.get("path"):
            path = Path(cell["path"])
            if not path.is_absolute():
                path = source_dir / path
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"))
    raise ValueError(f"unsupported LeRobot image cell: {type(cell).__name__}")


def preprocess_image(image: np.ndarray, *, image_size: int, mode: str) -> np.ndarray:
    if mode == "libero":
        return normalize_image(image, image_size=image_size)
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[:2] != (image_size, image_size):
        array = resize_nearest(array, (image_size, image_size))
    return np.ascontiguousarray(array)


def attach_mc_returns(transitions: list[Transition], *, gamma: float) -> None:
    running = 0.0
    for transition in reversed(transitions):
        running = float(transition.reward) + float(gamma) * float(transition.discount > 0.0) * running
        transition.info["mc_returns"] = float(running)
        transition.info["mc_returns_valid"] = True


def normalize_prompt(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    main()
