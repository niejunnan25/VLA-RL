#!/usr/bin/env python3
"""Official-style OpenPI LIBERO evaluator.

This script intentionally avoids importing VLA-RL modules.  It mirrors the
OpenPI `examples/libero/main_10.py` evaluation path so base-policy checks are
not affected by VLA-RL's Python/runtime stack.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import logging
import os
import pathlib
import sys
from typing import Any, Iterable

import imageio
import numpy as np

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
DEFAULT_OPENPI_ROOT = pathlib.Path("/vla/users/niejunnan/codebase/openpi")
DEFAULT_LIBERO_ROOT = pathlib.Path("/vla/users/niejunnan/codebase/LIBERO")


def _is_complete_libero_root(libero_root: pathlib.Path) -> bool:
    package_root = libero_root / "libero" / "libero"
    required_paths = (
        package_root / "bddl_files",
        package_root / "init_files",
        package_root / "assets" / "scenes" / "libero_tabletop_base_style.xml",
    )
    return all(path.exists() for path in required_paths)


def _write_libero_config(libero_root: pathlib.Path, config_dir: pathlib.Path) -> None:
    package_root = libero_root / "libero" / "libero"
    datasets_root = package_root.parent / "datasets"
    config_text = (
        f"benchmark_root: {package_root.as_posix()}\n"
        f"bddl_files: {(package_root / 'bddl_files').as_posix()}\n"
        f"init_states: {(package_root / 'init_files').as_posix()}\n"
        f"datasets: {datasets_root.as_posix()}\n"
        f"assets: {(package_root / 'assets').as_posix()}\n"
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(config_text)


def _setup_paths(openpi_root: pathlib.Path, libero_root: pathlib.Path, output_dir: pathlib.Path) -> None:
    openpi_root = openpi_root.expanduser().resolve()
    libero_root = libero_root.expanduser().resolve()
    if not _is_complete_libero_root(libero_root):
        raise FileNotFoundError(f"LIBERO root is incomplete: {libero_root}")
    openpi_client_src = openpi_root / "packages" / "openpi-client" / "src"
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))
    if str(openpi_client_src) not in sys.path:
        sys.path.insert(0, str(openpi_client_src))
    config_dir = output_dir / ".libero_config"
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    _write_libero_config(libero_root, config_dir)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def _sanitize(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text)


def _parse_task_ids(value: str | None, n_tasks: int) -> list[int]:
    if value is None or value.strip() == "":
        return list(range(n_tasks))
    ids = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        task_id = int(item)
        if task_id < 0 or task_id >= n_tasks:
            raise ValueError(f"task id {task_id} out of range [0, {n_tasks})")
        ids.append(task_id)
    if not ids:
        raise ValueError("--task-ids did not contain any task ids")
    return ids


def _max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"unknown task suite: {task_suite_name}")


def _make_logger(log_path: pathlib.Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _get_libero_env(task: Any, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, str(task.language)


def _policy_input(raw_obs: dict[str, Any], task_description: str, resize_size: int) -> dict[str, Any]:
    from openpi_client import image_tools

    image = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate(
        (
            raw_obs["robot0_eef_pos"],
            _quat2axisangle(raw_obs["robot0_eef_quat"]),
            raw_obs["robot0_gripper_qpos"],
        )
    )
    return {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": task_description,
    }


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_paths(pathlib.Path(args.openpi_root), pathlib.Path(args.libero_root), output_dir)

    from libero.libero import benchmark
    from openpi_client import websocket_client_policy

    np.random.seed(int(args.seed))
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = _parse_task_ids(args.task_ids, int(task_suite.n_tasks))
    max_steps = int(args.max_steps or _max_steps_for_suite(args.task_suite_name))
    client = websocket_client_policy.WebsocketClientPolicy(args.host, int(args.port))

    summary: dict[str, Any] = {
        "task_suite_name": args.task_suite_name,
        "task_ids": task_ids,
        "num_trials_per_task": int(args.num_trials_per_task),
        "seed": int(args.seed),
        "max_steps": max_steps,
        "num_steps_wait": int(args.num_steps_wait),
        "tasks": {},
    }
    total_successes = 0
    total_episodes = 0

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, int(args.seed))
        task_dir = output_dir / f"task_{task_id}"
        videos_dir = task_dir / "videos"
        if args.save_videos:
            videos_dir.mkdir(parents=True, exist_ok=True)
        logger = _make_logger(task_dir / "logs" / f"task_{task_id}_{_datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        episodes_path = task_dir / "eval_episodes.jsonl"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        episodes_path.write_text("")
        task_successes = 0
        records = []
        logger.info("Task suite: %s", args.task_suite_name)
        logger.info("Task id: %s", task_id)
        logger.info("Task description: %s", task_description)
        logger.info("Loaded %s init states", len(initial_states))
        try:
            for episode_idx in range(int(args.num_trials_per_task)):
                init_state_idx = episode_idx % len(initial_states)
                raw_obs = env.reset()
                del raw_obs
                raw_obs = env.set_init_state(initial_states[init_state_idx])
                done = False
                steps = 0
                frames = []
                while steps < max_steps + int(args.num_steps_wait):
                    if steps < int(args.num_steps_wait):
                        raw_obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        steps += 1
                        continue
                    element = _policy_input(raw_obs, task_description, int(args.resize_size))
                    if args.save_videos:
                        frames.append(np.asarray(element["observation/image"]))
                    action_chunk = client.infer(element)["actions"]
                    if len(action_chunk) < int(args.replan_steps):
                        raise RuntimeError(
                            f"policy returned {len(action_chunk)} actions, replan_steps={args.replan_steps}"
                        )
                    for action in action_chunk[: int(args.replan_steps)]:
                        raw_obs, _, done, _ = env.step(np.asarray(action).tolist())
                        steps += 1
                        if done or steps >= max_steps + int(args.num_steps_wait):
                            break
                    if done:
                        break
                success = bool(done)
                task_successes += int(success)
                total_successes += int(success)
                total_episodes += 1
                record = {
                    "episode": episode_idx,
                    "init_state_idx": init_state_idx,
                    "success": success,
                    "length_with_wait": steps,
                    "env_steps_after_wait": max(0, steps - int(args.num_steps_wait)),
                }
                records.append(record)
                with episodes_path.open("a") as f:
                    f.write(json.dumps(record, sort_keys=True) + "\n")
                if args.save_videos and frames:
                    suffix = "success" if success else "failure"
                    video_name = f"episode_{episode_idx:03d}_{_sanitize(task_description)}_{suffix}.mp4"
                    imageio.mimwrite(videos_dir / video_name, frames, fps=10)
                logger.info("Episode %s init_state_idx=%s Success: %s", episode_idx, init_state_idx, success)
        finally:
            env.close()
        task_summary = {
            "task_description": task_description,
            "successes": task_successes,
            "episodes": len(records),
            "success_rate": float(task_successes / len(records)) if records else 0.0,
        }
        (task_dir / "eval_summary.json").write_text(json.dumps(task_summary, indent=2, sort_keys=True) + "\n")
        summary["tasks"][str(task_id)] = task_summary

    summary["total_successes"] = total_successes
    summary["total_episodes"] = total_episodes
    summary["total_success_rate"] = float(total_successes / total_episodes) if total_episodes else 0.0
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official-style OpenPI LIBERO evaluator with task-id filtering.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids. Defaults to all tasks.")
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--openpi-root", default=str(DEFAULT_OPENPI_ROOT))
    parser.add_argument("--libero-root", default=str(DEFAULT_LIBERO_ROOT))
    parser.add_argument("--save-videos", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    summary = run_eval(parse_args())
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
