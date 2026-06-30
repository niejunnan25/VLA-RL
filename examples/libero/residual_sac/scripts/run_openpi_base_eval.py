from __future__ import annotations

"""Official-style OpenPI base-policy evaluation for LIBERO residual SAC.

This script intentionally bypasses residual SAC policy/checkpoint loading and runs
the OpenPI base policy with zero residual. It mirrors
openpi/examples/libero/main_10.py for observation preprocessing and websocket
policy calls, while allowing either a local LIBERO env import or the residual_sac
remote env RPC wrapper.
"""

import argparse
import collections
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))


def _prepend_openpi_client_path(openpi_root: str | None) -> None:
    if not openpi_root:
        return
    client_src = Path(openpi_root).expanduser().resolve() / "packages" / "openpi-client" / "src"
    if client_src.exists() and str(client_src) not in sys.path:
        sys.path.insert(0, str(client_src))


_prepend_openpi_client_path(
    os.environ.get("OPENPI_ROOT", "/vla/users/niejunnan/codebase/openpi-modified")
)

from openpi_client import image_tools
from openpi_client import websocket_client_policy

from examples.libero.residual_sac.env.remote_task_env import RemoteLiberoTaskEnv
from examples.libero.residual_sac.env.setup import resolve_max_episode_steps
from examples.libero.residual_sac.env.task_env import LiberoTaskEnv


LIBERO_ENV_RESOLUTION = 256
LIBERO_ACTION_DIM = 7


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat_arr = np.asarray(quat, dtype=np.float32).copy()
    quat_arr[3] = np.clip(quat_arr[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat_arr[3] * quat_arr[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat_arr[:3] * 2.0 * math.acos(float(quat_arr[3])) / den).astype(
        np.float32
    )


def _build_official_openpi_input(
    obs: dict[str, Any],
    *,
    prompt: str,
    resize_size: int,
) -> dict[str, Any]:
    image = np.ascontiguousarray(np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1, ::-1])
    wrist_image = np.ascontiguousarray(
        np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)[::-1, ::-1]
    )
    image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(image, int(resize_size), int(resize_size))
    )
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, int(resize_size), int(resize_size))
    )
    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)
    return {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": str(prompt),
    }


def _make_env(args: argparse.Namespace):
    common_kwargs = dict(
        suite_name=args.task_suite_name,
        task_id=int(args.task_id),
        action_dim=LIBERO_ACTION_DIM,
        resolution=int(args.resolution),
        num_steps_wait=int(args.num_steps_wait),
        max_episode_steps=args.max_steps,
        libero_root=args.libero_root,
        libero_config_dir=args.libero_config_dir,
        libero_datasets_root=args.libero_datasets_root,
        env_seed=int(args.seed),
        logger=logging.getLogger("openpi_base_eval.env"),
    )
    if args.env_backend == "local":
        return LiberoTaskEnv(**common_kwargs)
    if args.env_backend == "remote":
        return RemoteLiberoTaskEnv(
            host=str(args.remote_env_host),
            port=int(args.remote_env_port),
            timeout_sec=float(args.remote_env_timeout_sec),
            **common_kwargs,
        )
    raise ValueError(f"Unsupported env backend: {args.env_backend}")


def _episode_record(
    *,
    episode_idx: int,
    init_state_idx: int,
    success: bool,
    episode_return: float,
    episode_steps: int,
    policy_calls: int,
    elapsed_sec: float,
    final_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "episode_idx": int(episode_idx),
        "init_state_idx": int(init_state_idx),
        "success": bool(success),
        "episode_return": float(episode_return),
        "episode_steps": int(episode_steps),
        "policy_calls": int(policy_calls),
        "elapsed_sec": float(elapsed_sec),
        "final_info": final_info,
    }


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_log_path = output_dir / "episodes.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "config.json"

    np.random.seed(int(args.seed))
    max_steps = int(args.max_steps or resolve_max_episode_steps(args.task_suite_name))

    logging.info("Output dir: %s", output_dir)
    logging.info("Env backend: %s", args.env_backend)
    logging.info("Task: %s task_id=%s", args.task_suite_name, args.task_id)
    logging.info("Policy endpoint: %s:%s", args.policy_host, args.policy_port)
    logging.info("Max policy/env steps per episode after warmup: %s", max_steps)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    env = _make_env(args)
    client = websocket_client_policy.WebsocketClientPolicy(
        str(args.policy_host),
        int(args.policy_port),
    )
    task_prompt = str(env.task_description)
    records: list[dict[str, Any]] = []
    successes = 0
    total_policy_calls = 0
    total_env_steps = 0
    started_at = time.time()

    try:
        with episode_log_path.open("w", encoding="utf-8") as episode_file:
            for episode_idx in range(int(args.num_trials)):
                init_episode_idx = int(args.start_episode_idx) + int(episode_idx)
                episode_start = time.time()
                obs = env.reset(seed=int(args.seed), init_episode_idx=init_episode_idx)
                init_state_idx = getattr(env, "current_init_state_idx", None)
                if init_state_idx is None:
                    init_state_idx = init_episode_idx

                action_plan: collections.deque[np.ndarray] = collections.deque()
                episode_return = 0.0
                episode_steps = 0
                policy_calls = 0
                success = False
                final_info: dict[str, Any] = {}

                while episode_steps < max_steps:
                    if not action_plan:
                        element = _build_official_openpi_input(
                            obs,
                            prompt=task_prompt,
                            resize_size=int(args.resize_size),
                        )
                        prediction = client.infer(element)
                        action_chunk = np.asarray(prediction["actions"], dtype=np.float32)
                        if len(action_chunk) < int(args.replan_steps):
                            raise RuntimeError(
                                "OpenPI returned fewer actions than replan_steps: "
                                f"actions={len(action_chunk)} replan_steps={args.replan_steps}"
                            )
                        for action in action_chunk[: int(args.replan_steps)]:
                            action_plan.append(np.asarray(action, dtype=np.float32))
                        policy_calls += 1

                    action = np.asarray(action_plan.popleft(), dtype=np.float32)
                    obs, reward, done, truncated, info = env.step(action)
                    final_info = dict(info)
                    episode_return += float(reward)
                    episode_steps += 1
                    success = bool(
                        success
                        or final_info.get("success", False)
                        or final_info.get("env_done", False)
                    )
                    if bool(done) or bool(truncated):
                        break

                if success:
                    successes += 1
                total_policy_calls += int(policy_calls)
                total_env_steps += int(episode_steps)
                record = _episode_record(
                    episode_idx=episode_idx,
                    init_state_idx=int(init_state_idx),
                    success=bool(success),
                    episode_return=float(episode_return),
                    episode_steps=int(episode_steps),
                    policy_calls=int(policy_calls),
                    elapsed_sec=float(time.time() - episode_start),
                    final_info=final_info,
                )
                records.append(record)
                episode_file.write(json.dumps(record, sort_keys=True) + "\n")
                episode_file.flush()
                logging.info(
                    "episode=%s init_state_idx=%s success=%s steps=%s running_success=%.3f",
                    episode_idx,
                    init_state_idx,
                    success,
                    episode_steps,
                    successes / float(episode_idx + 1),
                )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
        close_client = getattr(client, "close", None)
        if callable(close_client):
            close_client()

    summary = {
        "suite_name": str(args.task_suite_name),
        "task_id": int(args.task_id),
        "task_description": task_prompt,
        "env_backend": str(args.env_backend),
        "policy_host": str(args.policy_host),
        "policy_port": int(args.policy_port),
        "seed": int(args.seed),
        "start_episode_idx": int(args.start_episode_idx),
        "episodes": int(args.num_trials),
        "successes": int(successes),
        "success_rate": float(successes / max(1, int(args.num_trials))),
        "max_steps": int(max_steps),
        "num_steps_wait": int(args.num_steps_wait),
        "resize_size": int(args.resize_size),
        "replan_steps": int(args.replan_steps),
        "total_policy_calls": int(total_policy_calls),
        "total_env_steps": int(total_env_steps),
        "elapsed_sec": float(time.time() - started_at),
        "episode_log_path": str(episode_log_path),
        "fail_init_state_indices": [
            int(record["init_state_idx"]) for record in records if not bool(record["success"])
        ],
        "success_init_state_indices": [
            int(record["init_state_idx"]) for record in records if bool(record["success"])
        ],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    logging.info("Summary: %s", json.dumps(summary, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--start-episode-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--env-backend", choices=("local", "remote"), required=True)
    parser.add_argument("--remote-env-host", default="127.0.0.1")
    parser.add_argument("--remote-env-port", type=int, default=30000)
    parser.add_argument("--remote-env-timeout-sec", type=float, default=180.0)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--resolution", type=int, default=LIBERO_ENV_RESOLUTION)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--libero-root", default=None)
    parser.add_argument("--libero-config-dir", default=None)
    parser.add_argument("--libero-datasets-root", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
