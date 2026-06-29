from __future__ import annotations

"""Reference-style LIBERO residual DRQ training script."""

from collections import deque
import json
import logging
import signal
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from agentlace.data.data_store import QueuedDataStore
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm.auto import tqdm

from residual_sac.agents.continuous.drq_typed_config import (
    create_drq_agent_from_typed_cfg,
)
from residual_sac.common.agent_acceleration import apply_torch_compile
from residual_sac.common.checkpoint_codec import apply_checkpoint_payload_to_agent
from residual_sac.common.checkpoint_codec import snapshot_actor_network_payload
from residual_sac.common.checkpoint_codec import snapshot_agent_checkpoint_payload
from residual_sac.common.trainer_transport import build_actor_trainer_transport
from residual_sac.common.trainer_transport import build_learner_trainer_transport
from residual_sac.async_eval import append_async_eval_checkpoint_index
from residual_sac.async_eval import save_async_eval_checkpoint_payload
from residual_sac.common.training_payloads import build_rollout_payload
from residual_sac.common.training_payloads import build_rollout_stats_payload
from residual_sac.common.training_payloads import parse_rollout_stats_payload
from residual_sac.common.training_reporting import format_learner_heartbeat
from residual_sac.common.wandb import WandBLogger
from residual_sac.policy.typed_factory import build_policy_client
from residual_sac.policy.typed_factory import describe_policy_backend
from residual_sac.residual.chunk_window_replay import (
    create_chunk_replay_buffer,
    create_chunk_transition_replay_buffer,
)
from residual_sac.residual.chunk_window_replay import (
    PreparedStepWindowReplayBufferSampler,
)
from residual_sac.residual.chunk_window_replay import PrefetchingMixedBatchSampler
from residual_sac.residual.chunk_window_replay import ProfileAccumulator
from residual_sac.residual.action_metrics import ResidualActionStatsAccumulator
from residual_sac.residual.observation import build_chunk_residual_obs
from residual_sac.residual.observation import build_chunk_residual_observation_space
from residual_sac.residual.observation import build_chunk_residual_sample_obs
from residual_sac.residual.observation import prepare_base_actions_chunk
from residual_sac.residual.typed_action import ResidualActionSpec
from residual_sac.utils.checkpoint_utils import save_agent_checkpoint
from residual_sac.utils.jsonl import append_jsonl
from residual_sac.utils.seeding import set_global_seeds
from residual_sac.utils.serialization import to_jsonable
from residual_sac.utils.timer_utils import Timer

from examples.libero.residual_sac.config import LiberoTrainConfig
from examples.libero.residual_sac.config import cfg_to_log_payload
from examples.libero.residual_sac.config import parse_train_cfg
from examples.libero.residual_sac.env.factory import create_env
from examples.libero.residual_sac.env.observation import build_libero_state
from examples.libero.residual_sac.env.observation import extract_libero_images
from examples.libero.residual_sac.env.observation import LIBERO_STATE_DIM
from examples.libero.residual_sac.env.observation import RESIDUAL_IMAGE_HEIGHT
from examples.libero.residual_sac.env.observation import RESIDUAL_IMAGE_WIDTH
from examples.libero.residual_sac.env.policy_input import build_libero_policy_input
from examples.libero.residual_sac.env.offline_data import load_prepared_offline_replay
from examples.libero.residual_sac.env.offline_data import (
    resolve_and_validate_prepared_paths,
)
from examples.libero.residual_sac.metrics import eval_metric_aliases
from examples.libero.residual_sac.metrics import learner_metric_aliases
from examples.libero.residual_sac.metrics import rollout_metric_aliases
from examples.libero.residual_sac.runtime.raw_rollout_recorder import RawRolloutRecorder
from examples.libero.residual_sac.runtime.reward_relabel import build_reward_relabeler
from examples.libero.residual_sac.runtime.async_eval_runtime import (
    append_async_eval_request,
)
from examples.libero.residual_sac.runtime.async_eval_runtime import append_async_eval_stop
from examples.libero.residual_sac.runtime.async_eval_runtime import (
    check_async_eval_worker,
)
from examples.libero.residual_sac.runtime.async_eval_runtime import (
    load_new_async_eval_results,
)
from examples.libero.residual_sac.runtime.async_eval_runtime import (
    start_async_eval_worker,
)
from examples.libero.residual_sac.runtime.async_eval_runtime import (
    summarize_async_eval_results,
)
from examples.libero.residual_sac.runtime.async_eval_runtime import (
    wait_for_async_eval_worker,
)
from examples.libero.residual_sac.runtime.learner_shutdown import (
    ACTOR_DONE_DATA_COMMITTED_STOP_REASON,
)
from examples.libero.residual_sac.runtime.learner_shutdown import (
    actor_done_data_committed,
)
from examples.libero.residual_sac.runtime.transition_assembly import (
    AssemblyResult,
)
from examples.libero.residual_sac.runtime.transition_assembly import (
    ChunkExecutionRecord,
)
from examples.libero.residual_sac.runtime.transition_assembly import (
    LiberoActorTransitionAssembler,
)
from examples.libero.residual_sac.runtime.transition_assembly import (
    PrefetchedDecisionObs,
)
from examples.libero.residual_sac.runtime.transition_assembly import (
    infer_chunk_residual_obs,
)


def actor(
    cfg: LiberoTrainConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
) -> None:
    env = create_env(cfg, logger)
    task_prompt = str(env.task_description)
    policy_client = build_policy_client(cfg, logger=logger)
    policy_backend = describe_policy_backend(cfg)
    logger.info("Chunk policy backend: %s", policy_backend)

    image_keys = cfg.obs.image_keys
    action_dim = cfg.env.action_dim
    chunk_horizon = cfg.residual.chunk_horizon
    residual_alpha = float(cfg.residual.alpha)
    residual_action_spec = ResidualActionSpec.from_cfg(cfg, action_dim=action_dim)
    transition_assembler = LiberoActorTransitionAssembler(
        cfg=cfg,
        policy_client=policy_client,
        logger=logger,
    )
    reward_relabeler = build_reward_relabeler(cfg.reward, logger=logger)
    # Optimized actor dataflow:
    # chunk execute -> post-hoc transition assembly. Standard residual_sac
    # configs store one chunk-level transition per executed action chunk.
    if transition_assembler.async_transition_assembly_enabled:
        logger.info(
            "Async backfill enabled: mode=%s endpoint=%s:%s max_pending_chunks=%s",
            str(cfg.backfill_policy.mode),
            str(cfg.backfill_policy.host),
            int(cfg.backfill_policy.port),
            int(cfg.backfill_policy.max_pending_chunks),
        )

    sample_obs = build_chunk_residual_sample_obs(
        state_dim=LIBERO_STATE_DIM,
        action_dim=action_dim,
        chunk_horizon=chunk_horizon,
        image_keys=image_keys,
        image_height=RESIDUAL_IMAGE_HEIGHT,
        image_width=RESIDUAL_IMAGE_WIDTH,
    )

    agent = create_drq_agent_from_typed_cfg(
        cfg,
        sample_obs=sample_obs,
        action_dim=residual_action_spec.chunk_policy_action_dim,
        image_keys=image_keys,
        critic_action_dim=residual_action_spec.chunk_critic_action_dim,
        action_transform=residual_action_spec.build_chunk_action_transform(),
    )

    data_store = QueuedDataStore(cfg.runtime.data_store_queue_size)

    client = build_actor_trainer_transport(
        store_name="actor_env",
        server_ip=cfg.runtime.trainer_host,
        trainer_port=cfg.runtime.trainer_port,
        broadcast_port=cfg.runtime.broadcast_port,
        transport_cfg=cfg.runtime.trainer_transport,
        data_store=data_store,
        request_types=("send-stats",),
        wait_for_server=True,
        log_level=logger.level,
    )

    def update_actor(payload: dict[str, Any]) -> None:
        apply_checkpoint_payload_to_agent(
            agent,
            dict(payload),
            load_optimizers=False,
        )

    client.recv_network_callback(update_actor)

    timer = Timer()
    steps_per_update = cfg.training.steps_per_update
    log_period = cfg.training.log_period
    max_env_steps = cfg.training.max_env_steps
    env_seed = cfg.env.seed

    env_steps = 0
    committed_env_steps = 0
    episode_id = 0
    success_count = 0
    current_task_prompt: str | None = None
    recent_episode_successes_50: deque[int] = deque(maxlen=50)
    actor_timer_log_path = run_dir / "actor_timers.jsonl"
    rollout_log_path = run_dir / str(
        cfg.logging.episode_log_file or "episode_logs.jsonl"
    )
    summary: dict[str, Any] = {
        "role": "actor",
        "mode": "residual",
        "transport_mode": str(cfg.runtime.trainer_transport.mode),
        "env_steps": 0,
        "episodes": 0,
        "successes": 0,
        "timer_log_path": str(actor_timer_log_path),
        "episode_log_path": str(rollout_log_path),
        "reward": reward_relabeler.status_snapshot(),
    }
    recycle_output_root = Path(cfg.recycle.output_root)
    if not recycle_output_root.is_absolute():
        recycle_output_root = (run_dir / recycle_output_root).resolve()
    recycle_recorder: RawRolloutRecorder | None = None
    if cfg.recycle.enabled:
        recycle_recorder = RawRolloutRecorder(
            output_root=recycle_output_root,
            logger=logger,
        )
        logger.info(
            "raw rollout recycle enabled: output_root=%s",
            str(recycle_output_root),
        )

    def _transport_status() -> dict[str, Any]:
        try:
            return dict(client.get_transport_status("actor_env"))
        except Exception:  # noqa: BLE001
            return {"transport_mode": str(cfg.runtime.trainer_transport.mode)}

    consecutive_update_failures = 0
    consecutive_stats_failures = 0

    def _update_trainer_transport(*, context: str) -> bool:
        nonlocal consecutive_update_failures
        ok = bool(client.update())
        if ok:
            if int(consecutive_update_failures) > 0:
                logger.info(
                    "trainer transport update recovered: context=%s "
                    "consecutive_failures=%s status=%s",
                    str(context),
                    int(consecutive_update_failures),
                    _transport_status(),
                )
            consecutive_update_failures = 0
            return True

        consecutive_update_failures += 1
        logger.warning(
            "trainer transport update skipped after best-effort attempt: "
            "context=%s consecutive_failures=%s status=%s",
            str(context),
            int(consecutive_update_failures),
            _transport_status(),
        )
        # Keep scripts2 aligned with the original SERL actor loop: update() is
        # a best-effort datastore flush, not a training-liveness condition.  A
        # missed ack is retried on the next cadence from the learner's
        # last_update_id, so aborting here is more harmful than continuing.
        return False

    def _send_rollout_stats(*, payload: dict[str, Any]) -> None:
        nonlocal consecutive_stats_failures
        response = client.request("send-stats", payload)
        if response is not None:
            consecutive_stats_failures = 0
            return
        consecutive_stats_failures += 1
        logger.warning(
            "trainer transport send-stats failed: consecutive_failures=%s status=%s",
            int(consecutive_stats_failures),
            _transport_status(),
        )

    wait_for_episode_commit = bool(
        cfg.runtime.trainer_transport.wait_committed_on_episode_end
    )
    progress_bar = tqdm(
        total=int(max_env_steps),
        desc="actor env_steps",
        dynamic_ncols=True,
        leave=True,
    )

    def _commit_assembled_chunks(assembled_chunks: list[AssemblyResult]) -> None:
        nonlocal committed_env_steps
        for assembled_chunk in assembled_chunks:
            for transition in assembled_chunk.transitions:
                data_store.insert(transition)
            for step_offset in range(1, assembled_chunk.env_steps_delta + 1):
                next_committed_env_step = int(committed_env_steps + step_offset)
                if next_committed_env_step % steps_per_update == 0:
                    _update_trainer_transport(
                        context=f"commit_step_{int(next_committed_env_step)}"
                    )
            committed_env_steps += int(assembled_chunk.env_steps_delta)

    try:
        while env_steps < max_env_steps:
            episode_id += 1
            reset_seed = env_seed
            init_episode_idx = episode_id - 1
            obs = env.reset(seed=reset_seed, init_episode_idx=init_episode_idx)
            reward_relabeler.start_episode(
                episode_id=int(episode_id),
                task_prompt=str(task_prompt),
                initial_obs=obs,
                init_episode_idx=int(init_episode_idx),
                task_id=int(cfg.task.task_id),
            )
            current_task_prompt = str(task_prompt)
            prefetched = None
            episode_return = 0.0
            episode_env_return = 0.0
            episode_steps = 0
            episode_success = False
            last_info: dict[str, Any] = {}
            episode_chunk_seq = 0
            episode_residual_stats = ResidualActionStatsAccumulator()

            while env_steps < max_env_steps:
                if transition_assembler.async_transition_assembly_enabled:
                    with timer.context("commit_replay"):
                        _commit_assembled_chunks(transition_assembler.drain_ready())

                timer.tick("total")
                with timer.context("prepare_action_chunk"):
                    if prefetched is None:
                        robot_state = build_libero_state(obs)
                        image_observations = extract_libero_images(obs)
                        base_policy_input = build_libero_policy_input(
                            prompt=task_prompt,
                            state=robot_state,
                            images=image_observations,
                        )
                        base_actions, _ = policy_client.infer(base_policy_input)
                        base_actions = prepare_base_actions_chunk(
                            base_actions=base_actions,
                            chunk_horizon=chunk_horizon,
                        )
                        residual_obs = build_chunk_residual_obs(
                            robot_state=robot_state,
                            images=image_observations,
                            image_keys=image_keys,
                            base_actions=base_actions,
                            residual_alpha=residual_alpha,
                        )
                    else:
                        base_actions = prefetched.base_actions
                        residual_obs = prefetched.residual_obs
                        prefetched = None

                    residual_actions = agent.sample_action(
                        residual_obs,
                        deterministic=False,
                    )

                    final_actions = residual_action_spec.compose_chunk(
                        base_action_chunk=base_actions,
                        residual_action=residual_actions,
                    )

                episode_done = False
                should_log_timer = False
                remaining_env_steps = max(0, int(max_env_steps - env_steps))
                if remaining_env_steps <= 0:
                    timer.tock("total")
                    break

                action_chunk = np.asarray(final_actions, dtype=np.float32)[
                    :remaining_env_steps
                ]

                with timer.context("step_env"):
                    chunk_result = env.step_chunk(action_chunk)

                with timer.context("reward_relabel"):
                    chunk_result = reward_relabeler.relabel_chunk(
                        chunk_result,
                        episode_step_start=int(episode_steps),
                    )

                next_chunk_prefetched = None
                residual_obs_after_chunk = None
                if str(cfg.replay.transition_granularity) == "chunk" and not (
                    bool(chunk_result["done"]) or bool(chunk_result["truncated"])
                ):
                    with timer.context("prepare_next_chunk_obs"):
                        (
                            next_base_actions,
                            residual_obs_after_chunk,
                        ) = infer_chunk_residual_obs(
                            obs=dict(chunk_result["obs"]),
                            task_prompt=task_prompt,
                            policy_client=policy_client,
                            chunk_horizon=chunk_horizon,
                            image_keys=image_keys,
                            residual_alpha=residual_alpha,
                        )
                    next_chunk_prefetched = PrefetchedDecisionObs(
                        base_actions=next_base_actions,
                        residual_obs=residual_obs_after_chunk,
                    )

                raw_chunk = ChunkExecutionRecord.from_env_chunk_result(
                    episode_id=int(episode_id),
                    episode_step_start=int(episode_steps),
                    residual_obs_before_chunk=residual_obs,
                    action_chunk=action_chunk,
                    chunk_result=chunk_result,
                    residual_obs_after_chunk=residual_obs_after_chunk,
                )
                if recycle_recorder is not None:
                    try:
                        recycle_recorder.append_chunk(
                            payload={
                                "episode_id": int(episode_id),
                                "chunk_seq": int(episode_chunk_seq),
                                "episode_step_start": int(episode_steps),
                                "task_prompt": str(task_prompt),
                                "chunk_result": dict(chunk_result),
                            }
                        )
                    except Exception:
                        recycle_recorder.record_append_error()
                        logger.exception(
                            "raw rollout recycle append failed: episode_id=%s chunk_seq=%s",
                            int(episode_id),
                            int(episode_chunk_seq),
                        )
                episode_chunk_seq += 1
                previous_env_steps = int(env_steps)
                with timer.context("assemble_transitions"):
                    assembled_chunks = transition_assembler.handle_chunk(
                        raw=raw_chunk,
                        task_prompt=task_prompt,
                    )
                if transition_assembler.async_transition_assembly_enabled:
                    prefetched = None
                elif assembled_chunks:
                    prefetched = assembled_chunks[-1].prefetched
                    if prefetched is None and next_chunk_prefetched is not None:
                        prefetched = next_chunk_prefetched

                env_steps += int(raw_chunk.executed_steps)
                residual_chunk = np.asarray(residual_actions, dtype=np.float32).reshape(
                    int(chunk_horizon),
                    -1,
                )
                for executed_idx in range(int(raw_chunk.executed_steps)):
                    episode_residual_stats.add(
                        residual_action=residual_chunk[int(executed_idx)],
                        base_action=np.asarray(base_actions, dtype=np.float32)[
                            int(executed_idx)
                        ],
                        final_action=np.asarray(action_chunk, dtype=np.float32)[
                            int(executed_idx)
                        ],
                    )
                progress_bar.update(int(raw_chunk.executed_steps))
                episode_steps += int(raw_chunk.executed_steps)
                episode_return += float(raw_chunk.reward_sum)
                episode_env_return += float(
                    chunk_result.get("env_reward_sum", raw_chunk.reward_sum)
                )
                episode_success = bool(
                    episode_success
                    or any(
                        bool(info.get("env_done", False)) for info in raw_chunk.infos
                    )
                )
                last_info = dict(raw_chunk.chunk_info)
                obs = dict(raw_chunk.final_obs)
                episode_done = bool(raw_chunk.chunk_done or raw_chunk.chunk_truncated)

                for step_offset in range(1, int(raw_chunk.executed_steps) + 1):
                    next_env_step = int(previous_env_steps + step_offset)
                    if next_env_step % log_period == 0:
                        should_log_timer = True

                if transition_assembler.async_transition_assembly_enabled:
                    with timer.context("commit_replay"):
                        _commit_assembled_chunks(assembled_chunks)
                else:
                    _commit_assembled_chunks(assembled_chunks)

                timer.tock("total")

                if should_log_timer:
                    append_jsonl(
                        actor_timer_log_path,
                        {
                            "source": "actor",
                            "env_steps": int(env_steps),
                            "episode_id": int(episode_id),
                            "episode_steps": int(episode_steps),
                            "timer": timer.get_average_times(),
                            "transport": _transport_status(),
                        },
                    )

                if episode_done:
                    break

            if transition_assembler.async_transition_assembly_enabled:
                with timer.context("commit_replay"):
                    _commit_assembled_chunks(
                        transition_assembler.finish_episode(
                            block=bool(wait_for_episode_commit),
                        )
                    )
            _update_trainer_transport(context="episode_end")
            success_count += int(episode_success)
            recent_episode_successes_50.append(int(episode_success))
            recent_success_rate_50 = float(sum(recent_episode_successes_50)) / float(
                max(1, len(recent_episode_successes_50))
            )
            episode_stats = build_rollout_stats_payload(
                env_steps=int(env_steps),
                rollout=build_rollout_payload(
                    episode_id=int(episode_id),
                    episode_steps=int(episode_steps),
                    episode_return=float(episode_return),
                    init_episode_idx=int(init_episode_idx),
                    success=bool(episode_success),
                    cumulative_success_rate=float(success_count / max(1, episode_id)),
                    recent_success_rate_50=float(recent_success_rate_50),
                ),
                env_info=last_info,
                residual=episode_residual_stats.summary(),
            )
            episode_stats["reward"] = {
                **reward_relabeler.status_snapshot(),
                "episode_env_return": float(episode_env_return),
                "episode_train_return": float(episode_return),
            }

            append_jsonl(
                rollout_log_path,
                {
                    "source": "rollout",
                    **episode_stats,
                    "transport": _transport_status(),
                },
            )
            if wait_for_episode_commit:
                client.wait_until_committed()
            _send_rollout_stats(payload=episode_stats)
            if recycle_recorder is not None:
                try:
                    recycle_recorder.finalize_episode(
                        marker={
                            "episode_id": int(episode_id),
                            "rollout_stats": dict(episode_stats),
                        }
                    )
                except Exception:
                    logger.exception(
                        "raw rollout recycle finalize failed: episode_id=%s",
                        int(episode_id),
                    )
            progress_bar.set_postfix(
                episode=int(episode_id),
                success=int(bool(episode_success)),
                refresh=False,
            )
            logger.info(
                "episode=%s success=%s steps=%s return=%.3f env_return=%.3f env_steps=%s reward=%s",
                int(episode_id),
                bool(episode_success),
                int(episode_steps),
                float(episode_return),
                float(episode_env_return),
                int(env_steps),
                str(cfg.reward.name),
            )
            reward_relabeler.finish_episode()

    finally:
        try:
            if current_task_prompt is not None:
                assembled_chunks = transition_assembler.finish_episode(
                    block=True,
                )
                if assembled_chunks:
                    with timer.context("commit_replay"):
                        _commit_assembled_chunks(assembled_chunks)
            _update_trainer_transport(context="shutdown")
            if bool(cfg.runtime.trainer_transport.wait_committed_on_shutdown):
                client.wait_until_committed()
        except Exception:  # noqa: BLE001
            pass
        pending_recycle_episodes = 0
        recycle_status: dict[str, Any] = {
            "enabled": bool(cfg.recycle.enabled),
            "output_root": str(recycle_output_root),
            "episodes_written": 0,
            "steps_written": 0,
            "append_errors": 0,
            "write_errors": 0,
            "pending_episodes": 0,
        }
        if recycle_recorder is not None:
            try:
                pending_recycle_episodes = recycle_recorder.discard_pending()
                recycle_status = dict(recycle_recorder.status_snapshot())
                recycle_status["pending_episodes"] = int(pending_recycle_episodes)
                logger.info(
                    "raw rollout recycle summary: output_root=%s episodes_written=%s "
                    "steps_written=%s append_errors=%s write_errors=%s pending_episodes=%s",
                    str(recycle_status["output_root"]),
                    int(recycle_status["episodes_written"]),
                    int(recycle_status["steps_written"]),
                    int(recycle_status["append_errors"]),
                    int(recycle_status["write_errors"]),
                    int(recycle_status["pending_episodes"]),
                )
            except Exception:  # noqa: BLE001
                logger.exception("raw rollout recycle cleanup failed")
                recycle_status["write_errors"] = (
                    int(recycle_status.get("write_errors", 0)) + 1
                )
        summary.update(
            {
                "env_steps": int(env_steps),
                "episodes": int(episode_id),
                "successes": int(success_count),
                "transport": _transport_status(),
                "recycle_enabled": bool(recycle_status.get("enabled", False)),
                "recycle_output_root": recycle_status.get("output_root", None),
                "recycle_episodes_written": int(
                    recycle_status.get("episodes_written", 0)
                ),
                "recycle_steps_written": int(recycle_status.get("steps_written", 0)),
                "recycle_append_errors": int(recycle_status.get("append_errors", 0)),
                "recycle_write_errors": int(recycle_status.get("write_errors", 0)),
                "recycle": recycle_status,
                "reward": reward_relabeler.status_snapshot(),
            }
        )
        with open(run_dir / cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        try:
            client.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            progress_bar.close()
        except Exception:  # noqa: BLE001
            pass
        if str(cfg.env.backend) != "remote":
            try:
                env.close(clear_cache=False)
            except Exception:  # noqa: BLE001
                pass
        policy_client_close = getattr(policy_client, "close", None)
        if callable(policy_client_close):
            try:
                policy_client_close()
            except Exception:  # noqa: BLE001
                pass
        transition_assembler.close()
        reward_relabeler.close()


def learner(
    cfg: LiberoTrainConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
) -> None:
    def _request_graceful_shutdown(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _request_graceful_shutdown)

    image_keys = cfg.obs.image_keys
    action_dim = cfg.env.action_dim
    chunk_horizon = cfg.residual.chunk_horizon
    residual_action_spec = ResidualActionSpec.from_cfg(
        cfg,
        action_dim=action_dim,
    )
    sample_obs = build_chunk_residual_sample_obs(
        state_dim=LIBERO_STATE_DIM,
        action_dim=action_dim,
        chunk_horizon=chunk_horizon,
        image_keys=image_keys,
        image_height=RESIDUAL_IMAGE_HEIGHT,
        image_width=RESIDUAL_IMAGE_WIDTH,
    )
    agent = create_drq_agent_from_typed_cfg(
        cfg,
        sample_obs=sample_obs,
        action_dim=residual_action_spec.chunk_policy_action_dim,
        image_keys=image_keys,
        critic_action_dim=residual_action_spec.chunk_critic_action_dim,
        action_transform=residual_action_spec.build_chunk_action_transform(),
    )
    agent = apply_torch_compile(
        agent,
        compile_cfg=cfg.training.torch_compile,
    )
    observation_space = build_chunk_residual_observation_space(
        sample_obs=sample_obs,
        image_keys=image_keys,
    )
    if str(cfg.replay.transition_granularity) == "chunk":
        replay_buffer = create_chunk_transition_replay_buffer(
            observation_space=observation_space,
            action_dim=int(cfg.env.action_dim),
            chunk_horizon=int(cfg.residual.chunk_horizon),
            image_keys=image_keys,
            capacity=int(cfg.replay.capacity),
        )
    else:
        replay_buffer = create_chunk_replay_buffer(
            observation_space=observation_space,
            action_dim=int(cfg.env.action_dim),
            chunk_horizon=int(cfg.residual.chunk_horizon),
            discount=float(cfg.sac.discount),
            image_keys=image_keys,
            capacity=int(cfg.replay.capacity),
        )
    if bool(cfg.replay.prepared_chunk.online_enabled):
        raise NotImplementedError(
            "replay.prepared_chunk.online_enabled is not implemented yet. "
            "Use replay.prepared_chunk.offline_enabled for the immutable offline replay."
        )
    offline_replay_buffer: Any | None = None
    offline_prepared_chunk_profile: dict[str, float] | None = None
    offline_prepared_path: Path | None = None
    offline_manifest_path: Path | None = None
    offline_validation_stats: dict[str, Any] | None = None
    offline_load_stats: dict[str, Any] = {
        "files_total": 0,
        "episodes_loaded": 0,
        "steps_loaded": 0,
        "load_errors": 0,
    }
    wandb_cfg = WandBLogger.get_default_config()
    run_name = cfg.wandb.exp_name
    wandb_cfg.update(
        {
            "project": cfg.wandb.project,
            "entity": cfg.wandb.entity,
            "exp_descriptor": run_name,
            "tag": [run_name],
            "group": cfg.wandb.group,
            "mode": cfg.wandb.mode,
        }
    )
    wandb_variant = cfg_to_log_payload(cfg)
    swanlab_dir = run_dir / "swanlab"
    swanlab_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = WandBLogger(
        wandb_config=wandb_cfg,
        variant=wandb_variant,
        wandb_output_dir=str(swanlab_dir),
        mode=cfg.wandb.mode,
    )
    async_eval = start_async_eval_worker(
        cfg,
        run_dir=run_dir,
        logger=logger,
    )

    update_steps = 0
    env_steps = 0
    latest_completed_episode_id = 0
    completed_episode_env_steps: dict[int, int] = {}
    last_queued_async_eval_episode = 0
    last_rollout_wall_time: float | None = None
    last_rollout_env_steps = 0
    learner_timer_log_path = run_dir / "learner_timers.jsonl"
    metrics_log_path = run_dir / "metrics.jsonl"
    metrics_log_path.write_text("", encoding="utf-8")
    progress_state_lock = Lock()
    summary: dict[str, Any] = {
        "role": "learner",
        "mode": "residual",
        "transport_mode": str(cfg.runtime.trainer_transport.mode),
        "update_steps": 0,
        "env_steps": 0,
        "replay_size": 0,
        "timer_log_path": str(learner_timer_log_path),
        "stop_reason": None,
    }
    stop_reason = ACTOR_DONE_DATA_COMMITTED_STOP_REASON

    def _transport_status() -> dict[str, Any]:
        try:
            return dict(server.get_transport_status("actor_env"))
        except Exception:  # noqa: BLE001
            return {"transport_mode": str(cfg.runtime.trainer_transport.mode)}

    def _committed_online_steps() -> int:
        return int(replay_buffer.latest_data_id())

    def _async_eval_backlog() -> int:
        return max(
            0,
            int(async_eval.triggered_count) - int(async_eval.processed_summary_lines),
        )

    def _write_metric_record(record: dict[str, Any]) -> None:
        append_jsonl(metrics_log_path, to_jsonable(record))

    def _log_eval_records(records: list[dict[str, Any]]) -> None:
        for eval_record in records:
            eval_metrics = eval_metric_aliases(
                eval_record,
                eval_queue_backlog=_async_eval_backlog(),
            )
            if not eval_metrics:
                continue
            _write_metric_record({"role": "eval", **eval_metrics})
            wandb_logger.log(
                to_jsonable(eval_metrics),
                step=int(eval_metrics["eval/train_episode"]),
            )
            summary_payload = eval_record.get("summary", None)
            if str(eval_record.get("status", "")).lower() == "ok":
                logger.info(
                    "eval done: eval_index=%s episode=%s update_steps=%s env_steps=%s success_rate=%s",
                    eval_record.get("eval_index", None),
                    eval_record.get("train_episode_id", None),
                    eval_record.get("train_update_step", None),
                    eval_record.get("train_env_step", None),
                    summary_payload.get("success_rate", None)
                    if isinstance(summary_payload, dict)
                    else None,
                )
            else:
                logger.warning(
                    "eval failed: eval_index=%s episode=%s update_steps=%s error=%s",
                    eval_record.get("eval_index", None),
                    eval_record.get("train_episode_id", None),
                    eval_record.get("train_update_step", None),
                    eval_record.get("error", None),
                )

    def _should_stop_after_actor_done() -> bool:
        target_env_steps = int(cfg.training.max_env_steps)
        return actor_done_data_committed(
            actor_done=int(env_steps) >= target_env_steps,
            target_env_steps=target_env_steps,
            latest_data_id=int(_committed_online_steps()),
            transport_status=_transport_status(),
            require_transport_commit=True,
        )

    def stats_callback(request_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal env_steps
        nonlocal latest_completed_episode_id
        nonlocal last_rollout_wall_time
        nonlocal last_rollout_env_steps
        if request_type != "send-stats":
            raise ValueError(f"Invalid request type: {request_type}")
        rollout_stats = parse_rollout_stats_payload(payload)
        if rollout_stats is None:
            logger.warning(
                "ignore malformed rollout stats payload: keys=%s",
                sorted(payload.keys()),
            )
            return {}
        rollout_env_steps = int(rollout_stats["env_steps"])
        now = time.time()
        actor_env_steps_per_sec: float | None = None
        if last_rollout_wall_time is not None:
            elapsed_sec = max(float(now - last_rollout_wall_time), 1e-6)
            actor_env_steps_per_sec = float(
                max(0, int(rollout_env_steps - last_rollout_env_steps))
            ) / elapsed_sec
        last_rollout_wall_time = float(now)
        last_rollout_env_steps = int(rollout_env_steps)
        with progress_state_lock:
            env_steps = max(int(env_steps), int(rollout_env_steps))
            episode_id = int(rollout_stats["rollout"]["episode_id"])
            if episode_id > 0:
                latest_completed_episode_id = max(
                    int(latest_completed_episode_id),
                    int(episode_id),
                )
                completed_episode_env_steps[int(episode_id)] = int(env_steps)
        rollout_metrics = rollout_metric_aliases(
            rollout_stats,
            actor_env_steps_per_sec=actor_env_steps_per_sec,
        )
        if rollout_metrics:
            rollout_step = int(rollout_metrics["rollout/episode_id"])
            _write_metric_record({"role": "actor", **rollout_metrics})
            wandb_logger.log(to_jsonable(rollout_metrics), step=rollout_step)
        return {}

    server = build_learner_trainer_transport(
        trainer_port=cfg.runtime.trainer_port,
        broadcast_port=cfg.runtime.broadcast_port,
        transport_cfg=cfg.runtime.trainer_transport,
        request_callback=stats_callback,
        request_types=("send-stats",),
        log_level=logger.level,
    )
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    if cfg.offline.enabled:
        offline_resolution = resolve_and_validate_prepared_paths(
            cfg,
            logger=logger,
        )
        if offline_resolution.prepared_paths:
            offline_prepared_path = offline_resolution.prepared_paths[0]
        if offline_resolution.manifest_paths:
            offline_manifest_path = offline_resolution.manifest_paths[0]
        offline_validation_stats = dict(offline_resolution.validation_stats)
        offline_replay_buffer = create_chunk_replay_buffer(
            observation_space=observation_space,
            action_dim=int(cfg.env.action_dim),
            chunk_horizon=int(cfg.residual.chunk_horizon),
            discount=float(cfg.sac.discount),
            image_keys=image_keys,
            capacity=int(cfg.offline.capacity),
        )
        offline_load_stats = load_prepared_offline_replay(
            replay_buffer=offline_replay_buffer,
            prepared_paths=tuple()
            if offline_prepared_path is None
            else (offline_prepared_path,),
            logger=logger,
            max_episodes=cfg.offline.load_max_episodes,
            max_transitions=cfg.offline.load_max_transitions,
        )
        if len(offline_replay_buffer) <= 0:
            logger.warning(
                "offline replay prepared but empty; continuing with online-only training"
            )
            offline_replay_buffer = None
        else:
            if bool(cfg.replay.prepared_chunk.offline_enabled):
                prepare_start = time.perf_counter()
                offline_replay_buffer = PreparedStepWindowReplayBufferSampler(
                    offline_replay_buffer,
                    name="offline",
                )
                offline_prepared_chunk_profile = dict(
                    offline_replay_buffer.prepare_profile
                )
                offline_prepared_chunk_profile.setdefault(
                    "prepare_window_sec",
                    float(time.perf_counter() - prepare_start),
                )
                logger.info(
                    "offline prepared chunk cache ready: replay_size=%s windows=%s "
                    "prepare_window_sec=%.6f action_cache_sec=%.6f scalar_cache_sec=%.6f",
                    int(len(offline_replay_buffer)),
                    int(offline_replay_buffer.num_windows),
                    float(
                        offline_prepared_chunk_profile.get("prepare_window_sec", 0.0)
                    ),
                    float(
                        offline_prepared_chunk_profile.get(
                            "prepare_action_cache_sec",
                            0.0,
                        )
                    ),
                    float(
                        offline_prepared_chunk_profile.get(
                            "prepare_scalar_cache_sec",
                            0.0,
                        )
                    ),
                )
            logger.info(
                "offline replay ready: prepared_path=%s replay_size=%s ratio=%.3f pretrain_steps=%s",
                None if offline_prepared_path is None else str(offline_prepared_path),
                int(len(offline_replay_buffer)),
                float(cfg.offline.ratio),
                int(cfg.offline.pretrain_steps),
            )

    training_starts = cfg.training.training_starts
    checkpoint_every = cfg.training.checkpoint.every_steps
    checkpoint_keep = cfg.training.checkpoint.keep
    checkpoint_dir = Path(cfg.training.checkpoint.dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = run_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_period = cfg.training.log_period
    critic_actor_ratio = max(1, cfg.training.critic_actor_ratio)
    steps_per_update = cfg.training.steps_per_update
    replay_warmup_poll_interval_sec = 1.0
    idle_poll_interval_sec = 1.0
    timer = Timer()
    last_log_time = time.time()
    last_log_update_steps = int(update_steps)
    offline_pretrain_steps_done = 0
    initial_network_published = False
    batch_sampler = PrefetchingMixedBatchSampler(
        online_replay_buffer=replay_buffer,
        offline_replay_buffer=offline_replay_buffer,
        batch_size=int(cfg.replay.batch_size),
        device=agent.device,
        pack_obs_and_next_obs=True,
        prefer_device_concat=True,
        thread_name_prefix="libero-learner-prefetch",
    )
    update_profile_accumulator = ProfileAccumulator()
    logger.info("learner replay batch prefetch enabled: depth=1")

    def _next_training_batch(
        *,
        offline_ratio: float,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        with timer.context("sample_replay_buffer"):
            batch, batch_mix = batch_sampler.next_batch(
                offline_ratio=float(offline_ratio)
            )
        return batch, batch_mix

    def _run_training_update(
        *,
        offline_ratio: float,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        nonlocal agent
        train_batch_mix = {
            "online_batch_size": int(cfg.replay.batch_size),
            "offline_batch_size": 0,
        }
        update_profile: dict[str, float] = {}
        for _ in range(max(0, critic_actor_ratio - 1)):
            batch, _ = _next_training_batch(offline_ratio=float(offline_ratio))
            with timer.context("train_critics"):
                agent, _critics_info = agent.update_critics(
                    batch,
                    profile=update_profile,
                )

        batch, train_batch_mix = _next_training_batch(
            offline_ratio=float(offline_ratio)
        )
        with timer.context("train"):
            agent, update_info = agent.update_high_utd(
                batch,
                utd_ratio=cfg.sac.utd_ratio,
                profile=update_profile,
            )
        update_profile_accumulator.record(update_profile)
        return update_info, train_batch_mix

    def _maybe_queue_async_eval() -> None:
        nonlocal last_queued_async_eval_episode
        if (not async_eval.enabled) or async_eval.eval_checkpoint_dir is None:
            return
        every_episodes = int(async_eval.every_episodes)
        if every_episodes <= 0:
            return
        while True:
            with progress_state_lock:
                next_target_episode = (
                    int(last_queued_async_eval_episode) + every_episodes
                )
                if int(latest_completed_episode_id) < int(next_target_episode):
                    return
                target_episode = int(next_target_episode)
                target_env_step = int(
                    completed_episode_env_steps.get(target_episode, env_steps)
                )
            async_eval_checkpoint_payload = snapshot_agent_checkpoint_payload(
                agent,
                step=int(update_steps),
            )
            async_eval_checkpoint_path = save_async_eval_checkpoint_payload(
                async_eval.eval_checkpoint_dir,
                async_eval_checkpoint_payload,
                episode_id=int(target_episode),
            )
            append_async_eval_checkpoint_index(
                async_eval.eval_checkpoint_dir,
                episode_id=int(target_episode),
                checkpoint_step=int(update_steps),
                checkpoint_path=async_eval_checkpoint_path,
            )
            append_async_eval_request(
                async_eval,
                {
                    "eval_index": int(async_eval.triggered_count),
                    "train_episode_id": int(target_episode),
                    "train_update_step": int(update_steps),
                    "train_env_step": int(target_env_step),
                    "checkpoint_step": int(update_steps),
                    "checkpoint_path": str(async_eval_checkpoint_path),
                },
            )
            with progress_state_lock:
                last_queued_async_eval_episode = max(
                    int(last_queued_async_eval_episode),
                    int(target_episode),
                )
                stale_episode_ids = [
                    episode_id
                    for episode_id in completed_episode_env_steps
                    if int(episode_id) <= int(target_episode)
                ]
                for stale_episode_id in stale_episode_ids:
                    completed_episode_env_steps.pop(int(stale_episode_id), None)
            logger.info(
                "queued eval: eval_index=%s episode=%s update_steps=%s env_steps=%s checkpoint=%s",
                int(max(0, async_eval.triggered_count - 1)),
                int(target_episode),
                int(update_steps),
                int(target_env_step),
                async_eval_checkpoint_path,
            )

    def _publish_actor_network(*, step: int, reason: str) -> None:
        with timer.context("publish_snapshot"):
            payload = snapshot_actor_network_payload(agent, step=int(step))
        with timer.context("publish_network"):
            server.publish_network(payload)
        logger.info(
            "publish network: step=%s env_steps=%s reason=%s",
            int(step),
            int(env_steps),
            str(reason),
        )

    if offline_replay_buffer is not None and int(cfg.offline.pretrain_steps) > 0:
        logger.info(
            "starting offline pretrain: steps=%s offline_replay_size=%s",
            int(cfg.offline.pretrain_steps),
            int(len(offline_replay_buffer)),
        )
        pretrain_bar = tqdm(
            total=int(cfg.offline.pretrain_steps),
            desc="learner offline pretrain",
            dynamic_ncols=True,
            leave=True,
        )
        last_pretrain_info: dict[str, Any] | None = None
        try:
            while int(offline_pretrain_steps_done) < int(cfg.offline.pretrain_steps):
                check_async_eval_worker(async_eval, logger=logger)
                last_pretrain_info, _ = _run_training_update(offline_ratio=1.0)
                update_steps += 1
                offline_pretrain_steps_done += 1
                check_async_eval_worker(async_eval, logger=logger)
                pretrain_bar.update(1)
        finally:
            pretrain_bar.close()

        if last_pretrain_info is not None:
            pretrain_metrics = learner_metric_aliases(
                last_pretrain_info,
                update_steps=int(update_steps),
                env_steps=int(env_steps),
                replay_size=int(len(replay_buffer)),
                eval_queue_backlog=_async_eval_backlog(),
            )
            if pretrain_metrics:
                _write_metric_record({"role": "learner", **pretrain_metrics})
                wandb_logger.log(to_jsonable(pretrain_metrics), step=update_steps)
        logger.info(
            "offline pretrain complete: completed=%s update_steps=%s offline_replay_size=%s",
            int(offline_pretrain_steps_done),
            int(update_steps),
            0 if offline_replay_buffer is None else int(len(offline_replay_buffer)),
        )
        _publish_actor_network(
            step=int(update_steps),
            reason="offline_pretrain_complete",
        )
        initial_network_published = True

    warmup_stopped_for_actor_done = False
    if int(training_starts) > 0:
        warmup_bar = tqdm(
            total=int(training_starts),
            initial=min(int(len(replay_buffer)), int(training_starts)),
            desc="learner replay warmup",
            dynamic_ncols=True,
            leave=True,
        )
        warmup_replay_size = min(int(len(replay_buffer)), int(training_starts))
        try:
            while len(replay_buffer) < training_starts:
                check_async_eval_worker(async_eval, logger=logger)
                current_replay_size = min(int(len(replay_buffer)), int(training_starts))
                if current_replay_size > warmup_replay_size:
                    warmup_bar.update(current_replay_size - warmup_replay_size)
                    warmup_replay_size = current_replay_size
                warmup_bar.set_postfix(
                    replay=int(len(replay_buffer)),
                    env_steps=int(env_steps),
                    refresh=False,
                )
                if _should_stop_after_actor_done():
                    warmup_stopped_for_actor_done = True
                    stop_reason = ACTOR_DONE_DATA_COMMITTED_STOP_REASON
                    logger.info(
                        "stopping replay warmup: reason=%s replay=%s training_starts=%s env_steps=%s",
                        stop_reason,
                        int(len(replay_buffer)),
                        int(training_starts),
                        int(env_steps),
                    )
                    break
                time.sleep(replay_warmup_poll_interval_sec)
            current_replay_size = min(int(len(replay_buffer)), int(training_starts))
            if current_replay_size > warmup_replay_size:
                warmup_bar.update(current_replay_size - warmup_replay_size)
        finally:
            warmup_bar.close()
        logger.info(
            "replay warmup complete: replay=%s training_starts=%s env_steps=%s actor_done=%s",
            int(len(replay_buffer)),
            int(training_starts),
            int(env_steps),
            bool(warmup_stopped_for_actor_done),
        )

    if not initial_network_published:
        _publish_actor_network(step=int(update_steps), reason="initial")

    interrupted = False
    try:
        while True:
            check_async_eval_worker(async_eval, logger=logger)
            _maybe_queue_async_eval()
            online_update_steps = max(
                0, int(update_steps - offline_pretrain_steps_done)
            )
            if _should_stop_after_actor_done():
                stop_reason = ACTOR_DONE_DATA_COMMITTED_STOP_REASON
                _maybe_queue_async_eval()
                logger.info(
                    "stopping learner: reason=%s update_steps=%s env_steps=%s "
                    "target_env_steps=%s replay_latest_data_id=%s replay_size=%s "
                    "transport=%s",
                    stop_reason,
                    int(update_steps),
                    int(env_steps),
                    int(cfg.training.max_env_steps),
                    int(_committed_online_steps()),
                    int(len(replay_buffer)),
                    _transport_status(),
                )
                break
            if not online_update_steps < _committed_online_steps():
                time.sleep(idle_poll_interval_sec)
                continue

            update_info, batch_mix = _run_training_update(
                offline_ratio=float(cfg.offline.ratio),
            )
            update_steps += 1
            check_async_eval_worker(async_eval, logger=logger)
            _maybe_queue_async_eval()

            if update_steps % steps_per_update == 0:
                _publish_actor_network(step=int(update_steps), reason="periodic")

            if update_steps % log_period == 0:
                check_async_eval_worker(async_eval, logger=logger)
                eval_records = load_new_async_eval_results(async_eval)
                _log_eval_records(eval_records)
                update_metrics = to_jsonable(update_info)
                now = time.time()
                elapsed_sec = max(now - last_log_time, 1e-6)
                updates_since_last_log = max(
                    1, int(update_steps - last_log_update_steps)
                )
                updates_per_sec = float(updates_since_last_log) / float(elapsed_sec)
                last_log_time = now
                last_log_update_steps = int(update_steps)
                timer_metrics = to_jsonable(timer.get_average_times())
                sample_profile_metrics = to_jsonable(
                    batch_sampler.drain_sample_profile()
                )
                update_profile_metrics = to_jsonable(update_profile_accumulator.drain())
                learner_metrics = learner_metric_aliases(
                    update_metrics,
                    update_steps=int(update_steps),
                    env_steps=int(env_steps),
                    replay_size=int(len(replay_buffer)),
                    updates_per_sec=float(updates_per_sec),
                    eval_queue_backlog=_async_eval_backlog(),
                    batch_mix=batch_mix,
                )
                if learner_metrics:
                    _write_metric_record({"role": "learner", **learner_metrics})
                    wandb_logger.log(to_jsonable(learner_metrics), step=update_steps)
                append_jsonl(
                    learner_timer_log_path,
                    {
                        "source": "learner",
                        "update_steps": int(update_steps),
                        "env_steps": int(env_steps),
                        "replay_size": int(len(replay_buffer)),
                        "updates_per_sec": float(updates_per_sec),
                        "timer": timer_metrics,
                        "sample_profile": sample_profile_metrics,
                        "update_profile": update_profile_metrics,
                        "offline_prepared_chunk": (
                            None
                            if offline_prepared_chunk_profile is None
                            else to_jsonable(offline_prepared_chunk_profile)
                        ),
                        "transport": _transport_status(),
                    },
                )
                offline_suffix = ""
                if offline_replay_buffer is not None:
                    offline_total = max(
                        1,
                        int(batch_mix["online_batch_size"])
                        + int(batch_mix["offline_batch_size"]),
                    )
                    offline_ratio_actual = float(
                        batch_mix["offline_batch_size"]
                    ) / float(offline_total)
                    offline_suffix = (
                        " "
                        f"offline_replay={int(len(offline_replay_buffer))} "
                        f"offline_ratio_actual={offline_ratio_actual:.3f} "
                        f"offline_batch={int(batch_mix['offline_batch_size'])}/{offline_total} "
                        f"online_updates={max(0, int(update_steps - offline_pretrain_steps_done))} "
                        f"offline_pretrain={int(offline_pretrain_steps_done)}"
                    )
                logger.info(
                    format_learner_heartbeat(
                        update_steps=int(update_steps),
                        env_steps=int(env_steps),
                        replay_size=int(len(replay_buffer)),
                        updates_per_sec=float(updates_per_sec),
                        update_info=dict(update_metrics),
                        offline_suffix=offline_suffix,
                    )
                )

            if checkpoint_every > 0 and update_steps % checkpoint_every == 0:
                checkpoint_path = save_agent_checkpoint(
                    checkpoint_dir,
                    agent,
                    step=int(update_steps),
                    keep=int(checkpoint_keep),
                )
                logger.info(
                    "checkpoint saved: step=%s env_steps=%s path=%s",
                    int(update_steps),
                    int(env_steps),
                    checkpoint_path,
                )
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = "interrupted"
        logger.info("learner interrupted; shutting down gracefully")

    finally:
        batch_sampler.close()
        _maybe_queue_async_eval()
        async_eval_return_code = None
        if async_eval.enabled:
            append_async_eval_stop(async_eval)
            async_eval_return_code = wait_for_async_eval_worker(
                async_eval,
                logger=logger,
            )
            _log_eval_records(load_new_async_eval_results(async_eval))
            if async_eval_return_code not in (None, 0):
                logger.warning(
                    "eval worker exited with returncode=%s; see %s",
                    async_eval_return_code,
                    async_eval.worker_log_path,
                )
        async_eval_counts = summarize_async_eval_results(async_eval.summary_jsonl_path)
        with progress_state_lock:
            summary_last_completed_episode_id = int(latest_completed_episode_id)
            summary_last_queued_episode_id = int(last_queued_async_eval_episode)
        summary.update(
            {
                "update_steps": int(update_steps),
                "env_steps": int(env_steps),
                "replay_size": int(len(replay_buffer)),
                "stop_reason": str(stop_reason),
                "last_completed_episode_id": int(summary_last_completed_episode_id),
                "last_queued_async_eval_episode": int(
                    summary_last_queued_episode_id
                ),
                "transport": _transport_status(),
                "offline": {
                    "enabled": bool(cfg.offline.enabled),
                    "ratio": float(cfg.offline.ratio),
                    "prepared_path": (
                        None
                        if offline_prepared_path is None
                        else str(offline_prepared_path)
                    ),
                    "manifest_path": (
                        None
                        if offline_manifest_path is None
                        else str(offline_manifest_path)
                    ),
                    "validation_stats": (
                        None
                        if offline_validation_stats is None
                        else to_jsonable(offline_validation_stats)
                    ),
                    "load_stats": to_jsonable(offline_load_stats),
                    "prepared_chunk": {
                        "offline_enabled": bool(
                            cfg.replay.prepared_chunk.offline_enabled
                        ),
                        "online_enabled": bool(
                            cfg.replay.prepared_chunk.online_enabled
                        ),
                        "profile": (
                            None
                            if offline_prepared_chunk_profile is None
                            else to_jsonable(offline_prepared_chunk_profile)
                        ),
                    },
                    "replay_size": (
                        0
                        if offline_replay_buffer is None
                        else int(len(offline_replay_buffer))
                    ),
                    "pretrain_steps_requested": int(cfg.offline.pretrain_steps),
                    "pretrain_steps_completed": int(offline_pretrain_steps_done),
                },
                "eval": {
                    "enabled": bool(async_eval.enabled),
                    "every_episodes": int(async_eval.every_episodes),
                    "triggered": int(async_eval.triggered_count),
                    "backlog": int(_async_eval_backlog()),
                    "last_completed_episode_id": int(summary_last_completed_episode_id),
                    "last_queued_episode_id": int(summary_last_queued_episode_id),
                    "last_queued_async_eval_episode": int(
                        summary_last_queued_episode_id
                    ),
                    "results_total": int(async_eval_counts["total"]),
                    "results_ok": int(async_eval_counts["ok"]),
                    "results_failed": int(async_eval_counts["failed"]),
                    "queue_path": (
                        str(async_eval.queue_path)
                        if async_eval.queue_path is not None
                        else None
                    ),
                    "summary_jsonl_path": (
                        str(async_eval.summary_jsonl_path)
                        if async_eval.summary_jsonl_path is not None
                        else None
                    ),
                    "worker_log_path": (
                        str(async_eval.worker_log_path)
                        if async_eval.worker_log_path is not None
                        else None
                    ),
                    "worker_return_code": (
                        None
                        if async_eval_return_code is None
                        else int(async_eval_return_code)
                    ),
                    "eval_checkpoint_dir": (
                        str(async_eval.eval_checkpoint_dir)
                        if async_eval.eval_checkpoint_dir is not None
                        else None
                    ),
                },
            }
        )
        with open(run_dir / cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        try:
            wandb_logger.finish()
        except Exception:  # noqa: BLE001
            pass
        try:
            server.stop()
        except Exception:  # noqa: BLE001
            pass
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
    if interrupted:
        return


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_residual_chunk",
)
def main(cfg: DictConfig) -> None:
    typed_cfg = parse_train_cfg(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("libero_residual")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", json.dumps(cfg_to_log_payload(typed_cfg), indent=2))

    set_global_seeds(typed_cfg.global_seed)

    if typed_cfg.runtime.role == "actor":
        actor(typed_cfg, run_dir=run_dir, logger=logger)
        return
    learner(typed_cfg, run_dir=run_dir, logger=logger)


if __name__ == "__main__":
    main()
