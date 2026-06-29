from __future__ import annotations

"""State-potential reward relabeling for LIBERO online residual RL."""

from dataclasses import dataclass
import io
import logging
import math
import time
from typing import Any
from typing import Mapping
from typing import Sequence

import numpy as np

from examples.libero.residual_sac.config import RewardConfig

_VALID_TRANSFORMS = {
    "env_only",
    "progress_abs",
    "progress_delta",
    "env_plus_progress",
    "env_plus_delta",
    "potential_abs",
    "potential_delta",
    "env_plus_potential_delta",
}
_PROTO_CLASSES: tuple[type[Any], type[Any]] | None = None


@dataclass(frozen=True)
class RemoteRewardResult:
    progress_by_key: dict[str, list[float]]
    rewards: list[float]
    success_probs: list[float]
    message: str


@dataclass(frozen=True)
class _TrajectoryState:
    obs: dict[str, np.ndarray]
    step_in_episode: int


class RewardRelabelError(RuntimeError):
    """Raised when reward relabeling cannot produce aligned step rewards."""


class RewardRelabelGrpcClient:
    """gRPC client for a generic trajectory-progress service.

    The service is expected to return absolute state potentials/progress values for
    requested state indices. For compatibility with robometer-policy-learning's
    existing proto, states are encoded as Transition messages whose ``obs`` field
    contains the state observation, and ``batch_indices`` names the queried states.
    """

    def __init__(
        self,
        *,
        address: str,
        timeout_sec: float,
        max_message_mb: int,
        max_retries: int,
        retry_backoff_sec: float,
        max_retry_backoff_sec: float,
    ) -> None:
        try:
            import grpc  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - dependency-gated path
            raise RuntimeError(
                "reward.source=progress_rpc requires grpcio and protobuf in the "
                "actor Python environment. Install them in the serl_torch runtime env."
            ) from exc

        self._grpc = grpc
        self._request_cls, self._response_cls = _dynamic_reward_proto_classes()
        self.address = str(address)
        self.timeout_sec = float(timeout_sec)
        self.max_retries = int(max_retries)
        self.retry_backoff_sec = float(retry_backoff_sec)
        self.max_retry_backoff_sec = float(max_retry_backoff_sec)
        max_bytes = int(max_message_mb) * 1024 * 1024
        self._channel_options = [
            ("grpc.max_send_message_length", max_bytes),
            ("grpc.max_receive_message_length", max_bytes),
        ]
        self._create_channel()

    def _create_channel(self) -> None:
        self._channel = self._grpc.insecure_channel(
            self.address,
            options=self._channel_options,
        )
        self._call = self._channel.unary_unary(
            "/distributed.RewardRelabelService/RelabelRewards",
            request_serializer=self._request_cls.SerializeToString,
            response_deserializer=self._response_cls.FromString,
        )

    def _reset_channel(self) -> None:
        try:
            self._channel.close()
        except Exception:  # noqa: BLE001
            pass
        self._create_channel()

    def close(self) -> None:
        try:
            self._channel.close()
        except Exception:  # noqa: BLE001
            pass

    def query_potentials(
        self,
        *,
        states: Sequence[_TrajectoryState],
        query_indices: Sequence[int],
        episode_id: str,
        language_instruction: str,
        trajectory_start_idx: int,
    ) -> RemoteRewardResult:
        if not states:
            raise RewardRelabelError("cannot query potentials for an empty state sequence")
        if not query_indices:
            raise RewardRelabelError("cannot query an empty state index set")

        request = self._request_cls()
        request.batch_start_idx = 0
        request.batch_end_idx = len(states)
        request.batch_indices.extend(int(index) for index in query_indices)
        request.trajectory_start_idx = int(trajectory_start_idx)

        empty_action = _ndarray_to_bytes(np.zeros((0,), dtype=np.float32))
        for state in states:
            transition = request.transitions.add()
            for key, value in state.obs.items():
                transition.obs[str(key)].data = _ndarray_to_bytes(value)
            transition.action.data = empty_action
            transition.reward_env = 0.0
            transition.done = False
            transition.truncated = False
            transition.episode_id = str(episode_id)
            transition.step_in_episode = int(state.step_in_episode)
            transition.timestamp_ns = 0
            request.language_instructions.append(str(language_instruction))
            request.episode_ids.append(str(episode_id))
            request.step_in_episodes.append(int(state.step_in_episode))

        response = None
        backoff_sec = float(self.retry_backoff_sec)
        for attempt in range(int(self.max_retries) + 1):
            try:
                response = self._call(request, timeout=float(self.timeout_sec))
                break
            except self._grpc.RpcError:
                self._reset_channel()
                if attempt >= int(self.max_retries):
                    raise
                time.sleep(max(0.0, backoff_sec))
                backoff_sec = min(
                    max(0.0, backoff_sec) * 2.0,
                    float(self.max_retry_backoff_sec),
                )

        if response is None:
            raise RewardRelabelError("reward potential RPC returned no response")
        if not bool(response.ok):
            raise RewardRelabelError(str(response.message or "reward potential RPC failed"))

        progress_by_key: dict[str, list[float]] = {}
        for key, predictions in response.progress_predictions_by_key.items():
            progress_by_key[str(key)] = [float(value) for value in predictions.progress]
        return RemoteRewardResult(
            progress_by_key=progress_by_key,
            rewards=[float(value) for value in response.rewards],
            success_probs=[float(value) for value in response.success_probs],
            message=str(response.message),
        )


def _dynamic_reward_proto_classes() -> tuple[type[Any], type[Any]]:
    global _PROTO_CLASSES
    if _PROTO_CLASSES is not None:
        return _PROTO_CLASSES

    try:
        from google.protobuf import descriptor_pb2  # type: ignore[import-not-found]
        from google.protobuf import descriptor_pool  # type: ignore[import-not-found]
        from google.protobuf import message_factory  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency-gated path
        raise RuntimeError(
            "reward.source=progress_rpc requires protobuf in the actor Python environment."
        ) from exc

    label_optional = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    label_repeated = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    type_bool = descriptor_pb2.FieldDescriptorProto.TYPE_BOOL
    type_bytes = descriptor_pb2.FieldDescriptorProto.TYPE_BYTES
    type_float = descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT
    type_int32 = descriptor_pb2.FieldDescriptorProto.TYPE_INT32
    type_int64 = descriptor_pb2.FieldDescriptorProto.TYPE_INT64
    type_message = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    type_string = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    def add_field(
        message: Any,
        name: str,
        number: int,
        label: int,
        field_type: int,
        type_name: str | None = None,
    ) -> None:
        field = message.field.add()
        field.name = name
        field.number = int(number)
        field.label = int(label)
        field.type = int(field_type)
        if type_name is not None:
            field.type_name = type_name

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "reward_relabel_dynamic.proto"
    file_proto.package = "distributed"
    file_proto.syntax = "proto3"

    meta_entry = file_proto.message_type.add()
    meta_entry.name = "MetaEntry"
    add_field(meta_entry, "key", 1, label_optional, type_string)
    add_field(meta_entry, "value", 2, label_optional, type_bytes)

    ndarray = file_proto.message_type.add()
    ndarray.name = "NDArray"
    add_field(ndarray, "data", 1, label_optional, type_bytes)

    transition = file_proto.message_type.add()
    transition.name = "Transition"
    obs_entry = transition.nested_type.add()
    obs_entry.name = "ObsEntry"
    obs_entry.options.map_entry = True
    add_field(obs_entry, "key", 1, label_optional, type_string)
    add_field(obs_entry, "value", 2, label_optional, type_message, ".distributed.NDArray")
    next_obs_entry = transition.nested_type.add()
    next_obs_entry.name = "NextObsEntry"
    next_obs_entry.options.map_entry = True
    add_field(next_obs_entry, "key", 1, label_optional, type_string)
    add_field(next_obs_entry, "value", 2, label_optional, type_message, ".distributed.NDArray")
    add_field(transition, "info", 1, label_repeated, type_message, ".distributed.MetaEntry")
    add_field(transition, "obs", 2, label_repeated, type_message, ".distributed.Transition.ObsEntry")
    add_field(transition, "action", 3, label_optional, type_message, ".distributed.NDArray")
    add_field(transition, "reward_env", 4, label_optional, type_float)
    add_field(transition, "next_obs", 5, label_repeated, type_message, ".distributed.Transition.NextObsEntry")
    add_field(transition, "done", 6, label_optional, type_bool)
    add_field(transition, "truncated", 7, label_optional, type_bool)
    add_field(transition, "episode_id", 8, label_optional, type_string)
    add_field(transition, "step_in_episode", 9, label_optional, type_int32)
    add_field(transition, "timestamp_ns", 10, label_optional, type_int64)

    request = file_proto.message_type.add()
    request.name = "RelabelRewardsRequest"
    add_field(request, "transitions", 1, label_repeated, type_message, ".distributed.Transition")
    add_field(request, "language_instructions", 2, label_repeated, type_string)
    add_field(request, "video_frames", 3, label_repeated, type_bytes)
    add_field(request, "dino_embeddings", 4, label_repeated, type_bytes)
    add_field(request, "text_embeddings", 5, label_repeated, type_bytes)
    add_field(request, "episode_ids", 6, label_repeated, type_string)
    add_field(request, "step_in_episodes", 7, label_repeated, type_int32)
    add_field(request, "batch_start_idx", 8, label_optional, type_int32)
    add_field(request, "batch_end_idx", 9, label_optional, type_int32)
    add_field(request, "batch_indices", 10, label_repeated, type_int32)
    add_field(request, "trajectory_start_idx", 11, label_optional, type_int32)

    progress_predictions = file_proto.message_type.add()
    progress_predictions.name = "ProgressPredictions"
    add_field(progress_predictions, "progress", 1, label_repeated, type_float)
    add_field(progress_predictions, "success_probs", 2, label_repeated, type_float)

    response = file_proto.message_type.add()
    response.name = "RelabelRewardsResponse"
    progress_map_entry = response.nested_type.add()
    progress_map_entry.name = "ProgressPredictionsByKeyEntry"
    progress_map_entry.options.map_entry = True
    add_field(progress_map_entry, "key", 1, label_optional, type_string)
    add_field(progress_map_entry, "value", 2, label_optional, type_message, ".distributed.ProgressPredictions")
    add_field(response, "rewards", 1, label_repeated, type_float)
    add_field(response, "success_probs", 2, label_repeated, type_float)
    add_field(response, "ok", 3, label_optional, type_bool)
    add_field(response, "message", 4, label_optional, type_string)
    add_field(response, "progress_predictions_by_key", 5, label_repeated, type_message, ".distributed.RelabelRewardsResponse.ProgressPredictionsByKeyEntry")

    if hasattr(message_factory, "GetMessages"):
        classes = message_factory.GetMessages([file_proto])
        _PROTO_CLASSES = (
            classes["distributed.RelabelRewardsRequest"],
            classes["distributed.RelabelRewardsResponse"],
        )
        return _PROTO_CLASSES

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)
    factory = message_factory.MessageFactory(pool)
    _PROTO_CLASSES = (
        factory.GetPrototype(pool.FindMessageTypeByName("distributed.RelabelRewardsRequest")),
        factory.GetPrototype(pool.FindMessageTypeByName("distributed.RelabelRewardsResponse")),
    )
    return _PROTO_CLASSES


def _ndarray_to_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def _finite_float(value: Any, *, field_name: str) -> float:
    try:
        resolved = float(value)
    except Exception as exc:  # noqa: BLE001
        raise RewardRelabelError(f"{field_name} must be float-like, got {value!r}") from exc
    if not math.isfinite(resolved):
        raise RewardRelabelError(f"{field_name} must be finite, got {value!r}")
    return resolved


def _serializable_obs(obs: Mapping[str, Any], *, task_id: int) -> dict[str, np.ndarray]:
    serialized: dict[str, np.ndarray] = {}
    for key, value in dict(obs).items():
        try:
            arr = np.asarray(value)
        except Exception:  # noqa: BLE001
            continue
        if arr.dtype == object or arr.dtype.kind in {"S", "U"}:
            continue
        serialized[str(key)] = np.ascontiguousarray(arr)
    serialized["task_id"] = np.asarray(int(task_id), dtype=np.int32)
    return serialized


def _component_from_potential(
    *,
    transform: str,
    potential_before: float,
    potential_after: float,
    discount: float,
) -> float:
    if transform == "env_only":
        return 0.0
    if transform in {"progress_abs", "env_plus_progress", "potential_abs"}:
        return float(potential_after)
    if transform in {"progress_delta", "env_plus_delta"}:
        return float(potential_after) - float(potential_before)
    if transform in {"potential_delta", "env_plus_potential_delta"}:
        return float(discount) * float(potential_after) - float(potential_before)
    raise RewardRelabelError(f"unsupported reward transform: {transform!r}")


def _apply_reward_transform(
    *,
    transform: str,
    potential_before: float,
    potential_after: float,
    env_reward: float,
    discount: float,
    scale: float,
    shift: float,
    clip_min: float | None,
    clip_max: float | None,
) -> tuple[float, float]:
    if transform not in _VALID_TRANSFORMS:
        raise RewardRelabelError(f"unsupported reward transform: {transform!r}")

    component = _component_from_potential(
        transform=transform,
        potential_before=float(potential_before),
        potential_after=float(potential_after),
        discount=float(discount),
    )
    if transform == "env_only":
        reward = float(env_reward)
    elif transform.startswith("env_plus_"):
        reward = float(env_reward) + float(scale) * float(component) + float(shift)
    else:
        reward = float(scale) * float(component) + float(shift)

    if clip_min is not None:
        reward = max(float(clip_min), reward)
    if clip_max is not None:
        reward = min(float(clip_max), reward)
    return float(reward), float(component)


def _copy_chunk_result(chunk_result: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(chunk_result)
    copied["steps"] = [dict(step) for step in list(chunk_result.get("steps", ()))]
    copied["rewards"] = [float(value) for value in list(chunk_result.get("rewards", ()))]
    copied["infos"] = [dict(value) for value in list(chunk_result.get("infos", ()))]
    return copied


def _select_potentials(
    *,
    result: RemoteRewardResult,
    query_indices: Sequence[int],
    total_states: int,
) -> list[float]:
    expected_len = len(query_indices)
    candidates: list[tuple[str, list[float]]] = []
    if result.rewards:
        candidates.append(("rewards", result.rewards))
    if len(result.progress_by_key) == 1:
        key, values = next(iter(result.progress_by_key.items()))
        candidates.append((f"progress_predictions_by_key[{key!r}]", values))
    elif len(result.progress_by_key) > 1:
        keys = sorted(result.progress_by_key)
        lengths = {len(result.progress_by_key[key]) for key in keys}
        if len(lengths) == 1:
            stacked = np.asarray(
                [result.progress_by_key[key] for key in keys],
                dtype=np.float32,
            )
            candidates.append(
                ("mean(progress_predictions_by_key)", stacked.mean(axis=0).tolist())
            )
    if result.success_probs:
        candidates.append(("success_probs", result.success_probs))

    length_errors = []
    for source, values in candidates:
        if len(values) == int(expected_len):
            return [
                _finite_float(value, field_name=f"remote potential from {source}")
                for value in values
            ]
        if len(values) == int(total_states):
            return [
                _finite_float(
                    values[int(index)],
                    field_name=f"remote potential from {source}",
                )
                for index in query_indices
            ]
        length_errors.append(f"{source}: got {len(values)}, expected {int(expected_len)}")

    if length_errors:
        raise RewardRelabelError(
            "reward RPC response length mismatch: " + "; ".join(length_errors)
        )
    raise RewardRelabelError(
        "reward RPC response has no canonical potential. Expected response.rewards, "
        "a single progress_predictions_by_key entry, averaged same-length entries, or success_probs."
    )


class BaseRewardRelabeler:
    def __init__(self, cfg: RewardConfig, *, logger: logging.Logger | None = None) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    @property
    def enabled(self) -> bool:
        return False

    def start_episode(
        self,
        *,
        episode_id: int,
        task_prompt: str,
        initial_obs: Mapping[str, Any],
        init_episode_idx: int,
        task_id: int,
    ) -> None:
        del episode_id, task_prompt, initial_obs, init_episode_idx, task_id

    def relabel_chunk(
        self,
        chunk_result: Mapping[str, Any],
        *,
        episode_step_start: int,
    ) -> dict[str, Any]:
        del episode_step_start
        return dict(chunk_result)

    def relabel_chunk_batch(
        self,
        chunk_specs: Sequence[tuple[Mapping[str, Any], int]],
    ) -> list[dict[str, Any]]:
        return [
            self.relabel_chunk(
                chunk_result,
                episode_step_start=int(episode_step_start),
            )
            for chunk_result, episode_step_start in chunk_specs
        ]

    def finish_episode(self) -> None:
        pass

    def close(self) -> None:
        pass

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "source": str(self.cfg.source),
            "name": str(self.cfg.name),
            "transform": str(self.cfg.transform),
            "discount": float(self.cfg.discount),
        }


class RemoteRewardRelabeler(BaseRewardRelabeler):
    def __init__(
        self,
        cfg: RewardConfig,
        *,
        logger: logging.Logger | None = None,
        client: RewardRelabelGrpcClient | None = None,
    ) -> None:
        super().__init__(cfg, logger=logger)
        self._client = client or RewardRelabelGrpcClient(
            address=str(cfg.remote.address),
            timeout_sec=float(cfg.remote.timeout_sec),
            max_message_mb=int(cfg.remote.max_message_mb),
            max_retries=int(cfg.remote.max_retries),
            retry_backoff_sec=float(cfg.remote.retry_backoff_sec),
            max_retry_backoff_sec=float(cfg.remote.max_retry_backoff_sec),
        )
        self._episode_id = ""
        self._task_prompt = ""
        self._task_id = 0
        self._states: list[_TrajectoryState] = []
        self._potentials: dict[int, float] = {}
        self._last_potential = 0.0
        self._stats: dict[str, Any] = {
            "enabled": True,
            "source": str(cfg.source),
            "name": str(cfg.name),
            "transform": str(cfg.transform),
            "discount": float(cfg.discount),
            "address": str(cfg.remote.address),
            "compact_transitions": bool(getattr(cfg.remote, "compact_transitions", False)),
            "chunks_relabelled": 0,
            "steps_relabelled": 0,
            "rpc_errors": 0,
            "fallback_chunks": 0,
            "total_rpc_sec": 0.0,
            "last_rpc_sec": 0.0,
            'last_rpc_states': 0,
            'last_rpc_queries': 0,
            "last_env_reward_sum": 0.0,
            "last_model_reward_sum": 0.0,
            "last_progress": 0.0,
            "last_potential": 0.0,
            "last_error": None,
        }
        self.logger.info(
            "reward relabel enabled: source=%s name=%s transform=%s discount=%.4f endpoint=%s",
            str(cfg.source),
            str(cfg.name),
            str(cfg.transform),
            float(cfg.discount),
            str(cfg.remote.address),
        )

    @property
    def enabled(self) -> bool:
        return True

    def start_episode(
        self,
        *,
        episode_id: int,
        task_prompt: str,
        initial_obs: Mapping[str, Any],
        init_episode_idx: int,
        task_id: int,
    ) -> None:
        del init_episode_idx
        self._episode_id = str(int(episode_id))
        self._task_prompt = str(task_prompt)
        self._task_id = int(task_id)
        self._states = [
            _TrajectoryState(
                obs=_serializable_obs(initial_obs, task_id=self._task_id),
                step_in_episode=0,
            )
        ]
        self._potentials = {}
        self._last_potential = 0.0

    def _build_rpc_query(
        self,
        query_indices: Sequence[int],
    ) -> tuple[list[_TrajectoryState], list[int], int]:
        query_indices = [int(index) for index in query_indices]
        if not bool(getattr(self.cfg.remote, "compact_transitions", False)):
            return list(self._states), query_indices, int(self.cfg.trajectory_start_idx)

        if not query_indices:
            return [], [], int(self.cfg.trajectory_start_idx)

        anchor_idx = max(0, min(int(self.cfg.trajectory_start_idx), min(query_indices)))
        if anchor_idx >= len(self._states):
            anchor_idx = 0
        original_indices = [anchor_idx]
        for index in query_indices:
            if index not in original_indices:
                original_indices.append(index)

        local_index_by_original = {
            int(original_index): int(local_index)
            for local_index, original_index in enumerate(original_indices)
        }
        compact_states = [self._states[int(index)] for index in original_indices]
        compact_query_indices = [local_index_by_original[int(index)] for index in query_indices]
        return compact_states, compact_query_indices, local_index_by_original[int(anchor_idx)]

    def relabel_chunk(
        self,
        chunk_result: Mapping[str, Any],
        *,
        episode_step_start: int,
    ) -> dict[str, Any]:
        relabeled_chunks = self.relabel_chunk_batch(
            ((chunk_result, int(episode_step_start)),)
        )
        if len(relabeled_chunks) != 1:
            raise RewardRelabelError(
                "reward relabel batch returned a mismatched single-chunk result: "
                f"got {len(relabeled_chunks)}"
            )
        return relabeled_chunks[0]

    def relabel_chunk_batch(
        self,
        chunk_specs: Sequence[tuple[Mapping[str, Any], int]],
    ) -> list[dict[str, Any]]:
        prepared_chunks: list[dict[str, Any]] = []
        total_executed_steps = 0
        for chunk_result, episode_step_start_value in chunk_specs:
            episode_step_start = int(episode_step_start_value)
            copied = _copy_chunk_result(chunk_result)
            steps = copied["steps"]
            if not steps:
                raise RewardRelabelError(
                    "chunk_result.steps is required for reward relabeling"
                )
            executed_steps = int(copied.get("num_steps", len(steps)))
            if executed_steps <= 0:
                raise RewardRelabelError("chunk_result.num_steps must be positive")
            if len(steps) < executed_steps:
                raise RewardRelabelError(
                    "chunk_result.steps is shorter than num_steps: "
                    f"{len(steps)} < {executed_steps}"
                )
            if len(copied["rewards"]) < executed_steps:
                copied["rewards"] = [
                    float(dict(step).get("reward", 0.0)) for step in steps
                ]
            if len(copied["infos"]) < executed_steps:
                copied["infos"] = [dict(dict(step).get("info", {})) for step in steps]
            if not self._states:
                first_obs = dict(steps[0]).get("obs", {})
                self._states.append(
                    _TrajectoryState(
                        obs=_serializable_obs(first_obs, task_id=self._task_id),
                        step_in_episode=int(episode_step_start),
                    )
                )

            env_rewards: list[float] = []
            start_state_idx = len(self._states) - 1
            for local_idx in range(executed_steps):
                step = dict(steps[local_idx])
                env_reward = _finite_float(
                    step.get(
                        "env_reward",
                        step.get("reward", copied["rewards"][local_idx]),
                    ),
                    field_name="env reward",
                )
                env_rewards.append(env_reward)
                if "next_obs" not in step:
                    raise RewardRelabelError("chunk step is missing next_obs")
                self._states.append(
                    _TrajectoryState(
                        obs=_serializable_obs(step["next_obs"], task_id=self._task_id),
                        step_in_episode=int(episode_step_start) + int(local_idx) + 1,
                    )
                )

            prepared_chunks.append(
                {
                    "copied": copied,
                    "steps": steps,
                    "executed_steps": int(executed_steps),
                    "episode_step_start": int(episode_step_start),
                    "start_state_idx": int(start_state_idx),
                    "env_rewards": env_rewards,
                }
            )
            total_executed_steps += int(executed_steps)

        if not prepared_chunks:
            return []

        needed_indices: list[int] = []
        for prepared in prepared_chunks:
            start_state_idx = int(prepared["start_state_idx"])
            executed_steps = int(prepared["executed_steps"])
            for local_idx in range(executed_steps):
                before_idx = int(start_state_idx) + int(local_idx)
                after_idx = before_idx + 1
                if before_idx not in self._potentials:
                    needed_indices.append(before_idx)
                if after_idx not in self._potentials:
                    needed_indices.append(after_idx)
        query_indices = sorted(set(needed_indices))

        fallback_to_env = False
        rpc_sec = 0.0
        rpc_state_count = 0
        rpc_query_count = 0
        try:
            if query_indices:
                (
                    rpc_states,
                    rpc_query_indices,
                    rpc_trajectory_start_idx,
                ) = self._build_rpc_query(query_indices)
                rpc_state_count = len(rpc_states)
                rpc_query_count = len(rpc_query_indices)
                rpc_start = time.perf_counter()
                remote_result = self._client.query_potentials(
                    states=rpc_states,
                    query_indices=rpc_query_indices,
                    episode_id=self._episode_id,
                    language_instruction=self._task_prompt,
                    trajectory_start_idx=rpc_trajectory_start_idx,
                )
                rpc_sec = time.perf_counter() - rpc_start
                potentials = _select_potentials(
                    result=remote_result,
                    query_indices=rpc_query_indices,
                    total_states=len(rpc_states),
                )
                for state_idx, potential in zip(query_indices, potentials):
                    self._potentials[int(state_idx)] = float(potential)
        except Exception as exc:  # noqa: BLE001
            self._stats["rpc_errors"] = int(self._stats["rpc_errors"]) + 1
            self._stats["last_error"] = f"{type(exc).__name__}: {exc}"
            if bool(self.cfg.fail_on_error):
                raise
            self.logger.warning(
                "reward potential query failed; falling back to env rewards: %s: %s",
                type(exc).__name__,
                exc,
            )
            self._stats["fallback_chunks"] = int(self._stats["fallback_chunks"]) + len(
                prepared_chunks
            )
            fallback_to_env = True

        relabeled_chunks: list[dict[str, Any]] = []
        last_env_reward_sum = 0.0
        last_model_reward_sum = 0.0
        for prepared in prepared_chunks:
            copied = dict(prepared["copied"])
            steps = list(prepared["steps"])
            executed_steps = int(prepared["executed_steps"])
            start_state_idx = int(prepared["start_state_idx"])
            env_rewards = [
                float(value) for value in list(prepared["env_rewards"])
            ]

            relabeled_rewards: list[float] = []
            reward_model_components: list[float] = []
            potential_before_values: list[float] = []
            potential_after_values: list[float] = []
            for local_idx, env_reward in enumerate(env_rewards):
                before_idx = int(start_state_idx) + int(local_idx)
                after_idx = before_idx + 1
                potential_before = float(
                    self._potentials.get(before_idx, self._last_potential)
                )
                potential_after = float(
                    self._potentials.get(after_idx, potential_before)
                )
                potential_before_values.append(float(potential_before))
                potential_after_values.append(float(potential_after))

                if fallback_to_env:
                    reward = float(env_reward)
                    component = 0.0
                else:
                    reward, component = _apply_reward_transform(
                        transform=str(self.cfg.transform),
                        potential_before=float(potential_before),
                        potential_after=float(potential_after),
                        env_reward=float(env_reward),
                        discount=float(self.cfg.discount),
                        scale=float(self.cfg.scale),
                        shift=float(self.cfg.shift),
                        clip_min=self.cfg.clip_min,
                        clip_max=self.cfg.clip_max,
                    )
                relabeled_rewards.append(float(reward))
                reward_model_components.append(float(component))
            if potential_after_values:
                self._last_potential = float(potential_after_values[-1])

            relabeled_steps: list[dict[str, Any]] = []
            relabeled_infos: list[dict[str, Any]] = []
            for local_idx in range(executed_steps):
                step = dict(steps[local_idx])
                info = dict(copied["infos"][local_idx])
                step_info = dict(step.get("info", {}))
                step_info.update(info)
                step_info.update(
                    {
                        "env_reward": float(env_rewards[local_idx]),
                        "reward_source": str(self.cfg.name),
                        "reward_transform": str(self.cfg.transform),
                        "reward_model_progress": float(
                            potential_after_values[local_idx]
                        ),
                        "reward_model_previous_progress": float(
                            potential_before_values[local_idx]
                        ),
                        "reward_model_potential_before": float(
                            potential_before_values[local_idx]
                        ),
                        "reward_model_potential_after": float(
                            potential_after_values[local_idx]
                        ),
                        "reward_model_component": float(
                            reward_model_components[local_idx]
                        ),
                        "relabeled_reward": float(relabeled_rewards[local_idx]),
                    }
                )
                step["env_reward"] = float(env_rewards[local_idx])
                step["reward"] = float(relabeled_rewards[local_idx])
                step["info"] = step_info
                relabeled_steps.append(step)
                relabeled_infos.append(step_info)

            copied["steps"] = relabeled_steps
            copied["rewards"] = relabeled_rewards
            copied["infos"] = relabeled_infos
            copied["reward_sum"] = float(sum(relabeled_rewards))
            copied["env_rewards"] = env_rewards
            copied["env_reward_sum"] = float(sum(env_rewards))
            copied["reward_source"] = str(self.cfg.name)
            copied["reward_transform"] = str(self.cfg.transform)
            copied["reward_model_progress"] = potential_after_values
            copied["reward_model_potential_before"] = potential_before_values
            copied["reward_model_potential_after"] = potential_after_values
            copied["reward_model_components"] = reward_model_components
            if relabeled_infos:
                final_info = dict(copied.get("info", {}))
                final_info.update(relabeled_infos[-1])
                final_info["env_reward_sum"] = float(sum(env_rewards))
                final_info["relabeled_reward_sum"] = float(sum(relabeled_rewards))
                copied["info"] = final_info

            last_env_reward_sum = float(sum(env_rewards))
            last_model_reward_sum = float(sum(relabeled_rewards))
            relabeled_chunks.append(copied)

        self._stats["chunks_relabelled"] = int(self._stats["chunks_relabelled"]) + len(
            prepared_chunks
        )
        self._stats["steps_relabelled"] = int(self._stats["steps_relabelled"]) + int(
            total_executed_steps
        )
        self._stats["last_rpc_sec"] = float(rpc_sec)
        self._stats["last_rpc_states"] = int(rpc_state_count)
        self._stats["last_rpc_queries"] = int(rpc_query_count)
        self._stats["total_rpc_sec"] = float(self._stats["total_rpc_sec"]) + float(
            rpc_sec
        )
        self._stats["last_env_reward_sum"] = float(last_env_reward_sum)
        self._stats["last_model_reward_sum"] = float(last_model_reward_sum)
        self._stats["last_progress"] = float(self._last_potential)
        self._stats["last_potential"] = float(self._last_potential)
        if not fallback_to_env:
            self._stats["last_error"] = None
        return relabeled_chunks

    def finish_episode(self) -> None:
        self._states = []
        self._potentials = {}
        self._last_potential = 0.0

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def status_snapshot(self) -> dict[str, Any]:
        return dict(self._stats)


def build_reward_relabeler(
    cfg: RewardConfig,
    *,
    logger: logging.Logger | None = None,
) -> BaseRewardRelabeler:
    source = str(cfg.source)
    if source == "env":
        return BaseRewardRelabeler(cfg, logger=logger)
    if source in {"progress_rpc", "remote", "robodopamine", "robodopamine_rpc"}:
        return RemoteRewardRelabeler(cfg, logger=logger)
    raise ValueError(f"unsupported reward.source={source!r}")


__all__ = [
    "BaseRewardRelabeler",
    "RemoteRewardRelabeler",
    "RemoteRewardResult",
    "RewardRelabelError",
    "RewardRelabelGrpcClient",
    "build_reward_relabeler",
]
