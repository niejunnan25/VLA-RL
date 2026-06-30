from __future__ import annotations

"""State-potential reward relabeling for LIBERO online residual RL."""

from dataclasses import dataclass
from dataclasses import replace
import io
import logging
import queue
import threading
import math
import time
import uuid
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Sequence

import numpy as np

from examples.libero.residual_sac.config import RewardConfig
from examples.libero.residual_sac.env.observation import build_libero_state
from examples.libero.residual_sac.env.observation import extract_libero_images
from vla_rl.rewards.progress import RemoteProgressClient

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
MetricWriter = Callable[[dict[str, Any]], None]


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
        done: bool = False,
        truncated: bool = False,
        session_id: str | None = None,
        task_id: int | None = None,
    ) -> RemoteRewardResult:
        del done, truncated, session_id, task_id
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


class RewardRelabelHttpClient:
    """HTTP client for adapters exposing absolute trajectory progress."""

    def __init__(
        self,
        *,
        url: str,
        method: str,
        timeout_sec: float,
        max_retries: int,
        retry_backoff_sec: float,
        max_retry_backoff_sec: float,
    ) -> None:
        self.url = str(url)
        self.method = str(method)
        self._client = RemoteProgressClient(
            self.url,
            method=self.method,
            timeout=float(timeout_sec),
            retries=int(max_retries),
            retry_sleep=float(retry_backoff_sec),
            max_retry_sleep=float(max_retry_backoff_sec),
        )

    def close(self) -> None:
        self._client.close()

    def query_potentials(
        self,
        *,
        states: Sequence[_TrajectoryState],
        query_indices: Sequence[int],
        episode_id: str,
        language_instruction: str,
        trajectory_start_idx: int,
        done: bool = False,
        truncated: bool = False,
        session_id: str | None = None,
        task_id: int | None = None,
    ) -> RemoteRewardResult:
        if not states:
            raise RewardRelabelError("cannot query potentials for an empty state sequence")
        if not query_indices:
            raise RewardRelabelError("cannot query an empty state index set")

        trajectory_indices = [int(state.step_in_episode) for state in states]
        trajectory = [
            _state_to_reward_payload(state, task_prompt=str(language_instruction))
            for state in states
        ]
        query_indices = [int(index) for index in query_indices]
        absolute_query_indices = [trajectory_indices[int(index)] for index in query_indices]
        resolved_session_id = None if session_id is None else str(session_id)
        resolved_task_id = None if task_id is None else int(task_id)
        metadata = {
            "episode_id": str(episode_id),
            "task": str(language_instruction),
            "done": bool(done),
            "truncated": bool(truncated),
            "trajectory_indices": trajectory_indices,
            "absolute_query_indices": absolute_query_indices,
        }
        if resolved_session_id:
            metadata["session_id"] = resolved_session_id
        if resolved_task_id is not None:
            metadata["task_id"] = int(resolved_task_id)
        request = {
            "episode_id": str(episode_id),
            "task": str(language_instruction),
            "trajectory": trajectory,
            "trajectory_indices": trajectory_indices,
            "query_indices": query_indices,
            "absolute_query_indices": absolute_query_indices,
            "trajectory_start_idx": int(trajectory_start_idx),
            "metadata": metadata,
        }
        if resolved_session_id:
            request["session_id"] = resolved_session_id
        progress = self._client.predict_progress(
            request,
            expected_count=len(query_indices),
        )
        return RemoteRewardResult(
            progress_by_key={"progress": [float(value) for value in progress]},
            rewards=[float(value) for value in progress],
            success_probs=[],
            message="",
        )


def _state_to_reward_payload(
    state: _TrajectoryState,
    *,
    task_prompt: str,
) -> dict[str, Any]:
    obs = dict(state.obs)
    return {
        "images": {
            key: np.asarray(value).copy()
            for key, value in extract_libero_images(obs).items()
        },
        "proprio": build_libero_state(obs).astype(np.float32, copy=True),
        "task": str(task_prompt),
    }


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


def _chunk_potential_boundary(
    *,
    copied: Mapping[str, Any],
    executed_steps: int,
) -> bool:
    """Return whether PBRS should zero the next-state potential.

    PLD labels one chunk-level transition and uses ``critic_terminal`` to decide
    whether Phi(next) is bootstrapped. residual_sac env chunks do not always
    provide that field, so the fallback mirrors residual_sac's current chunk
    replay bootstrap boundary: success/env_done/done/truncated.
    """

    final_info = dict(copied.get("info", {}) or {})
    infos = [dict(info) for info in list(copied.get("infos", ()))[: int(executed_steps)]]
    for info in [final_info, *reversed(infos)]:
        if "critic_terminal" in info:
            return bool(info.get("critic_terminal"))
    for info in [final_info, *infos]:
        if (
            bool(info.get("env_done", False))
            or bool(info.get("success", False))
            or bool(info.get("chunk_success", False))
        ):
            return True
    dones = list(copied.get("dones", ()))
    if dones and bool(dones[min(len(dones), int(executed_steps)) - 1]):
        return True
    return bool(copied.get("done", False) or copied.get("truncated", False))


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
        client: RewardRelabelGrpcClient | RewardRelabelHttpClient | None = None,
        progress_event_writer: MetricWriter | None = None,
    ) -> None:
        super().__init__(cfg, logger=logger)
        self._progress_event_writer = progress_event_writer
        transport = str(getattr(cfg.remote, "transport", "grpc"))
        if client is not None:
            self._client = client
        elif transport == "http":
            if cfg.remote.url is None:
                raise RewardRelabelError(
                    "reward.remote.url is required when reward.remote.transport=http"
                )
            self._client = RewardRelabelHttpClient(
                url=str(cfg.remote.url),
                method=str(cfg.remote.method),
                timeout_sec=float(cfg.remote.timeout_sec),
                max_retries=int(cfg.remote.max_retries),
                retry_backoff_sec=float(cfg.remote.retry_backoff_sec),
                max_retry_backoff_sec=float(cfg.remote.max_retry_backoff_sec),
            )
        else:
            self._client = RewardRelabelGrpcClient(
                address=str(cfg.remote.address),
                timeout_sec=float(cfg.remote.timeout_sec),
                max_message_mb=int(cfg.remote.max_message_mb),
                max_retries=int(cfg.remote.max_retries),
                retry_backoff_sec=float(cfg.remote.retry_backoff_sec),
                max_retry_backoff_sec=float(cfg.remote.max_retry_backoff_sec),
            )
        endpoint = str(cfg.remote.url or cfg.remote.address)
        self._episode_id = ""
        self._task_prompt = ""
        self._task_id = 0
        self._session_id = uuid.uuid4().hex
        self._states: list[_TrajectoryState] = []
        self._potentials: dict[int, float] = {}
        self._last_potential = 0.0
        self._stats: dict[str, Any] = {
            "enabled": True,
            "source": str(cfg.source),
            "name": str(cfg.name),
            "transform": str(cfg.transform),
            "discount": float(cfg.discount),
            "transport": str(getattr(cfg.remote, "transport", "grpc")),
            "address": str(cfg.remote.address),
            "url": cfg.remote.url,
            "method": str(getattr(cfg.remote, "method", "predict_progress")),
            "endpoint": endpoint,
            "session_id": self._session_id,
            "async_enabled": bool(cfg.async_config.enabled),
            "async_batch_size": int(cfg.async_config.batch_size),
            "async_max_pending_chunks": int(cfg.async_config.max_pending_chunks),
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
            "progress_event_write_failed": 0,
        }
        self.logger.info(
            "reward relabel enabled: source=%s name=%s transform=%s discount=%.4f endpoint=%s",
            str(cfg.source),
            str(cfg.name),
            str(cfg.transform),
            float(cfg.discount),
            endpoint,
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
            end_state_idx = int(start_state_idx) + int(executed_steps)
            if start_state_idx not in self._potentials:
                needed_indices.append(start_state_idx)
            if end_state_idx not in self._potentials:
                needed_indices.append(end_state_idx)
        query_indices = sorted(set(needed_indices))

        fallback_to_env = False
        rpc_sec = 0.0
        rpc_state_count = 0
        rpc_query_count = 0
        rpc_trajectory_indices: list[int] = []
        rpc_query_indices_for_log: list[int] = []
        rpc_absolute_query_indices: list[int] = []
        batch_done = any(bool(prepared["copied"].get("done", False)) for prepared in prepared_chunks)
        batch_truncated = any(bool(prepared["copied"].get("truncated", False)) for prepared in prepared_chunks)
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
                rpc_trajectory_indices = [
                    int(state.step_in_episode) for state in rpc_states
                ]
                rpc_query_indices_for_log = [int(index) for index in rpc_query_indices]
                rpc_absolute_query_indices = [
                    rpc_trajectory_indices[int(index)]
                    for index in rpc_query_indices_for_log
                ]
                remote_result = self._client.query_potentials(
                    states=rpc_states,
                    query_indices=rpc_query_indices,
                    episode_id=self._episode_id,
                    language_instruction=self._task_prompt,
                    trajectory_start_idx=rpc_trajectory_start_idx,
                    done=bool(batch_done),
                    truncated=bool(batch_truncated),
                    session_id=self._session_id,
                    task_id=int(self._task_id),
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
            end_state_idx = int(start_state_idx) + int(executed_steps)
            env_rewards = [float(value) for value in list(prepared["env_rewards"])]
            chunk_env_reward = _finite_float(
                copied.get(
                    "env_reward_sum",
                    copied.get("reward_sum", float(sum(env_rewards))),
                ),
                field_name="chunk env reward",
            )

            potential_before = float(
                self._potentials.get(start_state_idx, self._last_potential)
            )
            potential_after = float(
                self._potentials.get(end_state_idx, potential_before)
            )
            reward_terminal = _chunk_potential_boundary(
                copied=copied,
                executed_steps=int(executed_steps),
            )
            potential_discount = (
                0.0
                if bool(reward_terminal)
                else float(self.cfg.discount) ** int(executed_steps)
            )

            if fallback_to_env:
                chunk_reward = float(chunk_env_reward)
                chunk_component = 0.0
            else:
                chunk_reward, chunk_component = _apply_reward_transform(
                    transform=str(self.cfg.transform),
                    potential_before=float(potential_before),
                    potential_after=float(potential_after),
                    env_reward=float(chunk_env_reward),
                    discount=float(potential_discount),
                    scale=float(self.cfg.scale),
                    shift=float(self.cfg.shift),
                    clip_min=self.cfg.clip_min,
                    clip_max=self.cfg.clip_max,
                )
            self._last_potential = float(potential_after)

            relabeled_rewards = [float(chunk_reward)] + [0.0] * max(
                0,
                int(executed_steps) - 1,
            )
            relabeled_infos: list[dict[str, Any]] = []
            relabeled_steps: list[dict[str, Any]] = []
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
                        "reward_relabel_granularity": "chunk",
                        "reward_model_progress": float(potential_after),
                        "reward_model_previous_progress": float(potential_before),
                        "reward_model_potential_before": float(potential_before),
                        "reward_model_potential_after": float(potential_after),
                        "reward_model_component": float(chunk_component),
                        "reward_potential_discount": float(potential_discount),
                        "reward_model_terminal": bool(reward_terminal),
                        "critic_terminal": bool(reward_terminal),
                        "chunk_env_reward": float(chunk_env_reward),
                        "chunk_relabeled_reward": float(chunk_reward),
                        "chunk_executed_steps": int(executed_steps),
                        "relabeled_reward": float(relabeled_rewards[local_idx]),
                    }
                )
                step["env_reward"] = float(env_rewards[local_idx])
                step["reward"] = float(relabeled_rewards[local_idx])
                step["info"] = step_info
                relabeled_steps.append(step)
                relabeled_infos.append(step_info)

            self._write_progress_event(
                episode_step_start=int(prepared["episode_step_start"]),
                local_idx=int(executed_steps) - 1,
                env_reward=float(chunk_env_reward),
                relabeled_reward=float(chunk_reward),
                component=float(chunk_component),
                progress=float(potential_after),
                previous_progress=float(potential_before),
                potential_discount=float(potential_discount),
                reward_terminal=bool(reward_terminal),
                rpc_sec=float(rpc_sec),
                rpc_trajectory_indices=rpc_trajectory_indices,
                rpc_query_indices=rpc_query_indices_for_log,
                rpc_absolute_query_indices=rpc_absolute_query_indices,
                fallback_to_env=bool(fallback_to_env),
                done=bool(copied.get("done", False)),
                truncated=bool(copied.get("truncated", False)),
            )

            copied["steps"] = relabeled_steps
            copied["rewards"] = relabeled_rewards
            copied["infos"] = relabeled_infos
            copied["reward_sum"] = float(chunk_reward)
            copied["env_rewards"] = env_rewards
            copied["env_reward_sum"] = float(chunk_env_reward)
            copied["reward_source"] = str(self.cfg.name)
            copied["reward_transform"] = str(self.cfg.transform)
            copied["reward_relabel_granularity"] = "chunk"
            copied["reward_model_progress"] = [float(potential_after)]
            copied["reward_model_potential_before"] = [float(potential_before)]
            copied["reward_model_potential_after"] = [float(potential_after)]
            copied["reward_model_components"] = [float(chunk_component)]
            copied["reward_potential_discounts"] = [float(potential_discount)]
            copied["reward_model_terminals"] = [bool(reward_terminal)]
            final_info = dict(copied.get("info", {}))
            if relabeled_infos:
                final_info.update(relabeled_infos[-1])
            final_info.update(
                {
                    "env_reward_sum": float(chunk_env_reward),
                    "relabeled_reward_sum": float(chunk_reward),
                    "reward_model_component": float(chunk_component),
                    "reward_model_progress": float(potential_after),
                    "reward_model_previous_progress": float(potential_before),
                    "reward_potential_discount": float(potential_discount),
                    "reward_model_terminal": bool(reward_terminal),
                    "critic_terminal": bool(reward_terminal),
                    "reward_relabel_granularity": "chunk",
                }
            )
            copied["info"] = final_info

            last_env_reward_sum = float(chunk_env_reward)
            last_model_reward_sum = float(chunk_reward)
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

    def _write_progress_event(
        self,
        *,
        episode_step_start: int,
        local_idx: int,
        env_reward: float,
        relabeled_reward: float,
        component: float,
        progress: float,
        previous_progress: float,
        potential_discount: float,
        reward_terminal: bool,
        rpc_sec: float,
        rpc_trajectory_indices: Sequence[int],
        rpc_query_indices: Sequence[int],
        rpc_absolute_query_indices: Sequence[int],
        fallback_to_env: bool,
        done: bool,
        truncated: bool,
    ) -> None:
        if self._progress_event_writer is None:
            return
        absolute_step = int(episode_step_start) + int(local_idx) + 1
        event = {
            "role": "reward_progress",
            "source": "remote_progress",
            "status": "fallback_env" if fallback_to_env else "ok",
            "episode_id": self._episode_id,
            "task": self._task_prompt,
            "task_id": int(self._task_id),
            "reward_name": str(self.cfg.name),
            "reward_transform": str(self.cfg.transform),
            "reward_relabel_granularity": "chunk",
            "transport": str(getattr(self.cfg.remote, "transport", "grpc")),
            "endpoint": str(self.cfg.remote.url or self.cfg.remote.address),
            "episode_step": int(absolute_step),
            "boundary_index": int(absolute_step),
            "previous_boundary_index": int(episode_step_start),
            "trajectory_indices": [int(index) for index in rpc_trajectory_indices],
            "query_indices": [int(index) for index in rpc_query_indices],
            "absolute_query_indices": [int(index) for index in rpc_absolute_query_indices],
            "progress": float(progress),
            "previous_progress": float(previous_progress),
            "env_reward": float(env_reward),
            "computed_reward": float(relabeled_reward),
            "reward_model_component": float(component),
            "scale": float(self.cfg.scale),
            "discount": float(self.cfg.discount),
            "potential_discount": float(potential_discount),
            "reward_model_terminal": bool(reward_terminal),
            "latency_sec": float(rpc_sec),
            "done": bool(done),
            "truncated": bool(truncated),
        }
        try:
            self._progress_event_writer(event)
        except Exception:
            self._stats["progress_event_write_failed"] = int(
                self._stats.get("progress_event_write_failed", 0)
            ) + 1


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


@dataclass(frozen=True)
class _PendingRewardChunk:
    seq: int
    raw: Any


class AsyncRewardRelabelCoordinator:
    """Background reward relabeler for chunk-level residual replay."""

    def __init__(
        self,
        *,
        relabeler: BaseRewardRelabeler,
        batch_size: int,
        max_pending_chunks: int,
        max_wait_ms: int,
        logger: logging.Logger | None = None,
    ) -> None:
        if not relabeler.enabled:
            raise ValueError("AsyncRewardRelabelCoordinator requires an enabled relabeler")
        self._relabeler = relabeler
        self._batch_size = max(1, int(batch_size))
        self._max_wait_sec = max(0.0, float(max_wait_ms) / 1000.0)
        self._logger = logger or logging.getLogger(__name__)
        self._queue: queue.Queue[_PendingRewardChunk | None] = queue.Queue(
            maxsize=max(1, int(max_pending_chunks))
        )
        self._condition = threading.Condition()
        self._ready: dict[int, Any] = {}
        self._failed: BaseException | None = None
        self._closed = False
        self._next_submit_seq = 0
        self._next_commit_seq = 0
        self._last_submitted_seq: int | None = None
        self._stats: dict[str, Any] = {
            "async_enabled": True,
            "async_submitted": 0,
            "async_ready": 0,
            "async_committed": 0,
            "async_failed": 0,
            "async_pending": 0,
            "async_batch_size": int(self._batch_size),
            "async_max_wait_ms": int(max_wait_ms),
        }
        self._thread = threading.Thread(
            target=self._worker_main,
            name="residual-sac-reward-relabel",
            daemon=True,
        )
        self._thread.start()

    def _pending_count_locked(self) -> int:
        return int(self._queue.qsize() + len(self._ready))

    @property
    def pending_count(self) -> int:
        with self._condition:
            return self._pending_count_locked()

    @property
    def next_commit_seq(self) -> int:
        with self._condition:
            return int(self._next_commit_seq)

    def start_episode(self, **kwargs: Any) -> None:
        self._raise_if_failed()
        with self._condition:
            if self._queue.qsize() or self._ready:
                raise RuntimeError(
                    "cannot start a new reward episode with pending relabel work"
                )
            self._next_submit_seq = 0
            self._next_commit_seq = 0
            self._last_submitted_seq = None
            self._stats["async_pending"] = 0
            self._stats["async_ready"] = 0
        self._relabeler.start_episode(**kwargs)

    def submit_chunk(self, raw: Any) -> int:
        self._raise_if_failed()
        with self._condition:
            if self._closed:
                raise RuntimeError("async reward relabeler is closed")
            seq = int(self._next_submit_seq)
            self._next_submit_seq += 1
            self._last_submitted_seq = seq
        self._queue.put(_PendingRewardChunk(seq=seq, raw=raw))
        with self._condition:
            self._stats["async_submitted"] += 1
            self._stats["async_pending"] = self._pending_count_locked()
            self._condition.notify_all()
        return seq

    def pop_ready(self, *, block_until_seq: int | None = None) -> list[Any]:
        with self._condition:
            target_seq = None if block_until_seq is None else int(block_until_seq)
            ready: list[Any] = []
            while True:
                self._raise_if_failed_locked()
                while self._next_commit_seq in self._ready:
                    raw = self._ready.pop(int(self._next_commit_seq))
                    ready.append(raw)
                    self._next_commit_seq += 1
                if target_seq is None or self._next_commit_seq > int(target_seq):
                    break
                self._condition.wait(timeout=0.1)
            self._stats["async_ready"] = len(self._ready)
            self._stats["async_committed"] += len(ready)
            self._stats["async_pending"] = self._pending_count_locked()
            return ready

    def finish_episode(self, *, block: bool = True) -> list[Any]:
        last_submitted = self._last_submitted_seq
        ready = self.pop_ready(
            block_until_seq=(int(last_submitted) if block and last_submitted is not None else None)
        )
        self._relabeler.finish_episode()
        return ready

    def close(self, *, drain: bool = True) -> None:
        if bool(drain):
            self._queue.join()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._queue.put(None)
        self._thread.join(timeout=30.0)
        self._relabeler.close()
        self._raise_if_failed()

    def status_snapshot(self) -> dict[str, Any]:
        status = dict(self._relabeler.status_snapshot())
        with self._condition:
            status.update(self._stats)
            status["async_pending"] = self._pending_count_locked()
            status["async_ready"] = len(self._ready)
            status["async_next_submit_seq"] = int(self._next_submit_seq)
            status["async_next_commit_seq"] = int(self._next_commit_seq)
        return status

    def _worker_main(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            batch = [item]
            deadline = time.perf_counter() + self._max_wait_sec
            while len(batch) < self._batch_size:
                timeout = 0.0
                if self._max_wait_sec > 0.0:
                    timeout = max(0.0, deadline - time.perf_counter())
                    if timeout <= 0.0:
                        break
                try:
                    if self._max_wait_sec > 0.0:
                        next_item = self._queue.get(timeout=timeout)
                    else:
                        next_item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if next_item is None:
                    self._queue.task_done()
                    self._queue.put(None)
                    break
                batch.append(next_item)
            try:
                chunk_specs = [
                    (_chunk_result_from_record(pending.raw), int(pending.raw.episode_step_start))
                    for pending in batch
                ]
                relabeled = self._relabeler.relabel_chunk_batch(chunk_specs)
                if len(relabeled) != len(batch):
                    raise RewardRelabelError(
                        "async reward relabel returned mismatched batch length: "
                        f"got {len(relabeled)} expected {len(batch)}"
                    )
                with self._condition:
                    for pending, chunk_result in zip(batch, relabeled, strict=True):
                        self._ready[int(pending.seq)] = _record_with_chunk_result(
                            pending.raw,
                            chunk_result,
                        )
                    self._stats["async_ready"] = len(self._ready)
                    self._stats["async_pending"] = self._pending_count_locked()
                    self._condition.notify_all()
            except BaseException as exc:  # noqa: BLE001
                with self._condition:
                    self._failed = exc
                    self._stats["async_failed"] += len(batch)
                    self._condition.notify_all()
                self._logger.exception("async reward relabel failed")
            finally:
                for _pending in batch:
                    self._queue.task_done()

    def _raise_if_failed(self) -> None:
        with self._condition:
            self._raise_if_failed_locked()

    def _raise_if_failed_locked(self) -> None:
        if self._failed is not None:
            raise RuntimeError("async reward relabel worker failed") from self._failed


def _chunk_result_from_record(raw: Any) -> dict[str, Any]:
    return {
        "steps": [dict(step) for step in list(getattr(raw, "steps", ()))],
        "observations": [dict(obs) for obs in list(raw.post_step_observations)],
        "rewards": [float(value) for value in list(raw.rewards)],
        "dones": [bool(value) for value in list(raw.dones)],
        "infos": [dict(value) for value in list(raw.infos)],
        "obs": dict(raw.final_obs),
        "done": bool(raw.chunk_done),
        "truncated": bool(raw.chunk_truncated),
        "reward_sum": float(raw.reward_sum),
        "info": dict(raw.chunk_info),
        "num_steps": int(raw.executed_steps),
    }


def _record_with_chunk_result(raw: Any, chunk_result: Mapping[str, Any]) -> Any:
    executed_steps = int(chunk_result.get("num_steps", raw.executed_steps))
    return replace(
        raw,
        post_step_observations=[
            dict(obs) for obs in list(chunk_result.get("observations", raw.post_step_observations))[:executed_steps]
        ],
        rewards=[float(value) for value in list(chunk_result.get("rewards", raw.rewards))[:executed_steps]],
        dones=[bool(value) for value in list(chunk_result.get("dones", raw.dones))[:executed_steps]],
        infos=[dict(value) for value in list(chunk_result.get("infos", raw.infos))[:executed_steps]],
        final_obs=dict(chunk_result.get("obs", raw.final_obs)),
        chunk_done=bool(chunk_result.get("done", raw.chunk_done)),
        chunk_truncated=bool(chunk_result.get("truncated", raw.chunk_truncated)),
        reward_sum=float(chunk_result.get("reward_sum", raw.reward_sum)),
        chunk_info=dict(chunk_result.get("info", raw.chunk_info)),
        executed_steps=executed_steps,
        steps=[dict(step) for step in list(chunk_result.get("steps", raw.steps))[:executed_steps]],
    )


def build_reward_relabeler(
    cfg: RewardConfig,
    *,
    logger: logging.Logger | None = None,
    progress_event_writer: MetricWriter | None = None,
) -> BaseRewardRelabeler:
    source = str(cfg.source)
    if source == "env":
        return BaseRewardRelabeler(cfg, logger=logger)
    if source in {
        "progress_rpc",
        "remote",
        "robodopamine",
        "robodopamine_rpc",
        "robometer",
        "robometer_rpc",
    }:
        return RemoteRewardRelabeler(
            cfg,
            logger=logger,
            progress_event_writer=progress_event_writer,
        )
    raise ValueError(f"unsupported reward.source={source!r}")


__all__ = [
    "AsyncRewardRelabelCoordinator",
    "BaseRewardRelabeler",
    "RemoteRewardRelabeler",
    "RemoteRewardResult",
    "RewardRelabelError",
    "RewardRelabelGrpcClient",
    "RewardRelabelHttpClient",
    "build_reward_relabeler",
]
