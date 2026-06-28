from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf
import pytest

from examples.libero.rlt.reward import (
    AsyncRemoteProgressRLTRewardProcessor,
    PendingRLTRewardTransition,
    SparseRLTRewardProcessor,
    build_rlt_reward_processor,
)
from tests.test_rlt_algorithm import make_rlt_state
from vla_rl.data import Observation, Transition
from vla_rl.rewards import compute_potential_discount, compute_progress_reward, normalize_progress_response
from vla_rl.rewards.processor import observation_to_reward_payload as build_reward_payload
from scripts.serve_robodopamine_progress_http import RoboDopamineProgressHttpService
from scripts.serve_robometer_progress_http import (
    RoboMeterNativeBackend,
    RoboMeterProgressHttpService,
    _cache_key_from_request,
    parse_args as parse_robometer_progress_args,
)


class ListDataStore:
    def __init__(self):
        self.payloads = []

    def insert(self, payload):
        self.payloads.append(payload)


class FakeProgressClient:
    def __init__(self, progress_by_query):
        self.progress_by_query = list(progress_by_query)
        self.requests = []

    def predict_progress(self, request, *, expected_count=None):
        self.requests.append(request)
        values = self.progress_by_query.pop(0)
        return normalize_progress_response(values, expected_count=expected_count)

    def close(self):
        return None


def make_obs(task="open drawer"):
    return Observation(
        images={
            "image_rgb_0": np.zeros((4, 4, 3), dtype=np.uint8),
            "image_rgb_1": np.ones((4, 4, 3), dtype=np.uint8),
        },
        proprio=np.zeros((8,), dtype=np.float32),
        task=task,
    )


def observation_to_reward_request_item(obs: Observation) -> dict:
    return {"images": dict(obs.images), "proprio": obs.proprio, "task": obs.task}


def make_pending(
    env_reward=0.0,
    done=False,
    truncated=False,
    episode_id=0,
    chunk_index=0,
    discount=None,
    critic_terminal=None,
):
    if critic_terminal is None:
        critic_terminal = bool(done or truncated)
    transition = Transition(
        obs=make_rlt_state(0.0),
        next_obs=make_rlt_state(1.0),
        action=np.zeros((4,), dtype=np.float32),
        reward=float(env_reward),
        done=bool(done),
        truncated=bool(truncated),
        discount=float(0.99**2 if discount is None else discount),
        executed_steps=2,
        env_steps=2,
        info={"critic_terminal": bool(critic_terminal)},
    )
    return PendingRLTRewardTransition(
        episode_id=int(episode_id),
        chunk_index=int(chunk_index),
        start_observation=make_obs(),
        end_observation=make_obs(),
        transition=transition,
        env_reward=float(env_reward),
        executed_steps=2,
        task="open drawer",
    )


def test_robometer_cache_key_uses_session_id():
    assert _cache_key_from_request({"episode_id": 7, "session_id": "actor-a"}, {}) == "actor-a:7"
    assert _cache_key_from_request({"episode_id": 7}, {"session_id": "actor-b"}) == "actor-b:7"
    assert _cache_key_from_request({"episode_id": 7}, {}) == "default:7"


def test_progress_reward_transforms():
    assert compute_progress_reward(
        "progress_abs",
        env_reward=1.0,
        progress=0.4,
        previous_progress=0.1,
        gamma=0.9,
        executed_steps=2,
    ) == 0.4
    assert np.isclose(
        compute_progress_reward(
            "env_plus_potential_delta",
            env_reward=1.0,
            progress=0.4,
            previous_progress=0.1,
            gamma=0.9,
            executed_steps=2,
            scale=2.0,
        ),
        1.0 + 2.0 * ((0.9**2) * 0.4 - 0.1),
    )


def test_potential_reward_uses_explicit_transition_discount():
    assert np.isclose(
        compute_progress_reward(
            "potential_delta",
            env_reward=0.0,
            progress=0.8,
            previous_progress=0.2,
            gamma=0.99,
            executed_steps=5,
            discount=0.5,
        ),
        0.5 * 0.8 - 0.2,
    )


def test_potential_reward_zeroes_terminal_state():
    assert compute_potential_discount(
        gamma=0.99,
        executed_steps=5,
        discount=0.99**5,
        terminal=True,
    ) == 0.0
    assert np.isclose(
        compute_progress_reward(
            "env_plus_potential_delta",
            env_reward=0.0,
            progress=0.95,
            previous_progress=0.8,
            gamma=0.99,
            executed_steps=5,
            discount=0.99**5,
            terminal=True,
        ),
        -0.8,
    )


def test_normalize_progress_response_requires_full_query_start_pair():
    with pytest.raises(ValueError, match="expected 2 progress values"):
        normalize_progress_response({"progress": [0.7]}, expected_count=2)


def test_normalize_progress_response_rejects_extra_single_query_values():
    with pytest.raises(ValueError, match="expected 1 progress values"):
        normalize_progress_response({"progress": [0.2, 0.7]}, expected_count=1)


def test_normalize_progress_by_key_accepts_numpy_arrays():
    values = normalize_progress_response(
        {
            "progress_by_key": {
                "image_rgb_0": {"progress": np.array([0.2, 0.6], dtype=np.float32)},
                "image_rgb_1": {"progress": np.array([0.4, 0.8], dtype=np.float32)},
            }
        },
        expected_count=2,
    )

    np.testing.assert_allclose(values, [0.3, 0.7])


def test_sparse_processor_commits_env_reward():
    store = ListDataStore()
    processor = SparseRLTRewardProcessor(store)

    processor.submit(make_pending(env_reward=2.0))

    assert len(store.payloads) == 1
    assert store.payloads[0]["reward"] == 2.0
    assert store.payloads[0]["info"]["reward_type"] == "sparse"


def test_async_remote_progress_processor_commits_relabelled_reward():
    store = ListDataStore()
    client = FakeProgressClient([{"progress": [0.2, 0.5]}])
    processor = AsyncRemoteProgressRLTRewardProcessor(
        data_store=store,
        client=client,
        reward_type="progress_delta",
        gamma=0.99,
        scale=1.0,
        image_keys=("image_rgb_0", "image_rgb_1"),
        max_pending_chunks=4,
        initial_progress="query_start",
    )

    processor.submit(make_pending(env_reward=0.0))
    stats = processor.close(drain=True)

    assert stats["committed"] == 1
    assert len(store.payloads) == 1
    assert np.isclose(store.payloads[0]["reward"], 0.3)
    assert store.payloads[0]["info"]["reward_model_progress"] == 0.5
    assert client.requests[0]["query_indices"] == [0, 1]


def test_async_remote_progress_zeroes_terminal_state_by_default():
    store = ListDataStore()
    client = FakeProgressClient([{"progress": [0.2, 0.6]}, {"progress": [0.95]}])
    processor = AsyncRemoteProgressRLTRewardProcessor(
        data_store=store,
        client=client,
        reward_type="env_plus_potential_delta",
        gamma=0.99,
        scale=1.0,
        image_keys=("image_rgb_0", "image_rgb_1"),
        max_pending_chunks=4,
        initial_progress="query_start",
    )

    processor.submit(make_pending(env_reward=0.0, done=False, chunk_index=0, discount=0.99**2))
    processor.submit(make_pending(env_reward=0.0, done=True, chunk_index=1, discount=0.99**2))
    stats = processor.close(drain=True)

    assert stats["committed"] == 2
    assert np.isclose(store.payloads[0]["reward"], (0.99**2) * 0.6 - 0.2)
    assert np.isclose(store.payloads[1]["reward"], -0.6)
    assert store.payloads[1]["info"]["reward_potential_discount"] == 0.0


def test_async_remote_progress_writes_progress_event():
    store = ListDataStore()
    events = []
    client = FakeProgressClient([{"progress": [0.2, 0.5]}])
    processor = AsyncRemoteProgressRLTRewardProcessor(
        data_store=store,
        client=client,
        reward_type="env_plus_potential_delta",
        gamma=0.99,
        scale=1.0,
        image_keys=("image_rgb_0", "image_rgb_1"),
        max_pending_chunks=4,
        initial_progress="query_start",
        progress_event_writer=events.append,
        reward_remote_url="http://127.0.0.1:50055",
    )

    processor.submit(make_pending(env_reward=1.0))
    stats = processor.close(drain=True)

    assert stats["committed"] == 1
    assert len(events) == 1
    event = events[0]
    assert event["role"] == "reward_progress"
    assert event["status"] == "ok"
    assert event["source"] == "remote_progress"
    assert event["episode_id"] == 0
    assert event["chunk_index"] == 0
    assert event["boundary_index"] == 1
    assert event["previous_boundary_index"] == 0
    assert event["trajectory_indices"] == [0, 1]
    assert event["query_indices"] == [0, 1]
    assert event["absolute_query_indices"] == [0, 1]
    assert event["progress_values"] == [0.2, 0.5]
    assert event["previous_progress"] == 0.2
    assert event["progress"] == 0.5
    assert event["env_reward"] == 1.0
    assert np.isclose(event["computed_reward"], store.payloads[0]["reward"])
    assert event["reward_type"] == "env_plus_potential_delta"
    assert event["executed_steps"] == 2
    assert event["env_steps"] == 2
    assert event["reward_remote_url"] == "http://127.0.0.1:50055"


def test_async_remote_progress_sends_only_latest_boundary_after_first_chunk():
    store = ListDataStore()
    client = FakeProgressClient([{"progress": [0.2, 0.5]}, {"progress": [0.8]}])
    processor = AsyncRemoteProgressRLTRewardProcessor(
        data_store=store,
        client=client,
        reward_type="progress_delta",
        gamma=0.99,
        scale=1.0,
        image_keys=("image_rgb_0", "image_rgb_1"),
        max_pending_chunks=4,
        initial_progress="query_start",
    )

    processor.submit(make_pending(env_reward=0.0, chunk_index=0))
    processor.submit(make_pending(env_reward=0.0, chunk_index=1))
    stats = processor.close(drain=True)

    assert stats["committed"] == 2
    assert [request["query_indices"] for request in client.requests] == [[0, 1], [0]]
    assert [request["absolute_query_indices"] for request in client.requests] == [[0, 1], [2]]
    assert [request["trajectory_indices"] for request in client.requests] == [[0, 1], [2]]
    assert [len(request["trajectory"]) for request in client.requests] == [2, 1]
    assert client.requests[0]["session_id"] == client.requests[1]["session_id"]
    assert client.requests[0]["metadata"]["session_id"] == client.requests[0]["session_id"]
    assert np.isclose(store.payloads[0]["reward"], 0.3)
    assert np.isclose(store.payloads[1]["reward"], 0.3)


def test_async_remote_progress_batches_contiguous_episode_boundaries():
    store = ListDataStore()
    client = FakeProgressClient([{"progress": [0.1, 0.3, 0.6, 0.9]}])
    processor = AsyncRemoteProgressRLTRewardProcessor(
        data_store=store,
        client=client,
        reward_type="progress_delta",
        gamma=0.99,
        scale=1.0,
        image_keys=("image_rgb_0", "image_rgb_1"),
        max_pending_chunks=8,
        batch_size=3,
        max_wait_ms=50,
        initial_progress="query_start",
    )

    processor.submit(make_pending(env_reward=0.0, chunk_index=0))
    processor.submit(make_pending(env_reward=0.0, chunk_index=1))
    processor.submit(make_pending(env_reward=0.0, chunk_index=2))
    stats = processor.close(drain=True)

    assert stats["committed"] == 3
    assert stats["last_batch_size"] == 3
    assert len(client.requests) == 1
    assert client.requests[0]["trajectory_indices"] == [0, 1, 2, 3]
    assert client.requests[0]["query_indices"] == [0, 1, 2, 3]
    assert client.requests[0]["absolute_query_indices"] == [0, 1, 2, 3]
    assert [payload["info"]["reward_model_batch_size"] for payload in store.payloads] == [3, 3, 3]
    np.testing.assert_allclose(
        [payload["reward"] for payload in store.payloads],
        [0.2, 0.3, 0.3],
        atol=1e-6,
    )


def test_robodopamine_http_service_uses_compact_query_indices():
    class FakeEngine:
        def __init__(self):
            self.calls = []

        def predict(self, *, transitions, query_indices, trajectory_start_idx, task, goal_image, request_label):
            self.calls.append(
                {
                    "step_in_episode": [item.step_in_episode for item in transitions],
                    "query_indices": list(query_indices),
                    "trajectory_start_idx": int(trajectory_start_idx),
                    "task": task,
                    "goal_image": goal_image,
                    "request_label": request_label,
                }
            )
            return [0.7]

    engine = FakeEngine()
    service = RoboDopamineProgressHttpService(
        engine=engine,
        goal_provider=None,
        require_goal=False,
        response_key="fake",
    )

    result = service.predict_progress(
        {
            "episode_id": 3,
            "task": "open drawer",
            "trajectory": [observation_to_reward_request_item(make_obs()), observation_to_reward_request_item(make_obs())],
            "trajectory_indices": [0, 2],
            "query_indices": [1],
            "absolute_query_indices": [2],
        }
    )

    assert result["progress"] == [0.7]
    assert engine.calls == [
        {
            "step_in_episode": [0, 2],
            "query_indices": [1],
            "trajectory_start_idx": 0,
            "task": "open drawer",
            "goal_image": None,
            "request_label": "3_2_2",
        }
    ]


def test_robodopamine_http_service_caches_episode_start_for_compact_forward_requests():
    class FakeEngine:
        eval_modes = ("forward",)

        def __init__(self):
            self.calls = []
            self.responses = [[0.1, 0.4], [0.7]]

        def predict(self, *, transitions, query_indices, trajectory_start_idx, task, goal_image, request_label):
            self.calls.append(
                {
                    "step_in_episode": [item.step_in_episode for item in transitions],
                    "query_indices": list(query_indices),
                    "trajectory_start_idx": int(trajectory_start_idx),
                    "request_label": request_label,
                }
            )
            return self.responses.pop(0)

    service = RoboDopamineProgressHttpService(
        engine=FakeEngine(),
        goal_provider=None,
        require_goal=False,
        response_key="fake",
    )

    first = service.predict_progress(
        {
            "episode_id": 5,
            "session_id": "actor-a",
            "task": "open drawer",
            "trajectory": [
                observation_to_reward_request_item(make_obs()),
                observation_to_reward_request_item(make_obs()),
            ],
            "trajectory_indices": [0, 1],
            "query_indices": [0, 1],
            "absolute_query_indices": [0, 1],
        }
    )
    second = service.predict_progress(
        {
            "episode_id": 5,
            "session_id": "actor-a",
            "task": "open drawer",
            "trajectory": [observation_to_reward_request_item(make_obs())],
            "trajectory_indices": [2],
            "query_indices": [0],
            "absolute_query_indices": [2],
            "trajectory_start_idx": 0,
            "metadata": {"done": True},
        }
    )

    assert first["progress"] == [0.1, 0.4]
    assert second["progress"] == [0.7]
    assert service.engine.calls == [
        {
            "step_in_episode": [0, 1],
            "query_indices": [0, 1],
            "trajectory_start_idx": 0,
            "request_label": "5_0_1",
        },
        {
            "step_in_episode": [0, 2],
            "query_indices": [1],
            "trajectory_start_idx": 0,
            "request_label": "5_2_2",
        },
    ]
    assert second["metadata"]["trajectory_indices"] == [0, 2]
    assert service.stats["episodes_cached"] == 0


def test_robodopamine_http_service_uses_contiguous_context_for_incremental_mode():
    class FakeEngine:
        eval_modes = ("forward", "backward", "incremental")

        def __init__(self):
            self.calls = []
            self.responses = [[0.2], [0.6]]

        def predict(self, *, transitions, query_indices, trajectory_start_idx, task, goal_image, request_label):
            self.calls.append(
                {
                    "step_in_episode": [item.step_in_episode for item in transitions],
                    "query_indices": list(query_indices),
                    "trajectory_start_idx": int(trajectory_start_idx),
                }
            )
            return self.responses.pop(0)

    service = RoboDopamineProgressHttpService(
        engine=FakeEngine(),
        goal_provider=None,
        require_goal=False,
        response_key="fake",
    )

    service.predict_progress(
        {
            "episode_id": 6,
            "session_id": "actor-a",
            "task": "open drawer",
            "trajectory": [
                observation_to_reward_request_item(make_obs()),
                observation_to_reward_request_item(make_obs()),
            ],
            "trajectory_indices": [0, 1],
            "query_indices": [1],
            "absolute_query_indices": [1],
        }
    )
    result = service.predict_progress(
        {
            "episode_id": 6,
            "session_id": "actor-a",
            "task": "open drawer",
            "trajectory": [observation_to_reward_request_item(make_obs())],
            "trajectory_indices": [2],
            "query_indices": [0],
            "absolute_query_indices": [2],
        }
    )

    assert result["progress"] == [0.6]
    assert service.engine.calls == [
        {"step_in_episode": [0, 1], "query_indices": [1], "trajectory_start_idx": 0},
        {"step_in_episode": [0, 1, 2], "query_indices": [2], "trajectory_start_idx": 0},
    ]


def test_robodopamine_http_service_rejects_incremental_missing_middle_frame():
    class FakeEngine:
        eval_modes = ("incremental",)

        def predict(self, **kwargs):
            raise AssertionError("engine should not be called when context is incomplete")

    service = RoboDopamineProgressHttpService(
        engine=FakeEngine(),
        goal_provider=None,
        require_goal=False,
        response_key="fake",
    )

    with pytest.raises(RuntimeError, match="contiguous cached frames"):
        service.predict_progress(
            {
                "episode_id": 7,
                "session_id": "actor-a",
                "task": "open drawer",
                "trajectory": [
                    observation_to_reward_request_item(make_obs()),
                    observation_to_reward_request_item(make_obs()),
                ],
                "trajectory_indices": [0, 2],
                "query_indices": [1],
                "absolute_query_indices": [2],
                "metadata": {"done": True},
            }
        )
    assert service.stats["episodes_cached"] == 0


def test_robometer_prefix_per_query_builds_one_prefix_sample_per_query():
    class FakeNativeBackend:
        def __init__(self):
            self.frame_lengths = []

        def predict_progress_samples(self, samples):
            self.frame_lengths = [
                int(sample["trajectory"]["frames"].shape[0])
                for sample in samples
            ]
            return [[0.1], [0.3], [0.5], [0.7]]

    backend = FakeNativeBackend()
    service = RoboMeterProgressHttpService(
        backend="native",
        native_backend=backend,
        robometer_url="http://unused",
        image_keys=("image_rgb_0", "image_rgb_1"),
        view_mode="average_two",
        query_mode="prefix_per_query",
        max_history_frames=8,
        use_frame_steps=False,
        timeout=1.0,
        response_key="robometer",
        clamp_progress=False,
    )

    result = service.predict_progress(
        {
            "episode_id": 3,
            "session_id": "test-session",
            "task": "open drawer",
            "trajectory": [
                observation_to_reward_request_item(make_obs()),
                observation_to_reward_request_item(make_obs()),
                observation_to_reward_request_item(make_obs()),
            ],
            "trajectory_indices": [0, 1, 2],
            "query_indices": [1, 2],
            "absolute_query_indices": [1, 2],
        }
    )

    assert backend.frame_lengths == [2, 2, 3, 3]
    np.testing.assert_allclose(result["progress"], [0.2, 0.6])
    assert result["metadata"]["query_mode"] == "prefix_per_query"


def test_robometer_native_backend_groups_samples_by_prefix_length():
    import threading
    import torch

    class FakeCollator:
        def __init__(self):
            self.batches = []

        def __call__(self, samples):
            lengths = [int(sample["trajectory"]["frames"].shape[0]) for sample in samples]
            assert len(set(lengths)) == 1
            ids = [int(sample["trajectory"]["id"]) for sample in samples]
            self.batches.append(ids)
            return {"progress_inputs": {"sample_ids": ids}}

    def sample(sample_id, frame_count):
        return {
            "trajectory": {
                "id": int(sample_id),
                "frames": np.zeros((frame_count, 4, 4, 3), dtype=np.uint8),
            }
        }

    backend = object.__new__(RoboMeterNativeBackend)
    backend.torch = torch
    backend.device = torch.device("cpu")
    backend.forward_batch_size = 2
    backend.lock = threading.Lock()
    backend.batch_collator = FakeCollator()
    backend._compute_progress_outputs = lambda inputs: [[float(value) / 100.0] for value in inputs["sample_ids"]]

    outputs = backend.predict_progress_samples(
        [
            sample(10, 2),
            sample(20, 3),
            sample(30, 2),
            sample(40, 3),
            sample(50, 2),
        ]
    )

    assert outputs == [[0.1], [0.2], [0.3], [0.4], [0.5]]
    assert backend.batch_collator.batches == [[10, 30], [50], [20, 40]]


def test_robometer_progress_args_disable_frame_steps_by_default():
    defaults = parse_robometer_progress_args(["--backend", "native"])
    assert defaults.image_keys == ["image_rgb_0"]
    assert defaults.view_mode == "first"
    assert defaults.use_frame_steps is False
    assert parse_robometer_progress_args(["--backend", "native", "--use-frame-steps"]).use_frame_steps is True
    assert parse_robometer_progress_args(["--backend", "native", "--no-frame-steps"]).use_frame_steps is False


def test_async_remote_progress_processor_raises_on_error_without_commit():
    class FailingClient(FakeProgressClient):
        def predict_progress(self, request, *, expected_count=None):
            raise RuntimeError("service down")

    store = ListDataStore()
    processor = AsyncRemoteProgressRLTRewardProcessor(
        data_store=store,
        client=FailingClient([]),
        reward_type="progress_abs",
        gamma=0.99,
        max_pending_chunks=4,
    )

    processor.submit(make_pending(env_reward=1.0))
    with pytest.raises(RuntimeError, match="reward worker failed"):
        processor.close(drain=True)

    assert processor.stats()["failed"] == 1
    assert store.payloads == []


def test_async_remote_progress_event_write_failure_does_not_prevent_commit():
    store = ListDataStore()
    client = FakeProgressClient([{"progress": [0.2, 0.5]}])

    def failing_writer(event):
        raise RuntimeError("progress event sink failed")

    processor = AsyncRemoteProgressRLTRewardProcessor(
        data_store=store,
        client=client,
        reward_type="progress_delta",
        gamma=0.99,
        max_pending_chunks=4,
        image_keys=("image_rgb_0", "image_rgb_1"),
        initial_progress="query_start",
        progress_event_writer=failing_writer,
    )

    processor.submit(make_pending(env_reward=1.0, chunk_index=0))
    stats = processor.close(drain=True)

    assert stats["committed"] == 1
    assert stats["failed"] == 0
    assert stats["progress_event_write_failed"] == 1
    assert len(store.payloads) == 1
    assert np.isclose(store.payloads[0]["reward"], 0.3)


def test_progress_reward_rejects_env_source():
    cfg = OmegaConf.create({"reward": {"type": "progress_abs", "source": "env"}})
    with pytest.raises(ValueError, match="requires reward.source=remote_progress"):
        build_rlt_reward_processor(cfg, data_store=ListDataStore(), gamma=0.99)


def test_progress_reward_rejects_on_error_config():
    cfg = OmegaConf.create(
        {
            "reward": {
                "type": "progress_abs",
                "source": "remote_progress",
                "on_error": "fallback_sparse",
                "remote": {"url": "http://127.0.0.1:50052"},
                "trajectory": {"image_keys": ["image_rgb_0"]},
            }
        }
    )
    with pytest.raises(ValueError, match="reward.on_error is not supported"):
        build_rlt_reward_processor(cfg, data_store=ListDataStore(), gamma=0.99)


def test_progress_reward_rejects_terminal_potential_config():
    cfg = OmegaConf.create(
        {
            "reward": {
                "type": "env_plus_potential_delta",
                "source": "remote_progress",
                "terminal_potential": "bootstrap",
                "remote": {"url": "http://127.0.0.1:50052"},
                "trajectory": {"image_keys": ["image_rgb_0"]},
            }
        }
    )
    with pytest.raises(ValueError, match="terminal_potential is no longer supported"):
        build_rlt_reward_processor(cfg, data_store=ListDataStore(), gamma=0.99)


def test_reward_payload_requires_configured_image_keys():
    with pytest.raises(KeyError, match="missing reward image keys"):
        build_reward_payload(make_obs(), image_keys=("image_rgb_0", "missing_view"))
