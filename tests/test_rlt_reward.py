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
from vla_rl.rewards import compute_progress_reward, normalize_progress_response
from scripts.serve_robodopamine_progress_http import RoboDopamineProgressHttpService


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


def make_pending(env_reward=0.0, done=False, episode_id=0, chunk_index=0):
    transition = Transition(
        obs=make_rlt_state(0.0),
        next_obs=make_rlt_state(1.0),
        action=np.zeros((4,), dtype=np.float32),
        reward=float(env_reward),
        done=bool(done),
        truncated=False,
        discount=0.99**2,
        executed_steps=2,
        env_steps=2,
        info={"critic_terminal": False},
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


def test_normalize_progress_response_requires_full_query_start_pair():
    with pytest.raises(ValueError, match="expected 2 progress values"):
        normalize_progress_response({"progress": [0.7]}, expected_count=2)


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
    assert stats["progress_event_write_failed"] == 0
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
    assert [request["query_indices"] for request in client.requests] == [[0, 1], [1]]
    assert [request["absolute_query_indices"] for request in client.requests] == [[0, 1], [2]]
    assert [request["trajectory_indices"] for request in client.requests] == [[0, 1], [0, 2]]
    assert [len(request["trajectory"]) for request in client.requests] == [2, 2]
    assert np.isclose(store.payloads[0]["reward"], 0.3)
    assert np.isclose(store.payloads[1]["reward"], 0.3)


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


def test_async_remote_progress_processor_falls_back_to_sparse_on_error():
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
        on_error="fallback_sparse",
    )

    processor.submit(make_pending(env_reward=1.0))
    stats = processor.close(drain=True)

    assert stats["fallback_sparse"] == 1
    assert store.payloads[0]["reward"] == 1.0
    assert "reward_model_error" in store.payloads[0]["info"]


def test_async_remote_progress_fallback_uses_failed_end_as_next_boundary():
    class FlakyClient(FakeProgressClient):
        def __init__(self):
            super().__init__([])
            self.calls = 0

        def predict_progress(self, request, *, expected_count=None):
            self.requests.append(request)
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary outage")
            return normalize_progress_response({"progress": [0.4, 0.7]}, expected_count=expected_count)

    store = ListDataStore()
    client = FlakyClient()
    processor = AsyncRemoteProgressRLTRewardProcessor(
        data_store=store,
        client=client,
        reward_type="progress_delta",
        gamma=0.99,
        max_pending_chunks=4,
        initial_progress="query_start",
        on_error="fallback_sparse",
    )

    processor.submit(make_pending(env_reward=1.0, chunk_index=0))
    processor.submit(make_pending(env_reward=0.0, chunk_index=1))
    stats = processor.close(drain=True)

    assert stats["fallback_sparse"] == 1
    assert [request["query_indices"] for request in client.requests] == [[0, 1], [1, 2]]
    assert [request["absolute_query_indices"] for request in client.requests] == [[0, 1], [1, 2]]
    assert [request["trajectory_indices"] for request in client.requests] == [[0, 1], [0, 1, 2]]
    assert [len(request["trajectory"]) for request in client.requests] == [2, 3]
    assert store.payloads[0]["reward"] == 1.0
    assert np.isclose(store.payloads[1]["reward"], 0.3)
    assert store.payloads[1]["info"]["reward_model_previous_progress"] == 0.4


def test_async_remote_progress_zero_initial_requeries_previous_after_fallback():
    class FlakyClient(FakeProgressClient):
        def __init__(self):
            super().__init__([])
            self.calls = 0

        def predict_progress(self, request, *, expected_count=None):
            self.requests.append(request)
            self.calls += 1
            if self.calls == 1:
                return normalize_progress_response({"progress": [0.5]}, expected_count=expected_count)
            if self.calls == 2:
                raise RuntimeError("temporary outage")
            return normalize_progress_response({"progress": [0.6, 0.9]}, expected_count=expected_count)

    store = ListDataStore()
    client = FlakyClient()
    processor = AsyncRemoteProgressRLTRewardProcessor(
        data_store=store,
        client=client,
        reward_type="progress_delta",
        gamma=0.99,
        max_pending_chunks=4,
        initial_progress="zero",
        on_error="fallback_sparse",
    )

    processor.submit(make_pending(env_reward=0.0, chunk_index=0))
    processor.submit(make_pending(env_reward=1.0, chunk_index=1))
    processor.submit(make_pending(env_reward=0.0, chunk_index=2))
    stats = processor.close(drain=True)

    assert stats["fallback_sparse"] == 1
    assert [request["query_indices"] for request in client.requests] == [[1], [1], [1, 2]]
    assert [request["absolute_query_indices"] for request in client.requests] == [[1], [2], [2, 3]]
    assert [request["trajectory_indices"] for request in client.requests] == [[0, 1], [0, 2], [0, 2, 3]]
    assert [len(request["trajectory"]) for request in client.requests] == [2, 2, 3]
    assert np.isclose(store.payloads[0]["reward"], 0.5)
    assert store.payloads[1]["reward"] == 1.0
    assert np.isclose(store.payloads[2]["reward"], 0.3)
    assert store.payloads[2]["info"]["reward_model_previous_progress"] == 0.6


def test_async_remote_progress_fallback_clears_terminal_episode_state():
    class FailingClient(FakeProgressClient):
        def predict_progress(self, request, *, expected_count=None):
            self.requests.append(request)
            raise RuntimeError("service down")

    store = ListDataStore()
    processor = AsyncRemoteProgressRLTRewardProcessor(
        data_store=store,
        client=FailingClient([]),
        reward_type="progress_abs",
        gamma=0.99,
        max_pending_chunks=4,
        on_error="fallback_sparse",
    )

    processor.submit(make_pending(env_reward=1.0, done=True))
    stats = processor.close(drain=True)

    assert stats["fallback_sparse"] == 1
    assert processor._episodes == {}
    assert store.payloads[0]["reward"] == 1.0



def test_progress_reward_rejects_env_source():
    cfg = OmegaConf.create({"reward": {"type": "progress_abs", "source": "env"}})
    with pytest.raises(ValueError, match="requires reward.source=remote_progress"):
        build_rlt_reward_processor(cfg, data_store=ListDataStore(), gamma=0.99)
