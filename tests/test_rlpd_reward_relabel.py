from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf
import pytest

from examples.libero.rlpd.scripts.relabel_lerobot_expert_replay import (
    _require_remote_reward_cfg,
    predict_episode_progress_steps,
)
from vla_rl.data import Observation


class FakeProgressClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def predict_progress(self, request, *, expected_count=None):
        self.requests.append(request)
        values = list(self.responses.pop(0))
        if expected_count is not None:
            assert len(values) == expected_count
        return values


def _obs(task="pick up the block"):
    return Observation(
        images={
            "image_rgb_0": np.zeros((4, 4, 3), dtype=np.uint8),
            "image_rgb_1": np.ones((4, 4, 3), dtype=np.uint8),
        },
        proprio=np.zeros((8,), dtype=np.float32),
        task=task,
    )


def test_offline_relabel_progress_requests_match_online_query_start_protocol():
    client = FakeProgressClient([[0.2, 0.5], [0.7]])
    steps = predict_episode_progress_steps(
        client,
        [_obs(), _obs(), _obs()],
        episode_id=3,
        task_prompt="pick up the block",
        task_id=6,
        initial_progress="query_start",
        image_keys=("image_rgb_0", "image_rgb_1"),
    )

    assert len(steps) == 2
    assert steps[0].previous_progress == 0.2
    assert steps[0].progress == 0.5
    assert steps[1].previous_progress == 0.5
    assert steps[1].progress == 0.7
    assert client.requests[0]["trajectory_indices"] == [0, 1]
    assert client.requests[0]["query_indices"] == [0, 1]
    assert client.requests[0]["absolute_query_indices"] == [0, 1]
    assert client.requests[1]["trajectory_indices"] == [0, 2]
    assert client.requests[1]["query_indices"] == [1]
    assert client.requests[1]["absolute_query_indices"] == [2]
    assert client.requests[1]["metadata"]["done"] is True


def test_offline_relabel_progress_requests_support_zero_initial_progress():
    client = FakeProgressClient([[0.5], [0.9]])
    steps = predict_episode_progress_steps(
        client,
        [_obs(), _obs(), _obs()],
        episode_id=4,
        task_prompt="pick up the block",
        task_id=6,
        initial_progress="zero",
        image_keys=("image_rgb_0",),
    )

    assert steps[0].previous_progress == 0.0
    assert steps[0].progress == 0.5
    assert steps[1].previous_progress == 0.5
    assert steps[1].progress == 0.9
    assert client.requests[0]["query_indices"] == [1]
    assert client.requests[0]["absolute_query_indices"] == [1]


def test_relabel_episode_writes_dense_transition_rewards(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    from examples.libero.rlpd.scripts.relabel_lerobot_expert_replay import relabel_episode
    from vla_rl.algorithms.rlpd import ObservationBuilder

    source_repo = tmp_path / "repo"
    source_repo.mkdir()
    rows = []
    for idx in range(3):
        image_path = f"image_{idx}.png"
        wrist_path = f"wrist_{idx}.png"
        Image.fromarray(np.full((6, 6, 3), idx * 40, dtype=np.uint8)).save(source_repo / image_path)
        Image.fromarray(np.full((6, 6, 3), 255 - idx * 40, dtype=np.uint8)).save(source_repo / wrist_path)
        rows.append(
            {
                "image": {"path": image_path},
                "wrist_image": {"path": wrist_path},
                "state": [0.0] * 8,
                "actions": [0.1 * idx] * 7,
                "frame_index": idx,
                "episode_index": 11,
                "task_index": 6,
            }
        )
    parquet_path = tmp_path / "episode_000000.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    client = FakeProgressClient([[0.1, 0.4], [0.8]])
    relabelled = relabel_episode(
        parquet_path,
        obs_builder=ObservationBuilder(image_keys=("image_rgb_0", "image_rgb_1")),
        client=client,
        episode_id=0,
        task_prompt="pick up the block",
        suite_name="libero_spatial",
        task_id=6,
        action_dim=7,
        image_size=4,
        gamma=0.9,
        step_reward=0.0,
        terminal_reward=1.0,
        reward_type="env_plus_potential_delta",
        reward_scale=1.0,
        initial_progress="query_start",
        image_keys=("image_rgb_0", "image_rgb_1"),
        source_repo=source_repo,
    )

    transitions = [transition for transition, _progress in relabelled]
    assert len(transitions) == 2
    assert np.isclose(transitions[0].reward, 0.9 * 0.4 - 0.1)
    assert np.isclose(transitions[1].reward, 1.0 - 0.4)
    assert transitions[0].info["reward_type"] == "env_plus_potential_delta"
    assert transitions[0].info["reward_model_previous_progress"] == 0.1
    assert transitions[0].info["reward_model_progress"] == 0.4
    assert transitions[1].info["env_reward"] == 1.0
    assert transitions[1].info["reward_potential_discount"] == 0.0
    assert transitions[1].discount == 0.0


def test_relabel_episode_with_fake_http_progress_server(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer

    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    from examples.libero.rlpd.scripts.relabel_lerobot_expert_replay import relabel_episode
    from vla_rl.algorithms.rlpd import ObservationBuilder
    from vla_rl.rewards.progress import RemoteProgressClient
    from vla_rl.runtime.remote_http import make_pickle_rpc_handler

    class ProgressService:
        def __init__(self):
            self.responses = [[0.0, 0.25], [0.75]]
            self.requests = []

        def dispatch(self, method, kwargs):
            assert method == "predict_progress"
            request = kwargs["request"]
            self.requests.append(request)
            return {"progress": self.responses.pop(0)}

    source_repo = tmp_path / "repo"
    source_repo.mkdir()
    rows = []
    for idx in range(3):
        image_path = f"http_image_{idx}.png"
        wrist_path = f"http_wrist_{idx}.png"
        Image.fromarray(np.full((6, 6, 3), idx * 60, dtype=np.uint8)).save(source_repo / image_path)
        Image.fromarray(np.full((6, 6, 3), 200 - idx * 40, dtype=np.uint8)).save(source_repo / wrist_path)
        rows.append(
            {
                "image": {"path": image_path},
                "wrist_image": {"path": wrist_path},
                "state": [0.0] * 8,
                "actions": [0.0] * 7,
                "frame_index": idx,
                "episode_index": 12,
                "task_index": 6,
            }
        )
    parquet_path = tmp_path / "episode_000001.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    service = ProgressService()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_pickle_rpc_handler(service.dispatch))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = RemoteProgressClient(f"http://127.0.0.1:{server.server_port}")
    try:
        relabelled = relabel_episode(
            parquet_path,
            obs_builder=ObservationBuilder(image_keys=("image_rgb_0", "image_rgb_1")),
            client=client,
            episode_id=5,
            task_prompt="pick up the block",
            suite_name="libero_spatial",
            task_id=6,
            action_dim=7,
            image_size=4,
            gamma=0.8,
            step_reward=0.0,
            terminal_reward=1.0,
            reward_type="env_plus_potential_delta",
            reward_scale=1.0,
            initial_progress="query_start",
            image_keys=("image_rgb_0", "image_rgb_1"),
            source_repo=source_repo,
        )
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    transitions = [transition for transition, _progress in relabelled]
    assert np.isclose(transitions[0].reward, 0.8 * 0.25 - 0.0)
    assert np.isclose(transitions[1].reward, 1.0 - 0.25)
    assert transitions[1].info["reward_potential_discount"] == 0.0
    assert service.requests[0]["query_indices"] == [0, 1]
    assert service.requests[1]["query_indices"] == [1]


def test_offline_relabel_rejects_terminal_potential_config():
    cfg = OmegaConf.create(
        {"reward": {"source": "remote_progress", "terminal_potential": "bootstrap"}}
    )
    with pytest.raises(ValueError, match="terminal_potential is no longer supported"):
        _require_remote_reward_cfg(cfg)
