from __future__ import annotations

from typing import Any

import numpy as np
from agentlace.data.data_store import DataStoreBase


def make_agentlace_replay_store(replay: Any) -> Any:
    class AgentlaceReplayStore(DataStoreBase):
        def __init__(self) -> None:
            self._latest_data_id = 0

        def insert(self, data: Any) -> None:
            replay.add(data)
            self._latest_data_id += 1

        def latest_data_id(self) -> int:
            return int(self._latest_data_id)

        def get_latest_data(self, from_id: int) -> list[Any]:
            del from_id
            return []

        def __len__(self) -> int:
            return len(replay)

    return AgentlaceReplayStore()


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
