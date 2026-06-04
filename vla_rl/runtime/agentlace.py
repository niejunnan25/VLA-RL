from __future__ import annotations

from typing import Any

import numpy as np



def make_agentlace_replay_store(agentlace: Any, replay: Any) -> Any:
    class AgentlaceReplayStore(agentlace.DataStoreBase):
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


def make_trainer_config(agentlace: Any, trainer_port: int, broadcast_port: int, request_types: list[str]) -> Any:
    return agentlace.TrainerConfig(
        port_number=int(trainer_port),
        broadcast_port=int(broadcast_port),
        request_types=list(request_types),
    )


def import_agentlace() -> Any:
    try:
        from agentlace.data.data_store import DataStoreBase, QueuedDataStore
        from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer
    except ImportError as exc:
        raise ImportError(
            "Agentlace transport requires the 'agentlace' package. Install VLA-RL runtime dependencies before "
            "running an Agentlace example."
        ) from exc

    class _AgentlaceModule:
        pass

    module = _AgentlaceModule()
    module.DataStoreBase = DataStoreBase
    module.QueuedDataStore = QueuedDataStore
    module.TrainerClient = TrainerClient
    module.TrainerConfig = TrainerConfig
    module.TrainerServer = TrainerServer
    return module


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
