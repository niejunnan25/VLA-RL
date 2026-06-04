from __future__ import annotations

from dataclasses import dataclass

from vla_rl.data.replay import ReplayBuffer
from vla_rl.data.schema import RolloutBatch


@dataclass(frozen=True)
class MixedBatch:
    batch: RolloutBatch
    mix: dict[str, int]


class MixedReplaySampler:
    """Sample online/offline replay batches with a fixed offline ratio."""

    def __init__(
        self,
        online_replay: ReplayBuffer,
        offline_replay: ReplayBuffer | None = None,
        offline_ratio: float = 0.5,
    ) -> None:
        if not 0.0 <= float(offline_ratio) <= 1.0:
            raise ValueError(f"offline_ratio must be in [0, 1], got {offline_ratio}")
        self.online_replay = online_replay
        self.offline_replay = offline_replay
        self.offline_ratio = float(offline_ratio)

    def sample(self, batch_size: int) -> MixedBatch:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        offline_n = int(round(batch_size * self.offline_ratio)) if self.offline_replay is not None else 0
        online_n = batch_size - offline_n
        if offline_n > 0 and (self.offline_replay is None or len(self.offline_replay) == 0):
            raise ValueError("offline_ratio requested offline samples, but offline replay is empty")
        if online_n > 0 and len(self.online_replay) == 0:
            raise ValueError("online replay is empty")

        transitions = []
        if online_n > 0:
            transitions.extend(self.online_replay.sample(online_n).transitions)
        if offline_n > 0:
            assert self.offline_replay is not None
            transitions.extend(self.offline_replay.sample(offline_n).transitions)
        batch = RolloutBatch(transitions=transitions)
        batch.validate()
        return MixedBatch(batch=batch, mix={"online": int(online_n), "offline": int(offline_n)})
