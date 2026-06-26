from __future__ import annotations

from vla_rl.rewards.processor import (
    AsyncRemoteProgressRewardProcessor,
    BaseRewardProcessor,
    MetricWriter,
    PendingRewardTransition,
    SparseRewardProcessor,
    build_reward_processor,
    observation_to_reward_payload,
)

PendingRLTRewardTransition = PendingRewardTransition
BaseRLTRewardProcessor = BaseRewardProcessor
SparseRLTRewardProcessor = SparseRewardProcessor
AsyncRemoteProgressRLTRewardProcessor = AsyncRemoteProgressRewardProcessor
build_rlt_reward_processor = build_reward_processor

__all__ = [
    "AsyncRemoteProgressRLTRewardProcessor",
    "BaseRLTRewardProcessor",
    "MetricWriter",
    "PendingRLTRewardTransition",
    "SparseRLTRewardProcessor",
    "build_rlt_reward_processor",
    "observation_to_reward_payload",
]
