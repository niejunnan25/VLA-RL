from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from vla_rl.rewards.processor import BaseRewardProcessor, PendingRewardTransition, build_reward_processor


PendingRLPDRewardTransition = PendingRewardTransition
BaseRLPDRewardProcessor = BaseRewardProcessor


def build_rlpd_reward_processor(
    cfg: DictConfig,
    *,
    data_store: Any,
    gamma: float,
    metric_writer=None,
    progress_event_writer=None,
) -> BaseRLPDRewardProcessor:
    """Build the shared single-transition progress reward processor."""

    return build_reward_processor(
        cfg,
        data_store=data_store,
        gamma=gamma,
        metric_writer=metric_writer,
        progress_event_writer=progress_event_writer,
    )
