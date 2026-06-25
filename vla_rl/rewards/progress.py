from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np

from vla_rl.runtime.remote_http import RemoteHttpRpcClient


SPARSE_REWARD_TYPES = {"sparse", "env", "none"}
PROGRESS_REWARD_TYPES = {
    "progress_abs",
    "progress_delta",
    "potential_delta",
    "env_plus_potential_delta",
}
SUPPORTED_REWARD_TYPES = SPARSE_REWARD_TYPES | PROGRESS_REWARD_TYPES


class RemoteProgressClient:
    """Small pickle-HTTP client for reward services returning absolute progress.

    The RL side intentionally treats remote reward models as black boxes. The
    service receives a request dict containing trajectory observations and query
    indices, and returns absolute progress values for those query indices.
    """

    def __init__(
        self,
        url: str,
        *,
        method: str = "predict_progress",
        timeout: float = 120.0,
        retries: int = 1,
        retry_sleep: float = 0.5,
    ) -> None:
        self.method = str(method)
        self.client = RemoteHttpRpcClient(
            url=str(url),
            timeout=float(timeout),
            retries=int(retries),
            retry_sleep=float(retry_sleep),
            keep_alive=True,
        )

    def predict_progress(self, request: dict[str, Any], *, expected_count: int | None = None) -> list[float]:
        result = self.client.call(self.method, request=request)
        return normalize_progress_response(result, expected_count=expected_count)

    def close(self) -> None:
        self.client.close()


def compute_progress_reward(
    reward_type: str,
    *,
    env_reward: float,
    progress: float | None,
    previous_progress: float | None,
    gamma: float,
    executed_steps: int,
    scale: float = 1.0,
    discount: float | None = None,
) -> float:
    kind = str(reward_type)
    if kind not in SUPPORTED_REWARD_TYPES:
        raise ValueError(f"unsupported reward type: {kind}")
    if kind in SPARSE_REWARD_TYPES:
        return float(env_reward)
    if progress is None:
        raise ValueError(f"reward type {kind} requires progress")

    current = float(progress)
    previous = 0.0 if previous_progress is None else float(previous_progress)
    scaled = float(scale)
    if kind == "progress_abs":
        return scaled * current
    if kind == "progress_delta":
        return scaled * (current - previous)
    potential_discount = float(discount) if discount is not None else float(gamma) ** int(executed_steps)
    potential_delta = potential_discount * current - previous
    if kind == "potential_delta":
        return scaled * potential_delta
    if kind == "env_plus_potential_delta":
        return float(env_reward) + scaled * potential_delta
    raise AssertionError(f"unreachable reward type: {kind}")


def normalize_progress_response(result: Any, *, expected_count: int | None = None) -> list[float]:
    values = _extract_progress_values(result)
    if not values:
        raise ValueError(f"progress response did not contain progress values: {type(result).__name__}")
    out = [float(value) for value in values]
    if any(not np.isfinite(value) for value in out):
        raise ValueError(f"progress response contains non-finite values: {out}")
    if expected_count is not None and expected_count > 0 and len(out) != expected_count:
        if expected_count == 1:
            return [out[-1]]
        raise ValueError(f"expected {expected_count} progress values, got {len(out)}")
    return out


def _extract_progress_values(result: Any) -> list[float]:
    if isinstance(result, Number):
        return [float(result)]
    if isinstance(result, np.ndarray):
        return [float(value) for value in result.reshape(-1)]
    if isinstance(result, (list, tuple)):
        return [float(value) for value in result]
    if not isinstance(result, dict):
        raise ValueError(f"unsupported progress response type: {type(result).__name__}")

    for key in ("progress", "progresses", "progress_values", "rewards"):
        if key in result:
            return _extract_progress_values(result[key])

    for key in ("progress_predictions_by_key", "progress_by_key"):
        if key in result:
            averaged = _average_progress_by_key(result[key])
            if averaged:
                return averaged
    raise ValueError(f"progress response dict missing progress keys: {sorted(result)}")


def _average_progress_by_key(progress_by_key: Any) -> list[float]:
    if not isinstance(progress_by_key, dict):
        return []
    series: list[list[float]] = []
    for value in progress_by_key.values():
        if isinstance(value, dict):
            payload = None
            for key in ("progress", "progresses", "progress_values"):
                if key in value:
                    payload = value[key]
                    break
        else:
            payload = value
        if payload is None:
            continue
        series.append([float(item) for item in np.asarray(payload).reshape(-1)])
    if not series:
        return []
    min_len = min(len(items) for items in series)
    if min_len == 0:
        return []
    return [float(np.mean([items[idx] for items in series])) for idx in range(min_len)]
