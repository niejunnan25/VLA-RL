from __future__ import annotations

from vla_rl.policies import ReferencePolicyClient
from vla_rl.policies.base import PolicyBackend


def create_reference_policy(cfg) -> PolicyBackend:
    policy_cfg = cfg.policy
    if str(policy_cfg.type) != "reference_client":
        raise ValueError(f"unsupported policy.type={policy_cfg.type!r}")
    return ReferencePolicyClient(
        url=str(policy_cfg.url),
        action_dim=int(policy_cfg.get("action_dim", cfg.algorithm.action_dim)),
        timeout=float(policy_cfg.get("timeout", 120.0)),
    )
