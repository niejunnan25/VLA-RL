#!/usr/bin/env python3
from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vla_rl.policies.reference import create_reference_policy
from vla_rl.runtime.remote_http import make_pickle_rpc_handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a frozen reference policy for RLT action and prefix-feature inference.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--policy", choices=["openpi", "starvla"], default="openpi")
    parser.add_argument("--policy-root", default=None, help="Root checkout for the selected reference policy.")
    parser.add_argument("--openpi-root", default=None, help="Alias for --policy-root when --policy=openpi.")
    parser.add_argument("--config-name", default="pi0_libero")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_policy = create_reference_policy(
        args.policy,
        policy_root=args.policy_root,
        openpi_root=args.openpi_root,
        config_name=args.config_name,
        checkpoint_path=args.checkpoint_path,
        action_dim=args.action_dim,
        device=args.device,
    )

    def dispatch(method: str, kwargs: dict[str, Any]) -> Any:
        if method == "predict_action_with_features":
            obs = kwargs.pop("obs")
            call_kwargs = kwargs.pop("kwargs", {})
            if kwargs:
                call_kwargs = {**call_kwargs, **kwargs}
            return reference_policy.predict_action_with_features(obs, **call_kwargs)
        if method == "action_spec":
            return reference_policy.action_spec()
        raise ValueError(f"unsupported reference-policy RPC method: {method}")

    server = ThreadingHTTPServer((args.host, args.port), make_pickle_rpc_handler(dispatch))
    print(
        f"Serving RLT reference policy '{args.policy}' on http://{args.host}:{args.port} "
        f"with checkpoint {args.checkpoint_path}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
