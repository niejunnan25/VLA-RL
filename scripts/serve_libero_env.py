#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Start the external LIBERO env HTTP server used by VLA-RL.",
        add_help=True,
    )
    parser.add_argument(
        "--serl-torch-root",
        default="/vla/users/niejunnan/codebase/serl_torch-rlt-merge",
        help="Path to the validated serl_torch checkout that provides examples/libero/scripts/serve_env.py.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main() -> None:
    args, passthrough = parse_args()
    root = Path(args.serl_torch_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"serl_torch root does not exist: {root}")
    for path in (root, root / "serl_launcher"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from examples.libero.scripts.serve_env import main as serve_env_main

    sys.argv = [sys.argv[0], *passthrough]
    serve_env_main()


if __name__ == "__main__":
    main()
