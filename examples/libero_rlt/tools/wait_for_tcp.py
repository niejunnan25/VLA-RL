#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait until TCP ports accept connections.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ports", nargs="+", type=int, required=True)
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    parser.add_argument("--sleep-sec", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deadline = time.time() + float(args.timeout_sec)
    pending = set(int(port) for port in args.ports)
    while pending:
        for port in list(pending):
            with socket.socket() as sock:
                sock.settimeout(1.0)
                try:
                    sock.connect((args.host, port))
                except OSError:
                    continue
            print(f"ready: {args.host}:{port}", flush=True)
            pending.remove(port)
        if pending and time.time() >= deadline:
            raise TimeoutError(f"timed out waiting for ports: {sorted(pending)}")
        if pending:
            time.sleep(float(args.sleep_sec))


if __name__ == "__main__":
    main()
