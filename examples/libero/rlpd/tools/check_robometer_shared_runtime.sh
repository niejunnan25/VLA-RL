#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/robometer_shared_runtime_env.sh"

ROBOMETER_ALLOW_EXTERNAL_VENV="${ROBOMETER_ALLOW_EXTERNAL_VENV:-0}" \
  "$ROBOMETER_PYTHON" - <<'PY_CHECK'
import ctypes
import os
from pathlib import Path
import socket
import sys

import decord
import fastapi
import omegaconf
import torch
import uvicorn

ctypes.CDLL("libcudnn_graph.so.9")
cudnn_version = torch.backends.cudnn.version()
site_packages = Path(os.environ["ROBOMETER_SITE_PACKAGES"]).resolve()
expected_python_root = Path(os.environ["UV_PYTHON_INSTALL_DIR"]).resolve()
expected_python_realpath = Path(os.environ["ROBOMETER_PYTHON_REALPATH"]).resolve()
python_realpath = Path(sys.executable).resolve()
expected_cudnn_root = (site_packages / "nvidia" / "cudnn" / "lib").resolve()

if python_realpath != expected_python_realpath:
    raise RuntimeError(
        f"Python executable changed after runtime setup: {python_realpath} != "
        f"{expected_python_realpath}"
    )
if os.environ["ROBOMETER_ALLOW_EXTERNAL_VENV"] != "1":
    try:
        python_realpath.relative_to(expected_python_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Python resolves outside the shared uv install: {python_realpath}"
        ) from exc

cudnn_paths = sorted(
    {
        Path(line.rsplit(maxsplit=1)[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libcudnn" in line and line.rsplit(maxsplit=1)[-1].startswith("/")
    }
)
if not cudnn_paths:
    raise RuntimeError("No cuDNN library was loaded during validation")
for path in cudnn_paths:
    try:
        path.relative_to(expected_cudnn_root)
    except ValueError as exc:
        raise RuntimeError(f"Host cuDNN leaked into RoboMeter: {path}") from exc

print(f"host={socket.gethostname()}")
print(f"python={sys.executable}")
print(f"python_realpath={python_realpath}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cudnn={cudnn_version}")
print("cudnn_libraries=" + ",".join(str(path) for path in cudnn_paths))
PY_CHECK
