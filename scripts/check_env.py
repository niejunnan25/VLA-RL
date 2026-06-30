#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CORE_IMPORTS = [
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("omegaconf", "omegaconf"),
    ("agentlace", "agentlace"),
    ("vla_rl", "vla_rl"),
    ("residual_sac", "residual_sac"),
]

RESIDUAL_SAC_IMPORTS = [
    ("hydra", "hydra-core"),
    ("ml_collections", "ml-collections"),
    ("PIL", "Pillow"),
    ("cv2", "opencv-python"),
    ("tqdm", "tqdm"),
    ("gymnasium", "gymnasium"),
    ("gym", "gym"),
    ("websockets", "websockets"),
    ("msgpack", "msgpack"),
    ("requests", "requests"),
    ("transformers", "transformers"),
]

LOGGING_IMPORTS = [
    ("swanlab", "swanlab"),
    ("wandb", "wandb"),
]

PATH_CHECKS = [
    ("OPENPI_ROOT", False),
    ("POLICY_DIR", False),
    ("LIBERO_ROOT", False),
    ("LIBERO_DATASETS_ROOT", False),
    ("REWARD_GOAL_DATASET", False),
    ("REWARD_MODEL_REPO", False),
    ("REWARD_MODEL_CONDA_ENV", False),
    ("REWARD_MODEL_PATH", False),
    ("ROBOMETER_ROOT", False),
    ("ROBOMETER_MODEL_PATH", False),
    ("ROBOMETER_PROCESSED_DATASETS_PATH", False),
    ("ROBOMETER_SENTENCE_MODEL_PATH", False),
    ("HF_HOME", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a VLA-RL development environment.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if configured asset paths are missing. Without this flag, missing paths are warnings.",
    )
    parser.add_argument(
        "--skip-paths",
        action="store_true",
        help="Only check Python imports, not asset path variables.",
    )
    return parser.parse_args()


def import_status(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(module, "__version__", None)
    if version is None:
        return True, "ok"
    return True, f"ok ({version})"


def print_import_section(title: str, imports: Iterable[tuple[str, str]]) -> bool:
    print(f"\n{title}")
    print("-" * len(title))
    all_ok = True
    for module_name, package_name in imports:
        ok, detail = import_status(module_name)
        all_ok = all_ok and ok
        status = "OK" if ok else "MISSING"
        print(f"{status:8} {module_name:20} {package_name:24} {detail}")
    return all_ok


def print_path_section(strict: bool) -> bool:
    print("\nConfigured Paths")
    print("----------------")
    all_ok = True
    for var_name, required_by_default in PATH_CHECKS:
        value = os.environ.get(var_name, "").strip()
        if not value:
            status = "MISSING" if strict else "WARN"
            if strict:
                all_ok = False
            print(f"{status:8} {var_name:32} not set")
            continue
        path = Path(value).expanduser()
        exists = path.exists()
        if not exists and strict:
            all_ok = False
        status = "OK" if exists else ("MISSING" if strict else "WARN")
        print(f"{status:8} {var_name:32} {path}")
    return all_ok


def check_python_version() -> bool:
    print("Python")
    print("------")
    version = sys.version_info
    ok = (version.major, version.minor) >= (3, 10)
    status = "OK" if ok else "BAD"
    print(f"{status:8} python               {sys.version.split()[0]}")
    return ok


def check_repo_layout() -> bool:
    print("\nRepository Layout")
    print("-----------------")
    repo_root = Path(__file__).resolve().parents[1]
    required = [
        repo_root / "vla_rl",
        repo_root / "residual_sac",
        repo_root / "examples/libero/residual_sac/tools/launch_residual_sac.sh",
        repo_root / "examples/libero/residual_sac/configs/train_residual_chunk.yaml",
    ]
    ok = True
    for path in required:
        exists = path.exists()
        ok = ok and exists
        status = "OK" if exists else "MISSING"
        print(f"{status:8} {path.relative_to(repo_root)}")
    return ok


def main() -> int:
    args = parse_args()
    ok = True
    ok = check_python_version() and ok
    ok = check_repo_layout() and ok
    ok = print_import_section("Core Imports", CORE_IMPORTS) and ok
    ok = print_import_section("Residual SAC Imports", RESIDUAL_SAC_IMPORTS) and ok

    logging_ok = print_import_section("Logging Imports", LOGGING_IMPORTS)
    if not logging_ok:
        print("\nLogging imports are optional for local debug runs but required for formal online logging.")

    if not args.skip_paths:
        ok = print_path_section(strict=args.strict) and ok

    print("\nResult")
    print("------")
    if ok:
        print("OK: environment checks passed.")
        return 0
    print("FAILED: fix missing imports or paths above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
