#!/usr/bin/env bash
set -euo pipefail

ROBOMETER_ROOT="${ROBOMETER_ROOT:-/vla/users/niejunnan/workspace/robometer}"
ROBOMETER_SHARED_RUNTIME_ROOT="${ROBOMETER_SHARED_RUNTIME_ROOT:-/vla/users/niejunnan/runtime/robometer}"
ROBOMETER_VENV="${ROBOMETER_VENV:-$ROBOMETER_ROOT/.venv-shared}"
ROBOMETER_PYTHON_VERSION="${ROBOMETER_PYTHON_VERSION:-3.10.20}"
ROBOMETER_UV_SOURCE="${ROBOMETER_UV_SOURCE:-$(command -v uv || true)}"

UV_BIN="$ROBOMETER_SHARED_RUNTIME_ROOT/bin/uv"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/vla/users/niejunnan/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROBOMETER_SHARED_RUNTIME_ROOT/uv-python}"
export UV_PROJECT_ENVIRONMENT="$ROBOMETER_VENV"
export UV_MANAGED_PYTHON=1

mkdir -p \
  "$ROBOMETER_SHARED_RUNTIME_ROOT/bin" \
  "$UV_CACHE_DIR" \
  "$UV_PYTHON_INSTALL_DIR"

LOCK_DIR="$ROBOMETER_SHARED_RUNTIME_ROOT/.prepare.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another shared RoboMeter runtime build is active: $LOCK_DIR" >&2
  [[ -r "$LOCK_DIR/owner" ]] && cat "$LOCK_DIR/owner" >&2
  exit 1
fi
trap 'rm -f "$LOCK_DIR/owner"; rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
printf 'host=%s pid=%s started=%s\n' "$(hostname)" "$$" "$(date -Iseconds)" > "$LOCK_DIR/owner"

is_standalone_uv() {
  [[ -x "$1" ]] && [[ "$(od -An -t x1 -N4 "$1" | tr -d ' \n')" == "7f454c46" ]]
}

if ! is_standalone_uv "$UV_BIN"; then
  if ! is_standalone_uv "$ROBOMETER_UV_SOURCE"; then
    PYTHON_EXECUTABLE="$(command -v python3.10 || true)"
    if [[ -n "$PYTHON_EXECUTABLE" ]]; then
      ROBOMETER_UV_SOURCE="$(dirname "$(readlink -f "$PYTHON_EXECUTABLE")")/uv"
    fi
  fi
  if ! is_standalone_uv "$ROBOMETER_UV_SOURCE"; then
    echo "A standalone Linux uv binary is required; set ROBOMETER_UV_SOURCE explicitly" >&2
    exit 1
  fi
  install -m 0755 "$ROBOMETER_UV_SOURCE" "$UV_BIN"
fi

if [[ -x "$ROBOMETER_VENV/bin/python" && "${ROBOMETER_CONFIRM_CLUSTER_DRAINED:-0}" != "1" ]]; then
  echo "Refusing to update a shared environment without cluster-wide drain confirmation" >&2
  echo "Check 204, 225, 234, and 239, then rerun with ROBOMETER_CONFIRM_CLUSTER_DRAINED=1" >&2
  exit 1
fi

if pgrep -f "$ROBOMETER_VENV/bin/python" >/dev/null 2>&1; then
  echo "The shared RoboMeter environment is in use on this host; stop its jobs before syncing dependencies" >&2
  pgrep -af "$ROBOMETER_VENV/bin/python" >&2 || true
  exit 1
fi

"$UV_BIN" python install --no-bin "$ROBOMETER_PYTHON_VERSION"

MANAGED_PYTHON="$UV_PYTHON_INSTALL_DIR/cpython-$ROBOMETER_PYTHON_VERSION-linux-x86_64-gnu/bin/python3.10"
if [[ ! -x "$MANAGED_PYTHON" ]]; then
  echo "Managed Python was not installed at the expected path: $MANAGED_PYTHON" >&2
  exit 1
fi

CURRENT_PYTHON=""
if [[ -x "$ROBOMETER_VENV/bin/python" ]]; then
  CURRENT_PYTHON="$(readlink -f "$ROBOMETER_VENV/bin/python")"
fi
if [[ "$CURRENT_PYTHON" != "$(readlink -f "$MANAGED_PYTHON")" ]]; then
  "$UV_BIN" venv --clear "$ROBOMETER_VENV" --python "$MANAGED_PYTHON"
fi

cd "$ROBOMETER_ROOT"
UV_LINK_MODE=copy "$UV_BIN" sync \
  --frozen \
  --extra robometer \
  --python "$ROBOMETER_VENV/bin/python" \
  --no-progress

ROBOMETER_ALLOW_RUNTIME_MAINTENANCE=1 \
  "$(dirname "${BASH_SOURCE[0]}")/check_robometer_shared_runtime.sh"
