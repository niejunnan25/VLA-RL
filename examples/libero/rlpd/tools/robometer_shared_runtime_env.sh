#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This file configures the current shell and must be sourced" >&2
  exit 2
fi

_robometer_configure_shared_runtime() {
  local inherited_library_path
  local library_dir
  local library_path
  local python_realpath
  local site_packages
  local -a library_dirs

  export ROBOMETER_ROOT="${ROBOMETER_ROOT:-/vla/users/niejunnan/workspace/robometer}"
  export ROBOMETER_SHARED_RUNTIME_ROOT="${ROBOMETER_SHARED_RUNTIME_ROOT:-/vla/users/niejunnan/runtime/robometer}"
  export ROBOMETER_VENV="${ROBOMETER_VENV:-$ROBOMETER_ROOT/.venv-shared}"
  export ROBOMETER_PYTHON="${ROBOMETER_PYTHON_CMD:-$ROBOMETER_VENV/bin/python}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/vla/users/niejunnan/.cache/uv}"
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROBOMETER_SHARED_RUNTIME_ROOT/uv-python}"
  export UV_PROJECT_ENVIRONMENT="$ROBOMETER_VENV"
  export UV_MANAGED_PYTHON=1

  if [[ -d "$ROBOMETER_SHARED_RUNTIME_ROOT/.prepare.lock" && "${ROBOMETER_ALLOW_RUNTIME_MAINTENANCE:-0}" != "1" ]]; then
    echo "RoboMeter shared runtime maintenance is active: $ROBOMETER_SHARED_RUNTIME_ROOT/.prepare.lock" >&2
    if [[ -r "$ROBOMETER_SHARED_RUNTIME_ROOT/.prepare.lock/owner" ]]; then
      cat "$ROBOMETER_SHARED_RUNTIME_ROOT/.prepare.lock/owner" >&2
    fi
    return 1
  fi

  if [[ "$ROBOMETER_PYTHON" == *[[:space:]]* || ! -x "$ROBOMETER_PYTHON" ]]; then
    echo "RoboMeter Python must be one executable path: $ROBOMETER_PYTHON" >&2
    return 1
  fi

  python_realpath="$(readlink -f "$ROBOMETER_PYTHON")"
  if [[ "${ROBOMETER_ALLOW_EXTERNAL_VENV:-0}" != "1" && "$python_realpath" != "$UV_PYTHON_INSTALL_DIR"/* ]]; then
    echo "RoboMeter Python resolves outside the shared uv install: $python_realpath" >&2
    return 1
  fi

  if ! site_packages="$("$ROBOMETER_PYTHON" - <<'PY_SITE'
import site

paths = site.getsitepackages()
if not paths:
    raise SystemExit("RoboMeter Python has no site-packages directory")
print(paths[0])
PY_SITE
)"; then
    echo "Unable to resolve RoboMeter site-packages from $ROBOMETER_PYTHON" >&2
    return 1
  fi

  library_dirs=(
    "$site_packages/nvidia/cudnn/lib"
    "$site_packages/torch/lib"
  )
  for library_dir in "${library_dirs[@]}"; do
    if [[ ! -d "$library_dir" ]]; then
      echo "Required RoboMeter library directory is missing: $library_dir" >&2
      return 1
    fi
  done
  for library_dir in "$site_packages"/nvidia/*/lib; do
    [[ -d "$library_dir" ]] || continue
    [[ "$library_dir" == "${library_dirs[0]}" ]] && continue
    library_dirs+=("$library_dir")
  done
  library_path="$(IFS=:; printf '%s' "${library_dirs[*]}")"

  inherited_library_path="${LD_LIBRARY_PATH:-}"
  if [[ -n "${ROBOMETER_SHARED_LIBRARY_PATH:-}" ]]; then
    if [[ "$inherited_library_path" == "$ROBOMETER_SHARED_LIBRARY_PATH" ]]; then
      inherited_library_path=""
    elif [[ "$inherited_library_path" == "$ROBOMETER_SHARED_LIBRARY_PATH":* ]]; then
      inherited_library_path="${inherited_library_path#"$ROBOMETER_SHARED_LIBRARY_PATH":}"
    fi
  fi

  export ROBOMETER_PYTHON_REALPATH="$python_realpath"
  export ROBOMETER_SITE_PACKAGES="$site_packages"
  export ROBOMETER_SHARED_LIBRARY_PATH="$library_path"
  export LD_LIBRARY_PATH="$library_path${inherited_library_path:+:$inherited_library_path}"
}

if ! _robometer_configure_shared_runtime; then
  unset -f _robometer_configure_shared_runtime
  return 1
fi
unset -f _robometer_configure_shared_runtime
