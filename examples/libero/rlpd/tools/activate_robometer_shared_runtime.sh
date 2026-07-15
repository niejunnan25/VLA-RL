#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source ${BASH_SOURCE[0]}" >&2
  exit 2
fi

_ROBOMETER_ACTIVATE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROBOMETER_ACTIVATE_ROOT="${ROBOMETER_ROOT:-/vla/users/niejunnan/workspace/robometer}"
_ROBOMETER_ACTIVATE_VENV="${ROBOMETER_VENV:-$_ROBOMETER_ACTIVATE_ROOT/.venv-shared}"

if [[ "${ROBOMETER_SHARED_RUNTIME_ACTIVE:-0}" != "1" || "${VIRTUAL_ENV:-}" != "$_ROBOMETER_ACTIVATE_VENV" ]]; then
  declare -gA _ROBOMETER_OLD_ENV_VALUES=()
  declare -gA _ROBOMETER_OLD_ENV_SET=()
  for _robometer_var in \
    LD_LIBRARY_PATH \
    UV_CACHE_DIR \
    UV_PYTHON_INSTALL_DIR \
    UV_PROJECT_ENVIRONMENT \
    UV_MANAGED_PYTHON \
    ROBOMETER_ROOT \
    ROBOMETER_SHARED_RUNTIME_ROOT \
    ROBOMETER_VENV \
    ROBOMETER_PYTHON \
    ROBOMETER_PYTHON_REALPATH \
    ROBOMETER_SITE_PACKAGES \
    ROBOMETER_SHARED_LIBRARY_PATH \
    ROBOMETER_SHARED_RUNTIME_ACTIVE
  do
    if [[ -n "${!_robometer_var+x}" ]]; then
      _ROBOMETER_OLD_ENV_SET["$_robometer_var"]=1
      _ROBOMETER_OLD_ENV_VALUES["$_robometer_var"]="${!_robometer_var}"
    else
      _ROBOMETER_OLD_ENV_SET["$_robometer_var"]=0
    fi
  done

  if [[ ! -r "$_ROBOMETER_ACTIVATE_VENV/bin/activate" ]]; then
    echo "Shared RoboMeter activation script is unavailable: $_ROBOMETER_ACTIVATE_VENV/bin/activate" >&2
    unset _ROBOMETER_OLD_ENV_VALUES _ROBOMETER_OLD_ENV_SET
    unset _ROBOMETER_ACTIVATE_SCRIPT_DIR _ROBOMETER_ACTIVATE_ROOT _ROBOMETER_ACTIVATE_VENV _robometer_var
    return 1
  fi

  if ! source "$_ROBOMETER_ACTIVATE_VENV/bin/activate"; then
    echo "Failed to activate $_ROBOMETER_ACTIVATE_VENV" >&2
    unset _ROBOMETER_OLD_ENV_VALUES _ROBOMETER_OLD_ENV_SET
    unset _ROBOMETER_ACTIVATE_SCRIPT_DIR _ROBOMETER_ACTIVATE_ROOT _ROBOMETER_ACTIVATE_VENV _robometer_var
    return 1
  fi
  eval "$(declare -f deactivate | sed '1s/^deactivate /_robometer_base_deactivate /')"

  deactivate() {
    local restore_var
    _robometer_base_deactivate "$@"
    for restore_var in "${!_ROBOMETER_OLD_ENV_SET[@]}"; do
      if [[ "${_ROBOMETER_OLD_ENV_SET[$restore_var]}" == "1" ]]; then
        printf -v "$restore_var" '%s' "${_ROBOMETER_OLD_ENV_VALUES[$restore_var]}"
        export "$restore_var"
      else
        unset "$restore_var"
      fi
    done
    unset _ROBOMETER_OLD_ENV_VALUES _ROBOMETER_OLD_ENV_SET
    unset -f _robometer_base_deactivate deactivate
  }
fi

export ROBOMETER_ROOT="$_ROBOMETER_ACTIVATE_ROOT"
export ROBOMETER_VENV="$_ROBOMETER_ACTIVATE_VENV"
if ! source "$_ROBOMETER_ACTIVATE_SCRIPT_DIR/robometer_shared_runtime_env.sh"; then
  echo "Failed to configure the RoboMeter shared runtime" >&2
  if declare -F deactivate >/dev/null 2>&1; then
    deactivate
  fi
  unset _ROBOMETER_ACTIVATE_SCRIPT_DIR _ROBOMETER_ACTIVATE_ROOT _ROBOMETER_ACTIVATE_VENV _robometer_var
  return 1
fi
export ROBOMETER_SHARED_RUNTIME_ACTIVE=1
hash -r 2>/dev/null || true

if [[ "${ROBOMETER_ACTIVATE_CHECK:-1}" == "1" ]]; then
  if ! bash "$_ROBOMETER_ACTIVATE_SCRIPT_DIR/check_robometer_shared_runtime.sh"; then
    echo "RoboMeter shared runtime validation failed" >&2
    if declare -F deactivate >/dev/null 2>&1; then
      deactivate
    fi
    unset _ROBOMETER_ACTIVATE_SCRIPT_DIR _ROBOMETER_ACTIVATE_ROOT _ROBOMETER_ACTIVATE_VENV _robometer_var
    return 1
  fi
fi

echo "RoboMeter shared runtime activated: $ROBOMETER_VENV"
unset _ROBOMETER_ACTIVATE_SCRIPT_DIR _ROBOMETER_ACTIVATE_ROOT _ROBOMETER_ACTIVATE_VENV _robometer_var
