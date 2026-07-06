#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_ARGS=("$@")

format_cmd() {
    local formatted=""
    local token=""
    local escaped=""
    for token in "$@"; do
        printf -v escaped "%q" "$token"
        if [[ -n "$formatted" ]]; then
            formatted+=" "
        fi
        formatted+="$escaped"
    done
    printf "%s" "$formatted"
}

DEFAULT_OPENPI_ROOT="/vla/users/niejunnan/codebase/openpi"
DEFAULT_POLICY_CONFIG="pi05_libero_one_shot_spatial_traj1_rot180"
DEFAULT_POLICY_DIR="/vla/users/niejunnan/assets/openpi-assets/pi05_libero_one_shot_spatial_traj1_rot180/49999"
DEFAULT_OPENPI_CONDA_ENV="openpi-modified"

export OPENPI_ROOT="${OPENPI_ROOT:-$DEFAULT_OPENPI_ROOT}"
export OPENPI_CONDA_ENV="${OPENPI_CONDA_ENV:-$DEFAULT_OPENPI_CONDA_ENV}"
export POLICY_CONFIG="${POLICY_CONFIG:-$DEFAULT_POLICY_CONFIG}"
export POLICY_DIR="${POLICY_DIR:-$DEFAULT_POLICY_DIR}"

if [[ ! -d "$OPENPI_ROOT" ]]; then
    echo "ERROR: OpenPI root not found: $OPENPI_ROOT"
    exit 1
fi

if [[ ! -d "$POLICY_DIR" ]]; then
    echo "ERROR: policy checkpoint directory not found: $POLICY_DIR"
    exit 1
fi

LAUNCH_CMD_RECORD="$(format_cmd bash "$TOOLS_DIR/serve_openpi_one_shot_spatial_49999_policy.sh" "${ORIGINAL_ARGS[@]}")"

echo "=========================================="
echo "  LIBERO OpenPI One-Shot Spatial 49999 Policy Server"
echo "=========================================="
echo "  Launch cmd  : $LAUNCH_CMD_RECORD"
echo "  OpenPI root : $OPENPI_ROOT"
echo "  OpenPI env  : $OPENPI_CONDA_ENV"
echo "  Config      : $POLICY_CONFIG"
echo "  Policy dir  : $POLICY_DIR"
echo "=========================================="

exec bash "$TOOLS_DIR/serve_openpi_policy.sh" "$@"
