#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/vla/users/niejunnan/envs/serl_torch/bin/python}"
POLICY_PYTHON_BIN="${POLICY_PYTHON_BIN:-/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3}"
SERL_TORCH_ROOT="${SERL_TORCH_ROOT:-/vla/users/niejunnan/codebase/serl_torch}"
POLICY_ROOT="${POLICY_ROOT:-/vla/users/niejunnan/codebase/openpi-modified}"
POLICY_CONFIG="${POLICY_CONFIG:-pi0_libero_baseline_10_bs32_150000}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-/vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000}"
OPENPI_OFFICIAL_LAUNCHER="${OPENPI_OFFICIAL_LAUNCHER:-/vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi_10000_policy.sh}"
LIBERO_ENV_LAUNCHER="${LIBERO_ENV_LAUNCHER:-/vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh}"
LIBERO_CONDA_PREFIX="${LIBERO_CONDA_PREFIX:-/vla/users/niejunnan/envs/libero}"
ACTION_DIM="${ACTION_DIM:-7}"
POLICY_XLA_PREALLOCATE="${POLICY_XLA_PREALLOCATE:-false}"
POLICY_XLA_MEM_FRACTION="${POLICY_XLA_MEM_FRACTION:-0.35}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-900}"

CONFIG=""
SESSION=""
GPU=""
ENV_GPU=""
EVAL_ENV_GPU=""
POLICY_GPU=""
LEARNER_GPU=""
ACTOR_GPU=""
POLICY_MODE="direct"
ENV_PORT=""
EVAL_ENV_PORT=""
POLICY_PORT=""
TRAINER_PORT=""
BROADCAST_PORT=""

usage() {
  cat <<'USAGE'
Usage:
  bash examples/libero/pld/tools/launch_residual_sac_remote_isolated.sh \
    --config PATH --session NAME --gpu ID --policy-mode direct|official_ws \
    --env-port PORT --eval-env-port PORT --policy-port PORT \
    --trainer-port PORT --broadcast-port PORT

This launcher starts one complete remote-isolated PLD residual SAC experiment:
  actor env server + eval env server + frozen OpenPI policy server + learner + actor.
The YAML should already contain matching env.url, async_eval.env_url, policy.url,
runtime.async_eval.policy_url, trainer_port, and broadcast_port values.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --policy-mode) POLICY_MODE="$2"; shift 2 ;;
    --env-gpu) ENV_GPU="$2"; shift 2 ;;
    --eval-env-gpu) EVAL_ENV_GPU="$2"; shift 2 ;;
    --policy-gpu) POLICY_GPU="$2"; shift 2 ;;
    --learner-gpu) LEARNER_GPU="$2"; shift 2 ;;
    --actor-gpu) ACTOR_GPU="$2"; shift 2 ;;
    --env-port) ENV_PORT="$2"; shift 2 ;;
    --eval-env-port) EVAL_ENV_PORT="$2"; shift 2 ;;
    --policy-port) POLICY_PORT="$2"; shift 2 ;;
    --trainer-port) TRAINER_PORT="$2"; shift 2 ;;
    --broadcast-port) BROADCAST_PORT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

for item in CONFIG SESSION GPU ENV_PORT EVAL_ENV_PORT POLICY_PORT TRAINER_PORT BROADCAST_PORT; do
  if [[ -z "${!item}" ]]; then
    echo "ERROR: --${item,,} is required" >&2
    usage
    exit 2
  fi
done
if [[ -z "$ENV_GPU" ]]; then ENV_GPU="$GPU"; fi
if [[ -z "$EVAL_ENV_GPU" ]]; then EVAL_ENV_GPU="$ENV_GPU"; fi
if [[ -z "$POLICY_GPU" ]]; then POLICY_GPU="$GPU"; fi
if [[ -z "$LEARNER_GPU" ]]; then LEARNER_GPU="$GPU"; fi
if [[ -z "$ACTOR_GPU" ]]; then ACTOR_GPU="$GPU"; fi

if [[ "$POLICY_MODE" != "direct" && "$POLICY_MODE" != "official_ws" ]]; then
  echo "ERROR: --policy-mode must be direct or official_ws, got: $POLICY_MODE" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi

LOG_ROOT="$ROOT/examples/libero/pld/outputs/reward_model/launch_logs/${SESSION}"
mkdir -p "$LOG_ROOT"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n control "cd '$ROOT' && sleep infinity"

tmux new-window -t "$SESSION" -n env \
  "cd '$SERL_TORCH_ROOT' && LIBERO_CONDA_PREFIX='$LIBERO_CONDA_PREFIX' CUDA_VISIBLE_DEVICES='$ENV_GPU' MUJOCO_EGL_DEVICE_ID='$ENV_GPU' bash '$LIBERO_ENV_LAUNCHER' --host 127.0.0.1 --port '$ENV_PORT' --gpu-id '$ENV_GPU' 2>&1 | tee '$LOG_ROOT/env.log'"

tmux new-window -t "$SESSION" -n eval-env \
  "cd '$SERL_TORCH_ROOT' && LIBERO_CONDA_PREFIX='$LIBERO_CONDA_PREFIX' CUDA_VISIBLE_DEVICES='$EVAL_ENV_GPU' MUJOCO_EGL_DEVICE_ID='$EVAL_ENV_GPU' bash '$LIBERO_ENV_LAUNCHER' --host 127.0.0.1 --port '$EVAL_ENV_PORT' --gpu-id '$EVAL_ENV_GPU' 2>&1 | tee '$LOG_ROOT/eval_env.log'"

if [[ "$POLICY_MODE" == "direct" ]]; then
  tmux new-window -t "$SESSION" -n policy \
    "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$POLICY_GPU' XLA_PYTHON_CLIENT_PREALLOCATE='$POLICY_XLA_PREALLOCATE' XLA_PYTHON_CLIENT_MEM_FRACTION='$POLICY_XLA_MEM_FRACTION' '$POLICY_PYTHON_BIN' scripts/serve_reference_policy.py --policy openpi --policy-root '$POLICY_ROOT' --config-name '$POLICY_CONFIG' --checkpoint-path '$POLICY_CHECKPOINT' --action-dim '$ACTION_DIM' --device cuda --host 127.0.0.1 --port '$POLICY_PORT' 2>&1 | tee '$LOG_ROOT/policy.log'"
else
  tmux new-window -t "$SESSION" -n policy \
    "cd '$ROOT' && OPENPI_ROOT='$POLICY_ROOT' POLICY_CONFIG='$POLICY_CONFIG' POLICY_DIR='$POLICY_CHECKPOINT' OPENPI_CONDA_ENV='openpi-modified' XLA_PYTHON_CLIENT_PREALLOCATE='$POLICY_XLA_PREALLOCATE' bash '$OPENPI_OFFICIAL_LAUNCHER' --port '$POLICY_PORT' --gpu-id '$POLICY_GPU' 2>&1 | tee '$LOG_ROOT/policy.log'"
fi

WAIT_COMMON="cd '$ROOT' && '$PYTHON_BIN' examples/libero/rlt/tools/wait_for_tcp.py --host 127.0.0.1 --ports '$ENV_PORT' '$EVAL_ENV_PORT' '$POLICY_PORT' --timeout-sec '$WAIT_TIMEOUT_SEC'"

tmux new-window -t "$SESSION" -n learner \
  "$WAIT_COMMON && cd '$ROOT' && CUDA_VISIBLE_DEVICES='$LEARNER_GPU' MUJOCO_EGL_DEVICE_ID='$LEARNER_GPU' '$PYTHON_BIN' examples/libero/pld/scripts/train_residual_sac.py --role learner --config '$CONFIG' 2>&1 | tee '$LOG_ROOT/learner.log'"

tmux new-window -t "$SESSION" -n actor \
  "cd '$ROOT' && '$PYTHON_BIN' examples/libero/rlt/tools/wait_for_tcp.py --host 127.0.0.1 --ports '$ENV_PORT' '$EVAL_ENV_PORT' '$POLICY_PORT' '$TRAINER_PORT' --timeout-sec '$WAIT_TIMEOUT_SEC' && CUDA_VISIBLE_DEVICES='$ACTOR_GPU' MUJOCO_EGL_DEVICE_ID='$ACTOR_GPU' '$PYTHON_BIN' examples/libero/pld/scripts/train_residual_sac.py --role actor --config '$CONFIG' 2>&1 | tee '$LOG_ROOT/actor.log'"

echo "Started remote-isolated residual SAC session: $SESSION"
echo "  mode:    $POLICY_MODE"
echo "  gpu:     $GPU"
echo "  env_gpu: $ENV_GPU eval_env_gpu: $EVAL_ENV_GPU policy_gpu: $POLICY_GPU learner_gpu: $LEARNER_GPU actor_gpu: $ACTOR_GPU"
echo "  config:  $CONFIG"
echo "  env:     $ENV_PORT"
echo "  evalenv: $EVAL_ENV_PORT"
echo "  policy:  $POLICY_PORT"
echo "  trainer: $TRAINER_PORT"
echo "  logs:    $LOG_ROOT"
echo "Attach with: tmux attach -t $SESSION"
