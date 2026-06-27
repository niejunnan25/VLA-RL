#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/vla/users/niejunnan/envs/serl_torch/bin/python}"
CONFIG="$ROOT/examples/libero/rlpd/configs/libero_spatial_task4_rlpd.yaml"
SESSION="vlarl_libero_rlpd"
ACTOR_GPU="0"
LEARNER_GPU="0"
EVAL_GPU=""
TRAINER_PORT="5588"
BROADCAST_PORT="5589"
RUN_DIR="$ROOT/outputs/libero_spatial_task4_rlpd"
WITH_EVAL="1"
WAIT_TIMEOUT_SEC="600"

usage() {
  cat <<'EOF'
Usage: examples/libero/rlpd/tools/launch_rlpd.sh [options] [-- overrides...]

Starts a standard RLPD run with local in-process LIBERO envs, learner, actor,
and optional async eval. No OpenPI/reference-policy server is started because the
SAC policy outputs actions directly.

Options:
  --config PATH                RLPD config path.
  --session NAME               tmux session name.
  --actor-gpu ID               GPU for actor and actor-local LIBERO env.
  --learner-gpu ID             GPU for learner.
  --with-eval                  Start async eval worker with its own local LIBERO env. Enabled by default.
  --no-eval                    Disable async eval for debugging.
  --eval-gpu ID                GPU for async eval worker and eval-local LIBERO env. Defaults to actor GPU.
  --trainer-port PORT          Agentlace trainer port.
  --broadcast-port PORT        Agentlace broadcast port.
  --run-dir PATH               Run directory.
  --python PATH                Python for actor/learner/eval worker.
  --wait-timeout-sec SEC       Timeout for actor waiting on trainer port.

Legacy options accepted but ignored: --env-gpu, --env-port, --eval-env-port,
--serl-torch-root, --libero-conda-prefix, --no-env-server.
EOF
}

OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --actor-gpu) ACTOR_GPU="$2"; shift 2 ;;
    --learner-gpu) LEARNER_GPU="$2"; shift 2 ;;
    --env-gpu) shift 2 ;;
    --env-port) shift 2 ;;
    --with-eval) WITH_EVAL="1"; shift ;;
    --no-eval) WITH_EVAL="0"; shift ;;
    --eval-gpu) EVAL_GPU="$2"; shift 2 ;;
    --eval-env-port) shift 2 ;;
    --trainer-port) TRAINER_PORT="$2"; shift 2 ;;
    --broadcast-port) BROADCAST_PORT="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --serl-torch-root) shift 2 ;;
    --libero-conda-prefix) shift 2 ;;
    --with-env-server) echo "ERROR: launch_rlpd.sh now uses the local LIBERO backend; remove --with-env-server." >&2; exit 2 ;;
    --no-env-server) shift ;;
    --wait-timeout-sec) WAIT_TIMEOUT_SEC="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; OVERRIDES=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$EVAL_GPU" ]]; then EVAL_GPU="$ACTOR_GPU"; fi

COMMON_OVERRIDES=(
  "runtime.trainer_port=${TRAINER_PORT}"
  "runtime.broadcast_port=${BROADCAST_PORT}"
  "runtime.run_dir=${RUN_DIR}"
  "${OVERRIDES[@]}"
)

if [[ "$WITH_EVAL" == "1" ]]; then
  COMMON_OVERRIDES+=(
    "runtime.async_eval.enabled=true"
    "runtime.async_eval.worker_cuda_visible_devices=${EVAL_GPU}"
    "runtime.async_eval.worker_mujoco_egl_device_id=${EVAL_GPU}"
  )
else
  COMMON_OVERRIDES+=(
    "runtime.async_eval.enabled=false"
  )
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n control "cd '$ROOT' && sleep infinity"

tmux new-window -t "$SESSION" -n learner \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$LEARNER_GPU' MUJOCO_EGL_DEVICE_ID='$LEARNER_GPU' '$PYTHON_BIN' examples/libero/rlpd/scripts/train_rlpd.py --role learner --config '$CONFIG' -- ${COMMON_OVERRIDES[*]}"

tmux new-window -t "$SESSION" -n actor \
  "cd '$ROOT' && '$PYTHON_BIN' examples/libero/rlt/tools/wait_for_tcp.py --host 127.0.0.1 --ports '$TRAINER_PORT' --timeout-sec '$WAIT_TIMEOUT_SEC' && CUDA_VISIBLE_DEVICES='$ACTOR_GPU' MUJOCO_EGL_DEVICE_ID='$ACTOR_GPU' '$PYTHON_BIN' examples/libero/rlpd/scripts/train_rlpd.py --role actor --config '$CONFIG' -- ${COMMON_OVERRIDES[*]}"

echo "Started RLPD tmux session: $SESSION"
echo "  env:     local in actor process, GPU $ACTOR_GPU"
if [[ "$WITH_EVAL" == "1" ]]; then
  echo "  eval:    local env in worker, GPU $EVAL_GPU"
fi
echo "  learner: GPU $LEARNER_GPU, trainer=$TRAINER_PORT broadcast=$BROADCAST_PORT"
echo "  actor:   GPU $ACTOR_GPU"
echo "  run_dir: $RUN_DIR"
echo "Attach with: tmux attach -t $SESSION"
