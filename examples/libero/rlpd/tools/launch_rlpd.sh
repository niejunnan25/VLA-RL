#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/vla/users/niejunnan/envs/serl_torch/bin/python}"
CONFIG="$ROOT/examples/libero/rlpd/configs/libero_spatial_task4_rlpd.yaml"
SESSION="vlarl_libero_rlpd"
ACTOR_GPU="0"
LEARNER_GPU="0"
ENV_GPU=""
ENV_PORT="23100"
WITH_EVAL="0"
EVAL_GPU=""
EVAL_ENV_PORT="23110"
TRAINER_PORT="5588"
BROADCAST_PORT="5589"
RUN_DIR="$ROOT/outputs/libero_spatial_task4_rlpd"
SERL_TORCH_ROOT="/vla/users/niejunnan/codebase/serl_torch"
LIBERO_CONDA_PREFIX="/vla/users/niejunnan/envs/libero"
WITH_ENV_SERVER="1"
WAIT_TIMEOUT_SEC="600"

usage() {
  cat <<'EOF'
Usage: examples/libero/rlpd/tools/launch_rlpd.sh [options] [-- overrides...]

Starts a standard RLPD run: LIBERO env server, learner, actor, and optional async eval.
No OpenPI/reference-policy server is started because the SAC policy outputs actions directly.

Options:
  --config PATH                RLPD config path.
  --session NAME               tmux session name.
  --actor-gpu ID               GPU for actor.
  --learner-gpu ID             GPU for learner.
  --env-gpu ID                 GPU for actor LIBERO env server. Defaults to actor GPU.
  --env-port PORT              Actor LIBERO env server port.
  --with-eval                  Start async eval worker plus dedicated eval env server.
  --eval-gpu ID                GPU for async eval worker and eval env. Defaults to actor GPU.
  --eval-env-port PORT         Async eval LIBERO env server port.
  --trainer-port PORT          Agentlace trainer port.
  --broadcast-port PORT        Agentlace broadcast port.
  --run-dir PATH               Run directory.
  --python PATH                Python for actor/learner/eval worker.
  --serl-torch-root PATH       External serl_torch checkout for LIBERO env server.
  --libero-conda-prefix PATH   Conda prefix used by LIBERO env server.
  --no-env-server              Do not start LIBERO env servers; assumes ports are already served.
  --wait-timeout-sec SEC       Timeout for waiting on env/trainer ports.
EOF
}

OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --actor-gpu) ACTOR_GPU="$2"; shift 2 ;;
    --learner-gpu) LEARNER_GPU="$2"; shift 2 ;;
    --env-gpu) ENV_GPU="$2"; shift 2 ;;
    --env-port) ENV_PORT="$2"; shift 2 ;;
    --with-eval) WITH_EVAL="1"; shift ;;
    --eval-gpu) EVAL_GPU="$2"; shift 2 ;;
    --eval-env-port) EVAL_ENV_PORT="$2"; shift 2 ;;
    --trainer-port) TRAINER_PORT="$2"; shift 2 ;;
    --broadcast-port) BROADCAST_PORT="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --serl-torch-root) SERL_TORCH_ROOT="$2"; shift 2 ;;
    --libero-conda-prefix) LIBERO_CONDA_PREFIX="$2"; shift 2 ;;
    --no-env-server) WITH_ENV_SERVER="0"; shift ;;
    --wait-timeout-sec) WAIT_TIMEOUT_SEC="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; OVERRIDES=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$ENV_GPU" ]]; then ENV_GPU="$ACTOR_GPU"; fi
if [[ -z "$EVAL_GPU" ]]; then EVAL_GPU="$ACTOR_GPU"; fi

COMMON_OVERRIDES=(
  "env.url=http://127.0.0.1:${ENV_PORT}"
  "runtime.trainer_port=${TRAINER_PORT}"
  "runtime.broadcast_port=${BROADCAST_PORT}"
  "runtime.run_dir=${RUN_DIR}"
  "${OVERRIDES[@]}"
)

if [[ "$WITH_EVAL" == "1" ]]; then
  COMMON_OVERRIDES+=(
    "runtime.async_eval.enabled=true"
    "runtime.async_eval.env_url=http://127.0.0.1:${EVAL_ENV_PORT}"
    "runtime.async_eval.worker_cuda_visible_devices=${EVAL_GPU}"
    "runtime.async_eval.worker_mujoco_egl_device_id=${EVAL_GPU}"
  )
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

if [[ "$WITH_ENV_SERVER" == "1" ]]; then
  tmux new-session -d -s "$SESSION" -n env \
    "cd '$SERL_TORCH_ROOT' && LIBERO_CONDA_PREFIX='$LIBERO_CONDA_PREFIX' bash examples/libero/tools/serve_env.sh --host 127.0.0.1 --port '$ENV_PORT' --gpu-id '$ENV_GPU'"
  if [[ "$WITH_EVAL" == "1" ]]; then
    tmux new-window -t "$SESSION" -n eval-env \
      "cd '$SERL_TORCH_ROOT' && LIBERO_CONDA_PREFIX='$LIBERO_CONDA_PREFIX' bash examples/libero/tools/serve_env.sh --host 127.0.0.1 --port '$EVAL_ENV_PORT' --gpu-id '$EVAL_GPU'"
  fi
else
  tmux new-session -d -s "$SESSION" -n control "cd '$ROOT' && sleep infinity"
fi

LEARNER_WAIT=""
if [[ "$WITH_EVAL" == "1" ]]; then
  LEARNER_WAIT="'$PYTHON_BIN' examples/libero/rlt/tools/wait_for_tcp.py --host 127.0.0.1 --ports '$EVAL_ENV_PORT' --timeout-sec '$WAIT_TIMEOUT_SEC' && "
fi

tmux new-window -t "$SESSION" -n learner \
  "cd '$ROOT' && ${LEARNER_WAIT}CUDA_VISIBLE_DEVICES='$LEARNER_GPU' MUJOCO_EGL_DEVICE_ID='$LEARNER_GPU' '$PYTHON_BIN' examples/libero/rlpd/scripts/train_rlpd.py --role learner --config '$CONFIG' -- ${COMMON_OVERRIDES[*]}"

tmux new-window -t "$SESSION" -n actor \
  "cd '$ROOT' && '$PYTHON_BIN' examples/libero/rlt/tools/wait_for_tcp.py --host 127.0.0.1 --ports '$ENV_PORT' '$TRAINER_PORT' --timeout-sec '$WAIT_TIMEOUT_SEC' && CUDA_VISIBLE_DEVICES='$ACTOR_GPU' MUJOCO_EGL_DEVICE_ID='$ACTOR_GPU' '$PYTHON_BIN' examples/libero/rlpd/scripts/train_rlpd.py --role actor --config '$CONFIG' -- ${COMMON_OVERRIDES[*]}"

echo "Started RLPD tmux session: $SESSION"
echo "  env:     GPU $ENV_GPU, port $ENV_PORT"
if [[ "$WITH_EVAL" == "1" ]]; then
  echo "  eval:    GPU $EVAL_GPU, env port $EVAL_ENV_PORT"
fi
echo "  learner: GPU $LEARNER_GPU, trainer=$TRAINER_PORT broadcast=$BROADCAST_PORT"
echo "  actor:   GPU $ACTOR_GPU"
echo "  run_dir: $RUN_DIR"
echo "Attach with: tmux attach -t $SESSION"
