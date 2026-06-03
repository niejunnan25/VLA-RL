#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACTOR_PYTHON_BIN="${ACTOR_PYTHON_BIN:-/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3}"
CONFIG="$ROOT/recipes/libero_spatial_task4_openpi_rlt_agentlace.yaml"
SESSION="vlarl_rlt_agentlace"
ACTOR_GPU="0"
LEARNER_GPU="1"
ENV_GPU=""
ENV_PORT="23000"
TRAINER_PORT="5488"
BROADCAST_PORT="5489"
RUN_DIR="$ROOT/outputs/libero_spatial_task4_openpi_rlt_agentlace"
SERL_TORCH_ROOT="/vla/users/niejunnan/codebase/serl_torch-rlt-merge"
LIBERO_CONDA_PREFIX="/vla/users/niejunnan/envs/libero"
WITH_ENV_SERVER="1"

usage() {
  cat <<'EOF'
Usage: scripts/launch_agentlace_rlt.sh [options] [-- hydra.overrides=...]

Options:
  --config PATH              Recipe path.
  --session NAME             tmux session name.
  --actor-gpu ID             GPU for actor/OpenPI.
  --learner-gpu ID           GPU for learner.
  --env-gpu ID               GPU for LIBERO env server. Defaults to actor GPU.
  --env-port PORT            LIBERO env server port.
  --trainer-port PORT        Agentlace trainer port.
  --broadcast-port PORT      Agentlace broadcast port.
  --run-dir PATH             Run directory.
  --python PATH              Python executable.
  --actor-python PATH        Python executable for actor/OpenPI.
  --serl-torch-root PATH     External serl_torch checkout for serve_libero_env.py.
  --libero-conda-prefix PATH  Conda prefix used by the external LIBERO env server.
  --no-env-server            Do not start the LIBERO env server window.
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
    --trainer-port) TRAINER_PORT="$2"; shift 2 ;;
    --broadcast-port) BROADCAST_PORT="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --actor-python) ACTOR_PYTHON_BIN="$2"; shift 2 ;;
    --serl-torch-root) SERL_TORCH_ROOT="$2"; shift 2 ;;
    --libero-conda-prefix) LIBERO_CONDA_PREFIX="$2"; shift 2 ;;
    --no-env-server) WITH_ENV_SERVER="0"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; OVERRIDES=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$ENV_GPU" ]]; then
  ENV_GPU="$ACTOR_GPU"
fi

COMMON_OVERRIDES=(
  "env.url=http://127.0.0.1:${ENV_PORT}"
  "runtime.trainer_port=${TRAINER_PORT}"
  "runtime.broadcast_port=${BROADCAST_PORT}"
  "runtime.run_dir=${RUN_DIR}"
  "${OVERRIDES[@]}"
)

tmux kill-session -t "$SESSION" 2>/dev/null || true

if [[ "$WITH_ENV_SERVER" == "1" ]]; then
  tmux new-session -d -s "$SESSION" -n env \
    "cd '$SERL_TORCH_ROOT' && LIBERO_CONDA_PREFIX='$LIBERO_CONDA_PREFIX' bash examples/libero/tools/serve_env.sh --host 127.0.0.1 --port '$ENV_PORT' --gpu-id '$ENV_GPU'"
else
  tmux new-session -d -s "$SESSION" -n control "cd '$ROOT' && sleep infinity"
fi

tmux new-window -t "$SESSION" -n learner \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$LEARNER_GPU' '$PYTHON_BIN' scripts/train_agentlace.py --role learner --config '$CONFIG' -- ${COMMON_OVERRIDES[*]}"

tmux new-window -t "$SESSION" -n actor \
  "sleep 10; cd '$ROOT' && CUDA_VISIBLE_DEVICES='$ACTOR_GPU' '$ACTOR_PYTHON_BIN' scripts/train_agentlace.py --role actor --config '$CONFIG' -- ${COMMON_OVERRIDES[*]}"

echo "Started tmux session: $SESSION"
echo "  env:     GPU $ENV_GPU, port $ENV_PORT"
echo "  learner: GPU $LEARNER_GPU, trainer=$TRAINER_PORT broadcast=$BROADCAST_PORT"
echo "  actor:   GPU $ACTOR_GPU"
echo "  actor python: $ACTOR_PYTHON_BIN"
echo "  run_dir: $RUN_DIR"
echo "Attach with: tmux attach -t $SESSION"
