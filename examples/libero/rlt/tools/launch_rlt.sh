#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
POLICY_PYTHON_BIN="${POLICY_PYTHON_BIN:-/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3}"
CONFIG="$ROOT/examples/libero/rlt/configs/libero_spatial_task4_openpi_rlt.yaml"
SESSION="vlarl_libero_rlt"
ACTOR_GPU="0"
LEARNER_GPU="1"
ENV_GPU=""
POLICY_GPU=""
ENV_PORT="23000"
POLICY_PORT="8899"
WITH_EVAL="0"
EVAL_GPU=""
EVAL_ENV_PORT="24000"
EVAL_POLICY_PORT="8999"
TRAINER_PORT="5488"
BROADCAST_PORT="5489"
RUN_DIR="$ROOT/outputs/libero_spatial_task4_openpi_rlt"
SERL_TORCH_ROOT="/vla/users/niejunnan/codebase/serl_torch"
LIBERO_CONDA_PREFIX="/vla/users/niejunnan/envs/libero"
POLICY_ROOT="/vla/users/niejunnan/codebase/openpi-rlt-github"
POLICY_CONFIG="pi0_libero"
POLICY_CHECKPOINT="/vla/users/niejunnan/assets/openpi-assets/checkpoints/pi0_libero_pytorch"
ACTION_DIM="7"
WITH_ENV_SERVER="1"
WITH_POLICY_SERVER="1"
WAIT_TIMEOUT_SEC="600"

usage() {
  cat <<'EOF'
Usage: examples/libero/rlt/tools/launch_rlt.sh [options] [-- overrides...]

Options:
  --config PATH                RLT config path.
  --session NAME               tmux session name.
  --actor-gpu ID               GPU for actor.
  --learner-gpu ID             GPU for learner.
  --env-gpu ID                 GPU for LIBERO env server. Defaults to actor GPU.
  --policy-gpu ID              GPU for reference-policy server. Defaults to actor GPU.
  --env-port PORT              LIBERO env server port.
  --policy-port PORT           Reference-policy server port.
  --with-eval                  Start dedicated async-eval env/policy services.
  --eval-gpu ID                GPU for async eval env and policy. Defaults to actor GPU.
  --eval-env-port PORT         Async eval LIBERO env server port.
  --eval-policy-port PORT      Async eval reference-policy server port.
  --trainer-port PORT          Agentlace trainer port.
  --broadcast-port PORT        Agentlace broadcast port.
  --run-dir PATH               Run directory.
  --python PATH                Python for actor/learner.
  --policy-python PATH         Python for reference-policy server.
  --policy-root PATH           OpenPI/other policy checkout root.
  --policy-config NAME         Reference policy config name.
  --policy-checkpoint PATH     Reference policy checkpoint.
  --serl-torch-root PATH       External serl_torch checkout for LIBERO env server.
  --libero-conda-prefix PATH   Conda prefix used by LIBERO env server.
  --no-env-server              Do not start LIBERO env server.
  --no-policy-server           Do not start reference-policy server.
  --wait-timeout-sec SEC        Timeout for actor waiting on env/policy/trainer ports.
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
    --policy-gpu) POLICY_GPU="$2"; shift 2 ;;
    --env-port) ENV_PORT="$2"; shift 2 ;;
    --policy-port) POLICY_PORT="$2"; shift 2 ;;
    --with-eval) WITH_EVAL="1"; shift ;;
    --eval-gpu) EVAL_GPU="$2"; shift 2 ;;
    --eval-env-port) EVAL_ENV_PORT="$2"; shift 2 ;;
    --eval-policy-port) EVAL_POLICY_PORT="$2"; shift 2 ;;
    --trainer-port) TRAINER_PORT="$2"; shift 2 ;;
    --broadcast-port) BROADCAST_PORT="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --policy-python) POLICY_PYTHON_BIN="$2"; shift 2 ;;
    --policy-root) POLICY_ROOT="$2"; shift 2 ;;
    --policy-config) POLICY_CONFIG="$2"; shift 2 ;;
    --policy-checkpoint) POLICY_CHECKPOINT="$2"; shift 2 ;;
    --serl-torch-root) SERL_TORCH_ROOT="$2"; shift 2 ;;
    --libero-conda-prefix) LIBERO_CONDA_PREFIX="$2"; shift 2 ;;
    --no-env-server) WITH_ENV_SERVER="0"; shift ;;
    --no-policy-server) WITH_POLICY_SERVER="0"; shift ;;
    --wait-timeout-sec) WAIT_TIMEOUT_SEC="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; OVERRIDES=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$ENV_GPU" ]]; then ENV_GPU="$ACTOR_GPU"; fi
if [[ -z "$POLICY_GPU" ]]; then POLICY_GPU="$ACTOR_GPU"; fi
if [[ -z "$EVAL_GPU" ]]; then EVAL_GPU="$ACTOR_GPU"; fi

COMMON_OVERRIDES=(
  "env.url=http://127.0.0.1:${ENV_PORT}"
  "policy.url=http://127.0.0.1:${POLICY_PORT}"
  "runtime.trainer_port=${TRAINER_PORT}"
  "runtime.broadcast_port=${BROADCAST_PORT}"
  "runtime.run_dir=${RUN_DIR}"
  "${OVERRIDES[@]}"
)

if [[ "$WITH_EVAL" == "1" ]]; then
  COMMON_OVERRIDES+=(
    "runtime.async_eval.enabled=true"
    "runtime.async_eval.env_url=http://127.0.0.1:${EVAL_ENV_PORT}"
    "runtime.async_eval.policy_url=http://127.0.0.1:${EVAL_POLICY_PORT}"
  )
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

if [[ "$WITH_ENV_SERVER" == "1" ]]; then
  tmux new-session -d -s "$SESSION" -n env     "cd '$SERL_TORCH_ROOT' && LIBERO_CONDA_PREFIX='$LIBERO_CONDA_PREFIX' bash examples/libero/tools/serve_env.sh --host 127.0.0.1 --port '$ENV_PORT' --gpu-id '$ENV_GPU'"
else
  tmux new-session -d -s "$SESSION" -n control "cd '$ROOT' && sleep infinity"
fi

if [[ "$WITH_POLICY_SERVER" == "1" ]]; then
  tmux new-window -t "$SESSION" -n policy     "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$POLICY_GPU' '$POLICY_PYTHON_BIN' scripts/serve_reference_policy.py --policy openpi --policy-root '$POLICY_ROOT' --config-name '$POLICY_CONFIG' --checkpoint-path '$POLICY_CHECKPOINT' --action-dim '$ACTION_DIM' --device cuda --host 127.0.0.1 --port '$POLICY_PORT'"
fi

if [[ "$WITH_EVAL" == "1" ]]; then
  tmux new-window -t "$SESSION" -n eval-env     "cd '$SERL_TORCH_ROOT' && LIBERO_CONDA_PREFIX='$LIBERO_CONDA_PREFIX' bash examples/libero/tools/serve_env.sh --host 127.0.0.1 --port '$EVAL_ENV_PORT' --gpu-id '$EVAL_GPU'"
  tmux new-window -t "$SESSION" -n eval-policy     "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$EVAL_GPU' '$POLICY_PYTHON_BIN' scripts/serve_reference_policy.py --policy openpi --policy-root '$POLICY_ROOT' --config-name '$POLICY_CONFIG' --checkpoint-path '$POLICY_CHECKPOINT' --action-dim '$ACTION_DIM' --device cuda --host 127.0.0.1 --port '$EVAL_POLICY_PORT'"
fi

tmux new-window -t "$SESSION" -n learner   "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$LEARNER_GPU' '$PYTHON_BIN' examples/libero/rlt/scripts/train_stage2.py --role learner --config '$CONFIG' -- ${COMMON_OVERRIDES[*]}"

WAIT_PORTS=("$ENV_PORT" "$POLICY_PORT" "$TRAINER_PORT")
if [[ "$WITH_EVAL" == "1" ]]; then
  WAIT_PORTS+=("$EVAL_ENV_PORT" "$EVAL_POLICY_PORT")
fi
WAIT_PORT_ARGS="${WAIT_PORTS[*]}"

tmux new-window -t "$SESSION" -n actor   "cd '$ROOT' && '$PYTHON_BIN' examples/libero/rlt/tools/wait_for_tcp.py --host 127.0.0.1 --ports $WAIT_PORT_ARGS --timeout-sec '$WAIT_TIMEOUT_SEC' && CUDA_VISIBLE_DEVICES='$ACTOR_GPU' '$PYTHON_BIN' examples/libero/rlt/scripts/train_stage2.py --role actor --config '$CONFIG' -- ${COMMON_OVERRIDES[*]}"

echo "Started tmux session: $SESSION"
echo "  env:     GPU $ENV_GPU, port $ENV_PORT"
echo "  policy:  GPU $POLICY_GPU, port $POLICY_PORT"
if [[ "$WITH_EVAL" == "1" ]]; then
  echo "  eval:    GPU $EVAL_GPU, env=$EVAL_ENV_PORT policy=$EVAL_POLICY_PORT"
fi
echo "  learner: GPU $LEARNER_GPU, trainer=$TRAINER_PORT broadcast=$BROADCAST_PORT"
echo "  actor:   GPU $ACTOR_GPU"
echo "  run_dir: $RUN_DIR"
echo "Attach with: tmux attach -t $SESSION"
