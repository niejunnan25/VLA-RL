#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/vla/users/niejunnan/envs/serl_torch/bin/python}"
POLICY_PYTHON_BIN="${POLICY_PYTHON_BIN:-/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3}"
CONFIG="$ROOT/examples/libero/pld/configs/libero_spatial_task4_openpi_pld.yaml"
SESSION="vlarl_libero_pld"
ACTOR_GPU="0"
LEARNER_GPU="1"
POLICY_GPU=""
POLICY_PORT="8899"
TRAINER_PORT="5488"
BROADCAST_PORT="5489"
RUN_DIR="$ROOT/outputs/libero_spatial_task4_openpi_pld"
SERL_TORCH_ROOT="/vla/users/niejunnan/codebase/serl_torch"
POLICY_ROOT="/vla/users/niejunnan/codebase/openpi-rlt-github"
POLICY_CONFIG="pi0_libero"
POLICY_CHECKPOINT="/vla/users/niejunnan/assets/openpi-assets/checkpoints/pi0_libero_pytorch"
ACTION_DIM="7"
WITH_POLICY_SERVER="1"
WAIT_TIMEOUT_SEC="600"

usage() {
  cat <<'EOF'
Usage: examples/libero/pld/tools/launch_pld.sh [options] [-- overrides...]

Options:
  --config PATH                PLD config path.
  --session NAME               tmux session name.
  --actor-gpu ID               GPU for actor.
  --learner-gpu ID             GPU for learner.
  --policy-gpu ID              GPU for reference-policy server. Defaults to actor GPU.
  --policy-port PORT           Reference-policy server port.
  --trainer-port PORT          Agentlace trainer port.
  --broadcast-port PORT        Agentlace broadcast port.
  --run-dir PATH               Run directory.
  --python PATH                Python for actor/learner.
  --policy-python PATH         Python for reference-policy server.
  --policy-root PATH           OpenPI/other policy checkout root.
  --policy-config NAME         Reference policy config name.
  --policy-checkpoint PATH     Reference policy checkpoint.
  --serl-torch-root PATH       serl_torch checkout used by local LIBERO backend.
  --no-policy-server           Do not start reference-policy server.
  --wait-timeout-sec SEC       Timeout for actor waiting on policy/trainer ports.

Legacy env-server options accepted but ignored: --env-gpu, --env-port,
--libero-conda-prefix, --no-env-server.
PLD now uses vla_rl.envs.libero.LiberoLocalEnvBackend in the actor process.
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
    --policy-gpu) POLICY_GPU="$2"; shift 2 ;;
    --env-port) shift 2 ;;
    --policy-port) POLICY_PORT="$2"; shift 2 ;;
    --trainer-port) TRAINER_PORT="$2"; shift 2 ;;
    --broadcast-port) BROADCAST_PORT="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --policy-python) POLICY_PYTHON_BIN="$2"; shift 2 ;;
    --policy-root) POLICY_ROOT="$2"; shift 2 ;;
    --policy-config) POLICY_CONFIG="$2"; shift 2 ;;
    --policy-checkpoint) POLICY_CHECKPOINT="$2"; shift 2 ;;
    --serl-torch-root) SERL_TORCH_ROOT="$2"; shift 2 ;;
    --libero-conda-prefix) shift 2 ;;
    --no-env-server) shift ;;
    --no-policy-server) WITH_POLICY_SERVER="0"; shift ;;
    --wait-timeout-sec) WAIT_TIMEOUT_SEC="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; OVERRIDES=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$POLICY_GPU" ]]; then POLICY_GPU="$ACTOR_GPU"; fi

COMMON_OVERRIDES=(
  "policy.url=http://127.0.0.1:${POLICY_PORT}"
  "env.serl_torch_root=${SERL_TORCH_ROOT}"
  "runtime.trainer_port=${TRAINER_PORT}"
  "runtime.broadcast_port=${BROADCAST_PORT}"
  "runtime.run_dir=${RUN_DIR}"
  "${OVERRIDES[@]}"
)

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n control "cd '$ROOT' && sleep infinity"

if [[ "$WITH_POLICY_SERVER" == "1" ]]; then
  tmux new-window -t "$SESSION" -n policy \
    "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$POLICY_GPU' '$POLICY_PYTHON_BIN' scripts/serve_reference_policy.py --policy openpi --policy-root '$POLICY_ROOT' --config-name '$POLICY_CONFIG' --checkpoint-path '$POLICY_CHECKPOINT' --action-dim '$ACTION_DIM' --device cuda --host 127.0.0.1 --port '$POLICY_PORT'"
fi

tmux new-window -t "$SESSION" -n learner \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$LEARNER_GPU' '$PYTHON_BIN' examples/libero/pld/scripts/train.py --role learner --config '$CONFIG' -- ${COMMON_OVERRIDES[*]}"

tmux new-window -t "$SESSION" -n actor \
  "cd '$ROOT' && '$PYTHON_BIN' examples/libero/rlt/tools/wait_for_tcp.py --host 127.0.0.1 --ports '$POLICY_PORT' '$TRAINER_PORT' --timeout-sec '$WAIT_TIMEOUT_SEC' && CUDA_VISIBLE_DEVICES='$ACTOR_GPU' '$PYTHON_BIN' examples/libero/pld/scripts/train.py --role actor --config '$CONFIG' -- ${COMMON_OVERRIDES[*]}"

echo "Started tmux session: $SESSION"
echo "  env:     local LIBERO backend in actor process"
echo "  policy:  GPU $POLICY_GPU, port $POLICY_PORT"
echo "  learner: GPU $LEARNER_GPU, trainer=$TRAINER_PORT broadcast=$BROADCAST_PORT"
echo "  actor:   GPU $ACTOR_GPU"
echo "  run_dir: $RUN_DIR"
echo "Attach with: tmux attach -t $SESSION"
