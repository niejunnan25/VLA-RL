#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
POLICY_PYTHON_BIN="${POLICY_PYTHON_BIN:-/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3}"
CONFIG="$ROOT/examples/libero_pld/configs/libero_spatial_task4_openpi_pld.yaml"
SESSION="vlarl_libero_pld_after_collect"
ACTOR_GPU="0"
LEARNER_GPU="1"
ENV_GPU=""
POLICY_GPU=""
ENV_PORT="23000"
POLICY_PORT="8899"
TRAINER_PORT="5488"
BROADCAST_PORT="5489"
RUN_DIR="$ROOT/outputs/libero_spatial_task4_openpi_pld"
COLLECT_OUTPUT_DIR=""
TARGET_SUCCESSES="50"
MAX_ATTEMPTS="1000"
SERL_TORCH_ROOT="/vla/users/niejunnan/codebase/serl_torch"
LIBERO_CONDA_PREFIX="/vla/users/niejunnan/envs/libero"
POLICY_ROOT="/vla/users/niejunnan/codebase/openpi-rlt-github"
POLICY_CONFIG="pi0_libero"
POLICY_CHECKPOINT="/vla/users/niejunnan/assets/openpi-assets/checkpoints/pi0_libero_pytorch"
ACTION_DIM="7"
WAIT_TIMEOUT_SEC="600"
OVERRIDES=()

usage() {
  cat <<'EOF'
Usage: examples/libero_pld/tools/launch_pld_after_collect.sh [options] [-- overrides...]

Starts env and reference-policy services, collects base-success replay, then starts PLD learner and actor.

Options:
  --config PATH                PLD config path.
  --session NAME               tmux session name.
  --actor-gpu ID               GPU for actor.
  --learner-gpu ID             GPU for learner.
  --env-gpu ID                 GPU for LIBERO env server. Defaults to actor GPU.
  --policy-gpu ID              GPU for reference-policy server. Defaults to actor GPU.
  --env-port PORT              LIBERO env server port.
  --policy-port PORT           Reference-policy server port.
  --trainer-port PORT          Agentlace trainer port.
  --broadcast-port PORT        Agentlace broadcast port.
  --run-dir PATH               Training run directory.
  --collect-output-dir PATH    Base-success replay directory.
  --target-successes N         Base-success episodes to collect.
  --max-attempts N             Collector attempt limit.
  --python PATH                Python for VLA-RL scripts.
  --policy-python PATH         Python for reference-policy server.
  --policy-root PATH           OpenPI/other policy checkout root.
  --policy-config NAME         Reference policy config name.
  --policy-checkpoint PATH     Reference policy checkpoint.
  --serl-torch-root PATH       External serl_torch checkout for LIBERO env server.
  --libero-conda-prefix PATH   Conda prefix used by LIBERO env server.
  --wait-timeout-sec SEC       Timeout for waiting on service ports.
EOF
}

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
    --trainer-port) TRAINER_PORT="$2"; shift 2 ;;
    --broadcast-port) BROADCAST_PORT="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --collect-output-dir) COLLECT_OUTPUT_DIR="$2"; shift 2 ;;
    --target-successes) TARGET_SUCCESSES="$2"; shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --policy-python) POLICY_PYTHON_BIN="$2"; shift 2 ;;
    --policy-root) POLICY_ROOT="$2"; shift 2 ;;
    --policy-config) POLICY_CONFIG="$2"; shift 2 ;;
    --policy-checkpoint) POLICY_CHECKPOINT="$2"; shift 2 ;;
    --serl-torch-root) SERL_TORCH_ROOT="$2"; shift 2 ;;
    --libero-conda-prefix) LIBERO_CONDA_PREFIX="$2"; shift 2 ;;
    --wait-timeout-sec) WAIT_TIMEOUT_SEC="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; OVERRIDES=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$ENV_GPU" ]]; then ENV_GPU="$ACTOR_GPU"; fi
if [[ -z "$POLICY_GPU" ]]; then POLICY_GPU="$ACTOR_GPU"; fi
if [[ -z "$COLLECT_OUTPUT_DIR" ]]; then COLLECT_OUTPUT_DIR="$RUN_DIR/base_success_replay"; fi

COMMON_OVERRIDES=(
  "env.url=http://127.0.0.1:${ENV_PORT}"
  "policy.url=http://127.0.0.1:${POLICY_PORT}"
  "runtime.trainer_port=${TRAINER_PORT}"
  "runtime.broadcast_port=${BROADCAST_PORT}"
  "runtime.run_dir=${RUN_DIR}"
  "runtime.offline_replay_path=${COLLECT_OUTPUT_DIR}"
  "collect.output_dir=${COLLECT_OUTPUT_DIR}"
  "${OVERRIDES[@]}"
)

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n env \
  "cd '$SERL_TORCH_ROOT' && LIBERO_CONDA_PREFIX='$LIBERO_CONDA_PREFIX' bash examples/libero/tools/serve_env.sh --host 127.0.0.1 --port '$ENV_PORT' --gpu-id '$ENV_GPU'"

tmux new-window -t "$SESSION" -n policy \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$POLICY_GPU' '$POLICY_PYTHON_BIN' scripts/serve_reference_policy.py --policy openpi --policy-root '$POLICY_ROOT' --config-name '$POLICY_CONFIG' --checkpoint-path '$POLICY_CHECKPOINT' --action-dim '$ACTION_DIM' --device cuda --host 127.0.0.1 --port '$POLICY_PORT'"

tmux new-window -t "$SESSION" -n collect \
  "cd '$ROOT' && '$PYTHON_BIN' examples/libero_rlt/tools/wait_for_tcp.py --host 127.0.0.1 --ports '$ENV_PORT' '$POLICY_PORT' --timeout-sec '$WAIT_TIMEOUT_SEC' && CUDA_VISIBLE_DEVICES='$ACTOR_GPU' '$PYTHON_BIN' examples/libero_pld/collect_base_success_replay.py --config '$CONFIG' --output-dir '$COLLECT_OUTPUT_DIR' --target-successes '$TARGET_SUCCESSES' --max-attempts '$MAX_ATTEMPTS' -- ${COMMON_OVERRIDES[*]} && tmux wait-for -S '${SESSION}_collect_done'"

tmux new-window -t "$SESSION" -n learner \
  "cd '$ROOT' && tmux wait-for '${SESSION}_collect_done' && CUDA_VISIBLE_DEVICES='$LEARNER_GPU' '$PYTHON_BIN' examples/libero_pld/train.py --role learner --config '$CONFIG' -- ${COMMON_OVERRIDES[*]}"

tmux new-window -t "$SESSION" -n actor \
  "cd '$ROOT' && tmux wait-for '${SESSION}_collect_done' && '$PYTHON_BIN' examples/libero_rlt/tools/wait_for_tcp.py --host 127.0.0.1 --ports '$ENV_PORT' '$POLICY_PORT' '$TRAINER_PORT' --timeout-sec '$WAIT_TIMEOUT_SEC' && CUDA_VISIBLE_DEVICES='$ACTOR_GPU' '$PYTHON_BIN' examples/libero_pld/train.py --role actor --config '$CONFIG' -- ${COMMON_OVERRIDES[*]}"

echo "Started PLD collect-then-train tmux session: $SESSION"
echo "  collect output: $COLLECT_OUTPUT_DIR"
echo "  run_dir:        $RUN_DIR"
echo "Attach with: tmux attach -t $SESSION"
