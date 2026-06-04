#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="$ROOT/examples/libero_rlt/configs/libero_spatial_task4_openpi_rlt.yaml"
TASK_IDS="1,2,17,21,33,34,46,47"
SESSION_PREFIX="vlarl_rlt_libero90"
RUN_ROOT="$ROOT/outputs/rlt_libero90_8tasks"
ACTOR_GPUS="0,2,4,6,0,2,4,6"
LEARNER_GPUS="1,3,5,7,1,3,5,7"
ENV_PORT_START="24000"
POLICY_PORT_START="24100"
TRAINER_PORT_START="24200"
BROADCAST_PORT_START="24300"
DRY_RUN="0"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: examples/libero_rlt/tools/launch_libero90_8tasks_example.sh [options] [-- overrides...]

Starts one RLT tmux session per task by calling examples/libero_rlt/tools/launch_rlt.sh.
All paths, ports, GPUs, and checkpoints can be overridden; defaults are examples.

Options:
  --config PATH                Base RLT config.
  --task-ids CSV               LIBERO task ids. Default: 1,2,17,21,33,34,46,47.
  --session-prefix NAME        Prefix for tmux sessions.
  --run-root PATH              Directory containing per-task run dirs.
  --actor-gpus CSV             Actor/env/policy GPU list.
  --learner-gpus CSV           Learner GPU list.
  --env-port-start PORT        First env port.
  --policy-port-start PORT     First policy port.
  --trainer-port-start PORT    First trainer port.
  --broadcast-port-start PORT  First broadcast port.
  --dry-run                    Print commands without running them.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --task-ids) TASK_IDS="$2"; shift 2 ;;
    --session-prefix) SESSION_PREFIX="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --actor-gpus) ACTOR_GPUS="$2"; shift 2 ;;
    --learner-gpus) LEARNER_GPUS="$2"; shift 2 ;;
    --env-port-start) ENV_PORT_START="$2"; shift 2 ;;
    --policy-port-start) POLICY_PORT_START="$2"; shift 2 ;;
    --trainer-port-start) TRAINER_PORT_START="$2"; shift 2 ;;
    --broadcast-port-start) BROADCAST_PORT_START="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

IFS=',' read -r -a TASK_ARRAY <<< "$TASK_IDS"
IFS=',' read -r -a ACTOR_GPU_ARRAY <<< "$ACTOR_GPUS"
IFS=',' read -r -a LEARNER_GPU_ARRAY <<< "$LEARNER_GPUS"

for idx in "${!TASK_ARRAY[@]}"; do
  task_id="${TASK_ARRAY[$idx]}"
  actor_gpu="${ACTOR_GPU_ARRAY[$((idx % ${#ACTOR_GPU_ARRAY[@]}))]}"
  learner_gpu="${LEARNER_GPU_ARRAY[$((idx % ${#LEARNER_GPU_ARRAY[@]}))]}"
  env_port=$((ENV_PORT_START + idx))
  policy_port=$((POLICY_PORT_START + idx))
  trainer_port=$((TRAINER_PORT_START + idx))
  broadcast_port=$((BROADCAST_PORT_START + idx))
  session="${SESSION_PREFIX}_task${task_id}"
  run_dir="${RUN_ROOT}/task${task_id}"
  cmd=(
    bash "$ROOT/examples/libero_rlt/tools/launch_rlt.sh"
    --config "$CONFIG"
    --session "$session"
    --actor-gpu "$actor_gpu"
    --learner-gpu "$learner_gpu"
    --env-gpu "$actor_gpu"
    --policy-gpu "$actor_gpu"
    --env-port "$env_port"
    --policy-port "$policy_port"
    --trainer-port "$trainer_port"
    --broadcast-port "$broadcast_port"
    --run-dir "$run_dir"
    --
    env.task_suite_name=libero_90
    env.task_id="$task_id"
    wandb.exp_name="rlt_libero90_task${task_id}"
    "${EXTRA_ARGS[@]}"
  )
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}"
  fi
done
