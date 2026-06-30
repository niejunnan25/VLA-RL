#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_DIR="$(cd "$TOOLS_DIR/.." && pwd)"
REPO_ROOT="$(cd "$LIBERO_DIR/../../.." && pwd)"

TASK="4"
REWARD="sparse"
ACTOR_GPU="0"
LEARNER_GPU="1"
ENV_GPU=""
EVAL_ENV_GPU=""
POLICY_GPU=""
BACKFILL_GPU=""
REWARD_MODEL_GPU=""
OUTPUT_ROOT=""
DRY_RUN="${DRY_RUN:-0}"

MAX_ENV_STEPS="${MAX_ENV_STEPS:-2000}"
TRAINING_STARTS="${TRAINING_STARTS:-20}"
ASYNC_EVAL_EVERY_EPISODES="${ASYNC_EVAL_EVERY_EPISODES:-5}"
ASYNC_EVAL_EPISODES="${ASYNC_EVAL_EPISODES:-2}"

usage() {
  cat <<'EOF'
Usage:
  bash examples/libero/residual_sac/tools/smoke_residual_sac.sh \
    --task 4 \
    --reward sparse|robodopamine|robometer \
    --actor-gpu 0 \
    --learner-gpu 1 \
    [--reward-model-gpu 1]

Environment overrides:
  MAX_ENV_STEPS=2000
  TRAINING_STARTS=20
  ASYNC_EVAL_EVERY_EPISODES=5
  ASYNC_EVAL_EPISODES=2
  DRY_RUN=1

The smoke writes to examples/libero/residual_sac/outputs/smoke by default and
disables online logging. It is a connectivity check, not a formal result.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --reward)
      REWARD="$2"
      shift 2
      ;;
    --actor-gpu)
      ACTOR_GPU="$2"
      shift 2
      ;;
    --learner-gpu)
      LEARNER_GPU="$2"
      shift 2
      ;;
    --env-gpu)
      ENV_GPU="$2"
      shift 2
      ;;
    --eval-env-gpu)
      EVAL_ENV_GPU="$2"
      shift 2
      ;;
    --policy-gpu)
      POLICY_GPU="$2"
      shift 2
      ;;
    --backfill-gpu)
      BACKFILL_GPU="$2"
      shift 2
      ;;
    --reward-model-gpu)
      REWARD_MODEL_GPU="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$TASK" in
  4|5|8|9) ;;
  *) echo "ERROR: --task must be one of 4, 5, 8, 9" >&2; exit 1 ;;
esac

ENV_GPU="${ENV_GPU:-$ACTOR_GPU}"
EVAL_ENV_GPU="${EVAL_ENV_GPU:-$ACTOR_GPU}"
POLICY_GPU="${POLICY_GPU:-$ACTOR_GPU}"
BACKFILL_GPU="${BACKFILL_GPU:-$ACTOR_GPU}"
REWARD_MODEL_GPU="${REWARD_MODEL_GPU:-$LEARNER_GPU}"

TASK_PADDED="$(printf "%02d" "$TASK")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

case "$REWARD" in
  sparse)
    CONFIG_NAME="libero_spatial_task${TASK}_sparse"
    REWARD_MODEL_FLAG="false"
    ;;
  robodopamine)
    CONFIG_NAME="reward_model/libero_spatial_task${TASK}_robodopamine_pbrs"
    REWARD_MODEL_FLAG="true"
    ;;
  robometer)
    CONFIG_NAME="reward_model/libero_spatial_task${TASK}_robometer_pbrs"
    REWARD_MODEL_FLAG="true"
    ;;
  *)
    echo "ERROR: --reward must be sparse, robodopamine, or robometer" >&2
    exit 1
    ;;
esac

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$LIBERO_DIR/outputs/smoke/${REWARD}_task${TASK_PADDED}_${TIMESTAMP}"
fi

cmd=(
  bash "$TOOLS_DIR/launch_residual_sac.sh"
  --mode chunk
  --config-name "$CONFIG_NAME"
  --output-root "$OUTPUT_ROOT"
  --learner-gpu "$LEARNER_GPU"
  --actor-gpu "$ACTOR_GPU"
  --env-gpu "$ENV_GPU"
  --eval-env-gpu "$EVAL_ENV_GPU"
  --policy-gpu "$POLICY_GPU"
  --backfill-gpu "$BACKFILL_GPU"
  --policy-server managed
  --reward-model "$REWARD_MODEL_FLAG"
  --with-eval-env
  --learner-gpu-memory-guard-fraction 0
  --clean-output-dir
)

if [[ "$REWARD_MODEL_FLAG" == "true" ]]; then
  cmd+=(--reward-model-gpu "$REWARD_MODEL_GPU")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

cmd+=(
  --
  "training.max_env_steps=$MAX_ENV_STEPS"
  "training.training_starts=$TRAINING_STARTS"
  "training.torch_compile.enabled=false"
  "training.async_eval.every_episodes=$ASYNC_EVAL_EVERY_EPISODES"
  "training.async_eval.episodes=$ASYNC_EVAL_EPISODES"
  "wandb.mode=disabled"
)

cd "$REPO_ROOT"
echo "[smoke] reward=$REWARD task=$TASK output_root=$OUTPUT_ROOT"
printf '[smoke] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
