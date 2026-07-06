#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/vla/users/niejunnan/envs/robo-dopamine/bin/python}"
ROBODOPAMINE_ROOT="${ROBODOPAMINE_ROOT:-/vla/users/niejunnan/codebase/Robo-Dopamine}"
MODEL_PATH="${MODEL_PATH:-/vla/users/niejunnan/assets/Robo-Dopamine-GRM-2.0-4B-Preview}"
GOAL_DATASET="${GOAL_DATASET:-/vla/users/niejunnan/datasets/libero_lerobot}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-52000}"
GPU="${GPU:-0}"
DEFAULT_OUT_ROOT="/dev/shm/robodopamine_rlpd_reward_http"
if [[ ! -d "/dev/shm" ]]; then
  DEFAULT_OUT_ROOT="/tmp/robodopamine_rlpd_reward_http"
fi
OUT_ROOT="${OUT_ROOT:-$DEFAULT_OUT_ROOT}"
FORWARD_BATCH_SIZE="${FORWARD_BATCH_SIZE:-8}"
IMAGE_TRANSPORT="${IMAGE_TRANSPORT:-memory}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
ROBO_DOPAMINE_VLLM_GPU_MEMORY_UTILIZATION="${ROBO_DOPAMINE_VLLM_GPU_MEMORY_UTILIZATION:-0.4}"

export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND
export ROBO_DOPAMINE_VLLM_GPU_MEMORY_UTILIZATION

if [[ -d "/vla/users/niejunnan/envs/robo-dopamine/lib/python3.10/site-packages/nvidia" ]]; then
  NVIDIA_LIBS="$(find /vla/users/niejunnan/envs/robo-dopamine/lib/python3.10/site-packages/nvidia -mindepth 2 -maxdepth 2 -type d -name lib | paste -sd: -)"
  export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$ROOT/scripts/serve_robodopamine_progress_http.py" \
  --robodopamine-root "$ROBODOPAMINE_ROOT" \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --image-keys image_rgb_0 image_rgb_1 \
  --view-mode two_view_copy_main \
  --image-preprocess none \
  --goal-image-preprocess none \
  --image-transport "$IMAGE_TRANSPORT" \
  --eval-modes forward \
  --batch-size "$FORWARD_BATCH_SIZE" \
  --goal-dataset "$GOAL_DATASET" \
  --out-root "$OUT_ROOT"
