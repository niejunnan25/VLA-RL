#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/vla/users/niejunnan/envs/serl_torch/bin/python}"
ROBOMETER_ROOT="${ROBOMETER_ROOT:-/vla/users/niejunnan/workspace/robometer}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/vla/users/niejunnan/assets/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/vla/users/niejunnan/assets/hf_cache/hub}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-20}"
export ROBOMETER_PROCESSED_DATASETS_PATH="${ROBOMETER_PROCESSED_DATASETS_PATH:-/vla/users/niejunnan/assets/robometer_processed_datasets}"
export ROBOMETER_SENTENCE_MODEL_PATH="${ROBOMETER_SENTENCE_MODEL_PATH:-/vla/users/niejunnan/assets/sentence-transformers/all-MiniLM-L12-v2}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export USE_TF="${USE_TF:-0}"
export USE_FLAX="${USE_FLAX:-0}"
if [[ -z "${ROBOMETER_PYTHON_CMD:-}" ]]; then
  ROBOMETER_VENV_PYTHON="$ROBOMETER_ROOT/.venv/bin/python"
  if [[ -x "$ROBOMETER_VENV_PYTHON" ]] && "$ROBOMETER_VENV_PYTHON" - <<'PY_CHECK' >/dev/null 2>&1
import fastapi
import omegaconf
import torch
import uvicorn
PY_CHECK
  then
    ROBOMETER_PYTHON_CMD="$ROBOMETER_VENV_PYTHON"
  else
    ROBOMETER_PYTHON_CMD="uv run python"
  fi
fi
MODEL_PATH="${MODEL_PATH:-/vla/users/niejunnan/assets/Robometer-4B}"
HOST="${HOST:-127.0.0.1}"
ROBOMETER_PORT="${ROBOMETER_PORT:-8401}"
ADAPTER_PORT="${ADAPTER_PORT:-50152}"
GPU="${GPU:-0}"
BACKEND="${BACKEND:-native}"
QUERY_MODE="${QUERY_MODE:-prefix_per_query}"
FORWARD_BATCH_SIZE="${FORWARD_BATCH_SIZE:-4}"
MAX_HISTORY_FRAMES="${MAX_HISTORY_FRAMES:-8}"
VIEW_MODE="${VIEW_MODE:-first}"
ROBOMETER_IMAGE_KEYS="${ROBOMETER_IMAGE_KEYS:-image_rgb_0}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-600}"
ROBOMETER_USE_UNSLOTH="${ROBOMETER_USE_UNSLOTH:-true}"
ROBOMETER_EXTRA_ARGS="${ROBOMETER_EXTRA_ARGS:-}"
ROBOMETER_IMPORT_STUB_DIR="${ROBOMETER_IMPORT_STUB_DIR:-$ROOT/examples/libero/rlpd/tools/robometer_import_stubs}"
ROBOMETER_DISABLE_IMPORT_STUBS="${ROBOMETER_DISABLE_IMPORT_STUBS:-1}"

cleanup() {
  if [[ -n "${ROBOMETER_PID:-}" ]]; then
    kill "$ROBOMETER_PID" 2>/dev/null || true
    wait "$ROBOMETER_PID" 2>/dev/null || true
  fi
  if [[ -n "${ROBOMETER_TEMP_MODEL_DIR:-}" ]]; then
    rm -rf "$ROBOMETER_TEMP_MODEL_DIR"
  fi
}
trap cleanup EXIT INT TERM

MODEL_PATH_FOR_SERVER="$MODEL_PATH"
if [[ "$ROBOMETER_USE_UNSLOTH" == "false" && -f "$MODEL_PATH/config.yaml" ]]; then
  ROBOMETER_TEMP_MODEL_DIR="$(mktemp -d /tmp/vlarl_robometer_model.XXXXXX)"
  for item in "$MODEL_PATH"/* "$MODEL_PATH"/.[!.]* "$MODEL_PATH"/..?*; do
    [[ -e "$item" ]] || continue
    ln -s "$item" "$ROBOMETER_TEMP_MODEL_DIR/$(basename "$item")"
  done
  rm -f "$ROBOMETER_TEMP_MODEL_DIR/config.yaml"
  "$PYTHON_BIN" - "$MODEL_PATH/config.yaml" "$ROBOMETER_TEMP_MODEL_DIR/config.yaml" <<'PY_CONFIG'
from pathlib import Path
import re
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text()
text = re.sub(r"(^\s*use_unsloth:\s*)true\b", r"\1false", text, flags=re.MULTILINE)
dst.write_text(text)
PY_CONFIG
  MODEL_PATH_FOR_SERVER="$ROBOMETER_TEMP_MODEL_DIR"
fi

if [[ "$BACKEND" == "http" ]]; then
  (
    cd "$ROBOMETER_ROOT"
    export CUDA_VISIBLE_DEVICES="$GPU"
    export PYTHONUNBUFFERED=1
    if [[ "$ROBOMETER_DISABLE_IMPORT_STUBS" != "1" ]]; then
      export PYTHONPATH="$ROBOMETER_IMPORT_STUB_DIR${PYTHONPATH:+:$PYTHONPATH}"
    fi
    $ROBOMETER_PYTHON_CMD robometer/evals/eval_server.py \
      model_path="$MODEL_PATH_FOR_SERVER" \
      num_gpus=1 \
      max_workers=1 \
      server_url="$HOST" \
      server_port="$ROBOMETER_PORT" \
      $ROBOMETER_EXTRA_ARGS
  ) &
  ROBOMETER_PID=$!

  "$PYTHON_BIN" "$ROOT/examples/libero/rlt/tools/wait_for_tcp.py" \
    --host "$HOST" \
    --ports "$ROBOMETER_PORT" \
    --timeout-sec "$WAIT_TIMEOUT_SEC"

    # Intentionally leave ROBOMETER_IMAGE_KEYS unquoted so users can pass
    # multiple space-separated keys when running non-LIBERO multi-view ablations.
    "$PYTHON_BIN" "$ROOT/scripts/serve_robometer_progress_http.py" \
      --backend http \
      --robometer-url "http://${HOST}:${ROBOMETER_PORT}" \
      --host "$HOST" \
      --port "$ADAPTER_PORT" \
      --image-keys $ROBOMETER_IMAGE_KEYS \
      --view-mode "$VIEW_MODE" \
      --query-mode "$QUERY_MODE" \
      --max-history-frames "$MAX_HISTORY_FRAMES"
else
  (
    cd "$ROBOMETER_ROOT"
    export CUDA_VISIBLE_DEVICES="$GPU"
    export PYTHONUNBUFFERED=1
    export ROBOMETER_ROOT="$ROBOMETER_ROOT"
    if [[ "$ROBOMETER_DISABLE_IMPORT_STUBS" != "1" ]]; then
      export PYTHONPATH="$ROOT:$ROBOMETER_IMPORT_STUB_DIR${PYTHONPATH:+:$PYTHONPATH}"
    else
      export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    fi
    # Intentionally leave ROBOMETER_IMAGE_KEYS unquoted so users can pass
    # multiple space-separated keys when running non-LIBERO multi-view ablations.
    $ROBOMETER_PYTHON_CMD "$ROOT/scripts/serve_robometer_progress_http.py" \
      --backend native \
      --model-path "$MODEL_PATH_FOR_SERVER" \
      --device cuda \
      --forward-batch-size "$FORWARD_BATCH_SIZE" \
      --robometer-root "$ROBOMETER_ROOT" \
      --host "$HOST" \
      --port "$ADAPTER_PORT" \
      --image-keys $ROBOMETER_IMAGE_KEYS \
      --view-mode "$VIEW_MODE" \
      --query-mode "$QUERY_MODE" \
      --max-history-frames "$MAX_HISTORY_FRAMES"
  )
fi
