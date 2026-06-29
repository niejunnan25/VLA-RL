#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/vla/users/niejunnan/envs/serl_torch/bin/python}"
POLICY_PYTHON_BIN="${POLICY_PYTHON_BIN:-/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3}"
CONFIG="$ROOT/examples/libero/pld/configs/reward_model/libero_spatial_task0_residual_sac_sparse.yaml"
SESSION="vlarl_libero_residual_sac"
ACTOR_GPU="0"
LEARNER_GPU="1"
POLICY_GPU=""
POLICY_PORT="8899"
WITH_EVAL="1"
EVAL_GPU=""
EVAL_POLICY_PORT="8999"
TRAINER_PORT="5488"
BROADCAST_PORT="5489"
RUN_DIR=""
SERL_TORCH_ROOT="/vla/users/niejunnan/codebase/serl_torch"
POLICY_ROOT="/vla/users/niejunnan/codebase/openpi-modified"
POLICY_CONFIG="pi0_libero_baseline_10_bs32_150000"
POLICY_CHECKPOINT="/vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000"
ACTION_DIM="7"
WITH_POLICY_SERVER="1"
WAIT_TIMEOUT_SEC="600"
POLICY_XLA_PREALLOCATE="${POLICY_XLA_PREALLOCATE:-false}"
POLICY_XLA_MEM_FRACTION="${POLICY_XLA_MEM_FRACTION:-0.80}"

usage() {
  cat <<'EOF'
Usage: examples/libero/pld/tools/launch_residual_sac.sh [options] [-- overrides...]

Starts online chunk-level residual SAC under the PLD family:
  frozen OpenPI base action chunk + learned residual action chunk -> env.step_chunk.

Options:
  --config PATH                Residual SAC config path.
  --session NAME               tmux session name.
  --actor-gpu ID               GPU for actor and actor-local LIBERO env.
  --learner-gpu ID             GPU for learner.
  --policy-gpu ID              GPU for reference-policy server. Defaults to actor GPU.
  --policy-port PORT           Reference-policy server port.
  --with-eval                  Start async eval worker plus dedicated eval policy service. Enabled by default.
  --no-eval                    Disable async eval for debugging.
  --eval-gpu ID                GPU for async eval worker and eval policy. Defaults to actor GPU.
  --eval-policy-port PORT      Async eval reference-policy server port.
  --trainer-port PORT          Agentlace trainer port.
  --broadcast-port PORT        Agentlace broadcast port.
  --run-dir PATH               Optional run directory override. Defaults to YAML runtime.run_dir.
  --python PATH                Python for actor/learner/eval worker.
  --policy-python PATH         Python for reference-policy server.
  --policy-root PATH           OpenPI/other policy checkout root.
  --policy-config NAME         Reference policy config name.
  --policy-checkpoint PATH     Reference policy checkpoint.
  --serl-torch-root PATH       serl_torch checkout used by local LIBERO backend.
  --no-policy-server           Do not start reference-policy server.
  --wait-timeout-sec SEC       Timeout for actor waiting on policy/trainer ports.

Reward model servers are intentionally external. Start Robo-Dopamine or RoboMeter
progress services separately, then point reward.remote.url in the YAML to them.
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
    --with-eval) WITH_EVAL="1"; shift ;;
    --no-eval) WITH_EVAL="0"; shift ;;
    --eval-gpu) EVAL_GPU="$2"; shift 2 ;;
    --eval-env-port) shift 2 ;;
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
    --libero-conda-prefix) shift 2 ;;
    --with-env-server) echo "ERROR: launch_residual_sac.sh uses the local LIBERO backend; remove --with-env-server." >&2; exit 2 ;;
    --no-env-server) shift ;;
    --no-policy-server) WITH_POLICY_SERVER="0"; shift ;;
    --wait-timeout-sec) WAIT_TIMEOUT_SEC="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; OVERRIDES=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$POLICY_GPU" ]]; then POLICY_GPU="$ACTOR_GPU"; fi
if [[ -z "$EVAL_GPU" ]]; then EVAL_GPU="$ACTOR_GPU"; fi

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi

if grep -q "vla_rl.policies.OpenPIWebsocketPolicyClient" "$CONFIG"; then
  if [[ "$WITH_POLICY_SERVER" == "1" ]]; then
    echo "ERROR: $CONFIG uses OpenPIWebsocketPolicyClient, but launch_residual_sac.sh starts the VLA-RL HTTP policy server." >&2
    echo "Use launch_residual_sac_remote_isolated.sh --policy-mode official_ws, or pass --no-policy-server and provide an external OpenPI websocket server." >&2
    exit 2
  fi
  if [[ "$WITH_EVAL" == "1" && "$EVAL_POLICY_PORT" != "$POLICY_PORT" ]]; then
    echo "ERROR: OpenPIWebsocketPolicyClient configs cannot use launch_residual_sac.sh to start a separate eval HTTP policy server; set --eval-policy-port equal to --policy-port or use launch_residual_sac_remote_isolated.sh --policy-mode official_ws." >&2
    exit 2
  fi
fi

COMMON_OVERRIDES=(
  "policy.url=http://127.0.0.1:${POLICY_PORT}"
  "env.serl_torch_root=${SERL_TORCH_ROOT}"
  "runtime.trainer_port=${TRAINER_PORT}"
  "runtime.broadcast_port=${BROADCAST_PORT}"
  "${OVERRIDES[@]}"
)

if [[ -n "$RUN_DIR" ]]; then
  COMMON_OVERRIDES+=("runtime.run_dir=${RUN_DIR}")
fi

if [[ "$WITH_EVAL" == "1" ]]; then
  COMMON_OVERRIDES+=(
    "runtime.async_eval.enabled=true"
    "runtime.async_eval.policy_url=http://127.0.0.1:${EVAL_POLICY_PORT}"
    "runtime.async_eval.worker_cuda_visible_devices=${EVAL_GPU}"
    "runtime.async_eval.worker_mujoco_egl_device_id=${EVAL_GPU}"
  )
else
  COMMON_OVERRIDES+=("runtime.async_eval.enabled=false")
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n control "cd '$ROOT' && sleep infinity"

if [[ "$WITH_POLICY_SERVER" == "1" ]]; then
  tmux new-window -t "$SESSION" -n policy \
    "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$POLICY_GPU' XLA_PYTHON_CLIENT_PREALLOCATE='$POLICY_XLA_PREALLOCATE' XLA_PYTHON_CLIENT_MEM_FRACTION='$POLICY_XLA_MEM_FRACTION' '$POLICY_PYTHON_BIN' scripts/serve_reference_policy.py --policy openpi --policy-root '$POLICY_ROOT' --config-name '$POLICY_CONFIG' --checkpoint-path '$POLICY_CHECKPOINT' --action-dim '$ACTION_DIM' --device cuda --host 127.0.0.1 --port '$POLICY_PORT'"
fi

if [[ "$WITH_EVAL" == "1" && "$EVAL_POLICY_PORT" != "$POLICY_PORT" ]]; then
  tmux new-window -t "$SESSION" -n eval-policy \
    "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$EVAL_GPU' XLA_PYTHON_CLIENT_PREALLOCATE='$POLICY_XLA_PREALLOCATE' XLA_PYTHON_CLIENT_MEM_FRACTION='$POLICY_XLA_MEM_FRACTION' '$POLICY_PYTHON_BIN' scripts/serve_reference_policy.py --policy openpi --policy-root '$POLICY_ROOT' --config-name '$POLICY_CONFIG' --checkpoint-path '$POLICY_CHECKPOINT' --action-dim '$ACTION_DIM' --device cuda --host 127.0.0.1 --port '$EVAL_POLICY_PORT'"
elif [[ "$WITH_EVAL" == "1" ]]; then
  echo "Reusing reference-policy server for async eval on port $POLICY_PORT"
fi

tmux new-window -t "$SESSION" -n learner \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$LEARNER_GPU' MUJOCO_EGL_DEVICE_ID='$LEARNER_GPU' '$PYTHON_BIN' examples/libero/pld/scripts/train_residual_sac.py --role learner --config '$CONFIG' -- ${COMMON_OVERRIDES[*]}"

WAIT_PORTS=("$POLICY_PORT" "$TRAINER_PORT")
if [[ "$WITH_EVAL" == "1" && "$EVAL_POLICY_PORT" != "$POLICY_PORT" ]]; then
  WAIT_PORTS+=("$EVAL_POLICY_PORT")
fi
WAIT_PORT_ARGS="${WAIT_PORTS[*]}"

tmux new-window -t "$SESSION" -n actor \
  "cd '$ROOT' && '$PYTHON_BIN' examples/libero/rlt/tools/wait_for_tcp.py --host 127.0.0.1 --ports $WAIT_PORT_ARGS --timeout-sec '$WAIT_TIMEOUT_SEC' && CUDA_VISIBLE_DEVICES='$ACTOR_GPU' MUJOCO_EGL_DEVICE_ID='$ACTOR_GPU' '$PYTHON_BIN' examples/libero/pld/scripts/train_residual_sac.py --role actor --config '$CONFIG' -- ${COMMON_OVERRIDES[*]}"

echo "Started residual SAC tmux session: $SESSION"
echo "  env:     local in actor process, GPU $ACTOR_GPU"
echo "  policy:  GPU $POLICY_GPU, port $POLICY_PORT"
if [[ "$WITH_EVAL" == "1" ]]; then
  if [[ "$EVAL_POLICY_PORT" == "$POLICY_PORT" ]]; then
    echo "  eval:    local env in worker, GPU $EVAL_GPU, policy=reuse:$POLICY_PORT"
  else
    echo "  eval:    local env in worker, GPU $EVAL_GPU, policy=$EVAL_POLICY_PORT"
  fi
fi
echo "  learner: GPU $LEARNER_GPU, trainer=$TRAINER_PORT broadcast=$BROADCAST_PORT"
echo "  actor:   GPU $ACTOR_GPU"
if [[ -n "$RUN_DIR" ]]; then
  echo "  run_dir override: $RUN_DIR"
else
  echo "  run_dir: from YAML runtime.run_dir"
fi
echo "Attach with: tmux attach -t $SESSION"
