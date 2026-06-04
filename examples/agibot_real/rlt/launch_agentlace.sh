#!/usr/bin/env bash
set -euo pipefail
CONFIG=${CONFIG:-examples/agibot_real/rlt/configs/train_rlt.yaml}
RUN_DIR=${RUN_DIR:-/tmp/vlarl_agibot_rlt}
TRAINER_PORT=${TRAINER_PORT:-6688}
BROADCAST_PORT=${BROADCAST_PORT:-6689}
POLICY_PORT=${POLICY_PORT:-8899}
ACTOR_GPU=${ACTOR_GPU:-0}
LEARNER_GPU=${LEARNER_GPU:-1}
POLICY_GPU=${POLICY_GPU:-$ACTOR_GPU}
POLICY_PYTHON=${POLICY_PYTHON:-/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3}
POLICY_ROOT=${POLICY_ROOT:-/vla/users/niejunnan/codebase/openpi-rlt-github}
POLICY_CONFIG=${POLICY_CONFIG:-pi0_libero}
POLICY_CHECKPOINT=${POLICY_CHECKPOINT:-/vla/users/niejunnan/assets/openpi-assets/checkpoints/pi0_libero_pytorch}
ACTION_DIM=${ACTION_DIM:-7}
START_POLICY_SERVER=${START_POLICY_SERVER:-1}
SESSION=${SESSION:-vlarl_agibot_rlt}

cd "$(dirname "$0")/../../.."
ROOT=$(pwd)

tmux new-session -d -s "$SESSION" -n learner "source /vla/miniconda3/etc/profile.d/conda.sh && conda activate serl_torch && export CUDA_VISIBLE_DEVICES=$LEARNER_GPU && cd '$ROOT' && python examples/agibot_real/rlt/train.py --config $CONFIG --role learner -- runtime.run_dir=$RUN_DIR runtime.trainer_port=$TRAINER_PORT runtime.broadcast_port=$BROADCAST_PORT ${*:-}"

if [[ "$START_POLICY_SERVER" == "1" ]]; then
  tmux new-window -t "$SESSION" -n policy "cd '$ROOT' && CUDA_VISIBLE_DEVICES=$POLICY_GPU '$POLICY_PYTHON' scripts/serve_reference_policy.py --policy openpi --policy-root '$POLICY_ROOT' --config-name '$POLICY_CONFIG' --checkpoint-path '$POLICY_CHECKPOINT' --action-dim '$ACTION_DIM' --device cuda --host 127.0.0.1 --port '$POLICY_PORT'"
fi

tmux new-window -t "$SESSION" -n actor "source /vla/miniconda3/etc/profile.d/conda.sh && conda activate serl_torch && export CUDA_VISIBLE_DEVICES=$ACTOR_GPU && cd '$ROOT' && python examples/libero/rlt/tools/wait_for_tcp.py 127.0.0.1 $POLICY_PORT --timeout 300 && python examples/agibot_real/rlt/train.py --config $CONFIG --role actor -- runtime.run_dir=$RUN_DIR runtime.trainer_port=$TRAINER_PORT runtime.broadcast_port=$BROADCAST_PORT policy.url=http://127.0.0.1:$POLICY_PORT ${*:-}"

echo "tmux attach -t $SESSION"
