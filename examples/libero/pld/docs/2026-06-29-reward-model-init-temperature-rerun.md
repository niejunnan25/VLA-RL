# 2026-06-29 PLD reward_model init_temperature=1.0 rerun record

This note records the 2026-06-29 PLD residual SAC rerun for LIBERO spatial tasks
4, 5, 8, and 9. The purpose was to replace the previous `init_temperature=0.01`
runs with the corrected `init_temperature=1.0` YAMLs, relaunch the same 12
experiment groups, and verify early training speed and link health.

All paths below are under:

```bash
cd /vla/users/niejunnan/codebase/VLA-RL
```

## Code version and dirty state

The run was launched from:

```text
branch: feature/rlt-reward-model-integration
commit: 34887dd564e6bdaa06a76d1e176d20291cbd21ea
```

Important reproducibility caveat: this was not a clean git tree. The run depends
on uncommitted YAML edits in the 20 reward-model configs:

```text
examples/libero/pld/configs/reward_model/libero_spatial_task{0..9}_residual_sac_robodopamine_pbrs.yaml
examples/libero/pld/configs/reward_model/libero_spatial_task{0..9}_residual_sac_robometer_pbrs.yaml
```

The relevant diff is:

```diff
reward:
  async:
    max_pending_chunks: ...
-   batch_size: 8
+   batch_size: 1
    max_wait_ms: 5
```

There were also unrelated local changes in policy/eval code at the time of
documentation. Do not assume `34887dd` alone is enough to reproduce these runs;
either keep the dirty YAML edits or commit them.

## Scope

The rerun covered 12 groups:

| host | method | tasks | standard output root |
|---|---|---:|---|
| 204 | sparse | 4, 5, 8, 9 | `examples/libero/pld/outputs/reward_model/libero_spatial_residual_sac_sparse_300k` |
| 234 | Robo-Dopamine PBRS | 4, 5, 8, 9 | `examples/libero/pld/outputs/reward_model/libero_spatial_residual_sac_robodopamine_pbrs_300k` |
| 239 | RoboMeter PBRS | 4, 5, 8, 9 | `examples/libero/pld/outputs/reward_model/libero_spatial_residual_sac_robometer_pbrs_300k` |

The standard `runtime.run_dir` paths in YAML are intentionally reused for the
latest mainline run. Old `0.01` runs were archived before relaunch.

## YAML state

The authoritative configs are:

```text
examples/libero/pld/configs/reward_model/libero_spatial_task{0..9}_residual_sac_sparse.yaml
examples/libero/pld/configs/reward_model/libero_spatial_task{0..9}_residual_sac_robodopamine_pbrs.yaml
examples/libero/pld/configs/reward_model/libero_spatial_task{0..9}_residual_sac_robometer_pbrs.yaml
```

The 30 formal YAMLs were checked on 234:

```bash
python3 - <<'PY'
from pathlib import Path
import re

base = Path("examples/libero/pld/configs/reward_model")
files = sorted(base.glob("libero_spatial_task*_residual_sac_*.yaml"))
bad = []
for path in files:
    text = path.read_text(errors="ignore")
    match = re.search(r"init_temperature:\s*([0-9.]+)", text)
    value = match.group(1) if match else None
    if value != "1.0":
        bad.append((path.name, value))

print("count", len(files))
print("bad_temp", bad)
PY
```

Result:

```text
count 30
bad_temp []
```

For reward-model runs, the actor-side reward processor uses:

```yaml
reward:
  async:
    batch_size: 1
```

This is separate from the reward server's model forward batch size. The
Robo-Dopamine and RoboMeter servers still use `FORWARD_BATCH_SIZE=8` /
`--batch-size 8` internally.

## Archive old outputs

Before relaunch, the previous `0.01` outputs were moved out of the standard
paths:

```bash
ARCHIVE_SUFFIX=init_temp_0p01_20260628_archived_20260629_001926

for root in \
  examples/libero/pld/outputs/reward_model/libero_spatial_residual_sac_sparse_300k \
  examples/libero/pld/outputs/reward_model/libero_spatial_residual_sac_robodopamine_pbrs_300k \
  examples/libero/pld/outputs/reward_model/libero_spatial_residual_sac_robometer_pbrs_300k
do
  for task in 04 05 08 09; do
    if [ -d "${root}/task${task}" ]; then
      mv "${root}/task${task}" "${root}/task${task}.${ARCHIVE_SUFFIX}"
    fi
  done
done
```

Confirmed archived directories:

```text
.../libero_spatial_residual_sac_sparse_300k/task04.init_temp_0p01_20260628_archived_20260629_001926
.../libero_spatial_residual_sac_sparse_300k/task05.init_temp_0p01_20260628_archived_20260629_001926
.../libero_spatial_residual_sac_sparse_300k/task08.init_temp_0p01_20260628_archived_20260629_001926
.../libero_spatial_residual_sac_sparse_300k/task09.init_temp_0p01_20260628_archived_20260629_001926
.../libero_spatial_residual_sac_robodopamine_pbrs_300k/task04.init_temp_0p01_20260628_archived_20260629_001926
.../libero_spatial_residual_sac_robodopamine_pbrs_300k/task05.init_temp_0p01_20260628_archived_20260629_001926
.../libero_spatial_residual_sac_robodopamine_pbrs_300k/task08.init_temp_0p01_20260628_archived_20260629_001926
.../libero_spatial_residual_sac_robodopamine_pbrs_300k/task09.init_temp_0p01_20260628_archived_20260629_001926
.../libero_spatial_residual_sac_robometer_pbrs_300k/task04.init_temp_0p01_20260628_archived_20260629_001926
.../libero_spatial_residual_sac_robometer_pbrs_300k/task05.init_temp_0p01_20260628_archived_20260629_001926
.../libero_spatial_residual_sac_robometer_pbrs_300k/task08.init_temp_0p01_20260628_archived_20260629_001926
.../libero_spatial_residual_sac_robometer_pbrs_300k/task09.init_temp_0p01_20260628_archived_20260629_001926
```

## Cleanup before relaunch

The cleanup target was the previous experiment set on each assigned machine:
actor, learner, OpenPI policy server, reward server, reward adapter, async eval
workers, and any GPU-reservation LeRobot process.

The LeRobot reservation process was explicitly allowed to kill:

```bash
pkill -f '[l]erobot/.venv/bin/python3' || true
```

Training and reward tmux sessions were cleaned by exact session name before
relaunch, for example:

```bash
tmux kill-session -t pld_t4_sparse_300k_204 2>/dev/null || true
tmux kill-session -t pld_t4_robodopamine_pbrs_300k_234 2>/dev/null || true
tmux kill-session -t pld_t4_robometer_pbrs_300k_239 2>/dev/null || true
tmux kill-session -t pld_rm_t4_robodopamine_234 2>/dev/null || true
tmux kill-session -t pld_rm_t4_robometer_239 2>/dev/null || true
```

## Launch commands

The launch script is:

```text
examples/libero/pld/tools/launch_residual_sac.sh
```

Important detail: `--eval-policy-port` was set equal to `--policy-port` for
these runs. This makes async eval reuse the same OpenPI reference-policy server
instead of starting another server on the default `8999`, which previously caused
port collisions and startup failures when many groups ran together.

### 204 sparse

GPU rule used here:

| task | actor/policy/eval GPU | learner GPU | policy port | trainer/broadcast |
|---:|---:|---:|---:|---|
| 4 | 1 | 0 | 61704 | 60704 / 60705 |
| 5 | 2 | 3 | 61705 | 60714 / 60715 |
| 8 | 4 | 5 | 61708 | 60724 / 60725 |
| 9 | 6 | 7 | 61709 | 60734 / 60735 |

Task 4 uses the swapped 0/1 pair: actor/policy/eval on GPU 1 and learner on GPU
0. This matched the observed stable GPU placement for this host.

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

bash examples/libero/pld/tools/launch_residual_sac.sh \
  --session pld_t4_sparse_300k_204 \
  --config examples/libero/pld/configs/reward_model/libero_spatial_task4_residual_sac_sparse.yaml \
  --actor-gpu 1 \
  --learner-gpu 0 \
  --policy-gpu 1 \
  --policy-port 61704 \
  --eval-gpu 1 \
  --eval-policy-port 61704 \
  --trainer-port 60704 \
  --broadcast-port 60705 \
  --with-eval

bash examples/libero/pld/tools/launch_residual_sac.sh \
  --session pld_t5_sparse_300k_204 \
  --config examples/libero/pld/configs/reward_model/libero_spatial_task5_residual_sac_sparse.yaml \
  --actor-gpu 2 \
  --learner-gpu 3 \
  --policy-gpu 2 \
  --policy-port 61705 \
  --eval-gpu 2 \
  --eval-policy-port 61705 \
  --trainer-port 60714 \
  --broadcast-port 60715 \
  --with-eval

bash examples/libero/pld/tools/launch_residual_sac.sh \
  --session pld_t8_sparse_300k_204 \
  --config examples/libero/pld/configs/reward_model/libero_spatial_task8_residual_sac_sparse.yaml \
  --actor-gpu 4 \
  --learner-gpu 5 \
  --policy-gpu 4 \
  --policy-port 61708 \
  --eval-gpu 4 \
  --eval-policy-port 61708 \
  --trainer-port 60724 \
  --broadcast-port 60725 \
  --with-eval

bash examples/libero/pld/tools/launch_residual_sac.sh \
  --session pld_t9_sparse_300k_204 \
  --config examples/libero/pld/configs/reward_model/libero_spatial_task9_residual_sac_sparse.yaml \
  --actor-gpu 6 \
  --learner-gpu 7 \
  --policy-gpu 6 \
  --policy-port 61709 \
  --eval-gpu 6 \
  --eval-policy-port 61709 \
  --trainer-port 60734 \
  --broadcast-port 60735 \
  --with-eval
```

### 234 Robo-Dopamine PBRS

Robo-Dopamine reward server settings:

```text
model: /vla/users/niejunnan/assets/Robo-Dopamine-GRM-2.0-4B-Preview
root:  /vla/users/niejunnan/codebase/Robo-Dopamine
mode:  forward
views: image_rgb_0 image_rgb_1, view_mode=two_view_copy_main
goal:  /vla/users/niejunnan/datasets/libero_lerobot
server forward batch size: 8
```

Note: `examples/libero/pld/tools/serve_robodopamine_progress.sh` is a thin
wrapper around `examples/libero/rlpd/tools/serve_robodopamine_progress.sh`.
Therefore `ps` output can show the `rlpd/tools` path even though the command was
started through the PLD wrapper. This is expected.

GPU and port placement:

| task | actor/policy/eval GPU | learner GPU | RM GPU | RM port | policy port | trainer/broadcast |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0 | 1 | 1 | 52004 | 61604 | 60604 / 60605 |
| 5 | 2 | 3 | 3 | 52005 | 61605 | 60614 / 60615 |
| 8 | 4 | 5 | 5 | 52008 | 61608 | 60624 / 60625 |
| 9 | 6 | 7 | 7 | 52009 | 61609 | 60634 / 60635 |

Task 4 was intentionally swapped to actor/policy/eval on GPU 0 and learner/RM on
GPU 1 after the actor became very slow on the original placement.

Start reward servers:

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

tmux new-session -d -s pld_rm_t4_robodopamine_234 \
  "cd /vla/users/niejunnan/codebase/VLA-RL && GPU=1 PORT=52004 FORWARD_BATCH_SIZE=8 OUT_ROOT=/dev/shm/robodopamine_pld_t4_init_temp_1p0_actor_gpu0_20260629_004513 bash examples/libero/pld/tools/serve_robodopamine_progress.sh > examples/libero/pld/outputs/reward_model/launch_logs/pld_rm_t4_robodopamine_234_actor_gpu0_20260629_004513.log 2>&1"

tmux new-session -d -s pld_rm_t5_robodopamine_234 \
  "cd /vla/users/niejunnan/codebase/VLA-RL && GPU=3 PORT=52005 FORWARD_BATCH_SIZE=8 OUT_ROOT=/dev/shm/robodopamine_pld_t5_init_temp_1p0_correct_gpu_20260629_003814 bash examples/libero/pld/tools/serve_robodopamine_progress.sh > examples/libero/pld/outputs/reward_model/launch_logs/pld_rm_t5_robodopamine_234_correct_gpu_20260629_003814.log 2>&1"

tmux new-session -d -s pld_rm_t8_robodopamine_234 \
  "cd /vla/users/niejunnan/codebase/VLA-RL && GPU=5 PORT=52008 FORWARD_BATCH_SIZE=8 OUT_ROOT=/dev/shm/robodopamine_pld_t8_init_temp_1p0_correct_gpu_20260629_003814 bash examples/libero/pld/tools/serve_robodopamine_progress.sh > examples/libero/pld/outputs/reward_model/launch_logs/pld_rm_t8_robodopamine_234_correct_gpu_20260629_003814.log 2>&1"

tmux new-session -d -s pld_rm_t9_robodopamine_234 \
  "cd /vla/users/niejunnan/codebase/VLA-RL && GPU=7 PORT=52009 FORWARD_BATCH_SIZE=8 OUT_ROOT=/dev/shm/robodopamine_pld_t9_init_temp_1p0_correct_gpu_20260629_003814 bash examples/libero/pld/tools/serve_robodopamine_progress.sh > examples/libero/pld/outputs/reward_model/launch_logs/pld_rm_t9_robodopamine_234_correct_gpu_20260629_003814.log 2>&1"
```

Start training:

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

bash examples/libero/pld/tools/launch_residual_sac.sh \
  --session pld_t4_robodopamine_pbrs_300k_234 \
  --config examples/libero/pld/configs/reward_model/libero_spatial_task4_residual_sac_robodopamine_pbrs.yaml \
  --actor-gpu 0 --learner-gpu 1 --policy-gpu 0 --policy-port 61604 \
  --eval-gpu 0 --eval-policy-port 61604 \
  --trainer-port 60604 --broadcast-port 60605 --with-eval

bash examples/libero/pld/tools/launch_residual_sac.sh \
  --session pld_t5_robodopamine_pbrs_300k_234 \
  --config examples/libero/pld/configs/reward_model/libero_spatial_task5_residual_sac_robodopamine_pbrs.yaml \
  --actor-gpu 2 --learner-gpu 3 --policy-gpu 2 --policy-port 61605 \
  --eval-gpu 2 --eval-policy-port 61605 \
  --trainer-port 60614 --broadcast-port 60615 --with-eval

bash examples/libero/pld/tools/launch_residual_sac.sh \
  --session pld_t8_robodopamine_pbrs_300k_234 \
  --config examples/libero/pld/configs/reward_model/libero_spatial_task8_residual_sac_robodopamine_pbrs.yaml \
  --actor-gpu 4 --learner-gpu 5 --policy-gpu 4 --policy-port 61608 \
  --eval-gpu 4 --eval-policy-port 61608 \
  --trainer-port 60624 --broadcast-port 60625 --with-eval

bash examples/libero/pld/tools/launch_residual_sac.sh \
  --session pld_t9_robodopamine_pbrs_300k_234 \
  --config examples/libero/pld/configs/reward_model/libero_spatial_task9_residual_sac_robodopamine_pbrs.yaml \
  --actor-gpu 6 --learner-gpu 7 --policy-gpu 6 --policy-port 61609 \
  --eval-gpu 6 --eval-policy-port 61609 \
  --trainer-port 60634 --broadcast-port 60635 --with-eval
```

### 239 RoboMeter PBRS

RoboMeter reward server settings:

```text
model: /vla/users/niejunnan/assets/Robometer-4B
root:  /vla/users/niejunnan/workspace/robometer
backend: native
query_mode: prefix_per_query
views: image_rgb_0 only, view_mode=first
max_history_frames: 8
server forward batch size: 8
```

Note: `examples/libero/pld/tools/serve_robometer_progress_stack.sh` is a thin
wrapper around `examples/libero/rlpd/tools/serve_robometer_progress_stack.sh`.
Therefore `ps` output can show the `rlpd/tools` path even though the command was
started through the PLD wrapper. This is expected.

GPU and port placement:

| task | actor/policy/eval GPU | learner GPU | RM GPU | RM port | policy port | trainer/broadcast |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 1 | 0 | 0 | 50156 | 61504 | 60504 / 60505 |
| 5 | 2 | 3 | 3 | 50157 | 61505 | 60514 / 60515 |
| 8 | 4 | 5 | 5 | 50160 | 61508 | 60524 / 60525 |
| 9 | 6 | 7 | 7 | 50161 | 61509 | 60534 / 60535 |

Start reward servers:

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

tmux new-session -d -s pld_rm_t4_robometer_239 \
  "cd /vla/users/niejunnan/codebase/VLA-RL && ROBOMETER_ROOT=/vla/users/niejunnan/workspace/robometer ROBOMETER_PYTHON_CMD=/vla/users/niejunnan/workspace/robometer/.venv/bin/python BACKEND=native GPU=0 ADAPTER_PORT=50156 FORWARD_BATCH_SIZE=8 QUERY_MODE=prefix_per_query VIEW_MODE=first MAX_HISTORY_FRAMES=8 bash examples/libero/pld/tools/serve_robometer_progress_stack.sh > examples/libero/pld/outputs/reward_model/launch_logs/pld_rm_t4_robometer_239_correct_gpu_20260629_003814.log 2>&1"

tmux new-session -d -s pld_rm_t5_robometer_239 \
  "cd /vla/users/niejunnan/codebase/VLA-RL && ROBOMETER_ROOT=/vla/users/niejunnan/workspace/robometer ROBOMETER_PYTHON_CMD=/vla/users/niejunnan/workspace/robometer/.venv/bin/python BACKEND=native GPU=3 ADAPTER_PORT=50157 FORWARD_BATCH_SIZE=8 QUERY_MODE=prefix_per_query VIEW_MODE=first MAX_HISTORY_FRAMES=8 bash examples/libero/pld/tools/serve_robometer_progress_stack.sh > examples/libero/pld/outputs/reward_model/launch_logs/pld_rm_t5_robometer_239_correct_gpu_20260629_003814.log 2>&1"

tmux new-session -d -s pld_rm_t8_robometer_239 \
  "cd /vla/users/niejunnan/codebase/VLA-RL && ROBOMETER_ROOT=/vla/users/niejunnan/workspace/robometer ROBOMETER_PYTHON_CMD=/vla/users/niejunnan/workspace/robometer/.venv/bin/python BACKEND=native GPU=5 ADAPTER_PORT=50160 FORWARD_BATCH_SIZE=8 QUERY_MODE=prefix_per_query VIEW_MODE=first MAX_HISTORY_FRAMES=8 bash examples/libero/pld/tools/serve_robometer_progress_stack.sh > examples/libero/pld/outputs/reward_model/launch_logs/pld_rm_t8_robometer_239_correct_gpu_20260629_003814.log 2>&1"

tmux new-session -d -s pld_rm_t9_robometer_239 \
  "cd /vla/users/niejunnan/codebase/VLA-RL && ROBOMETER_ROOT=/vla/users/niejunnan/workspace/robometer ROBOMETER_PYTHON_CMD=/vla/users/niejunnan/workspace/robometer/.venv/bin/python BACKEND=native GPU=7 ADAPTER_PORT=50161 FORWARD_BATCH_SIZE=8 QUERY_MODE=prefix_per_query VIEW_MODE=first MAX_HISTORY_FRAMES=8 bash examples/libero/pld/tools/serve_robometer_progress_stack.sh > examples/libero/pld/outputs/reward_model/launch_logs/pld_rm_t9_robometer_239_correct_gpu_20260629_003814.log 2>&1"
```

Start training:

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

bash examples/libero/pld/tools/launch_residual_sac.sh \
  --session pld_t4_robometer_pbrs_300k_239 \
  --config examples/libero/pld/configs/reward_model/libero_spatial_task4_residual_sac_robometer_pbrs.yaml \
  --actor-gpu 1 --learner-gpu 0 --policy-gpu 1 --policy-port 61504 \
  --eval-gpu 1 --eval-policy-port 61504 \
  --trainer-port 60504 --broadcast-port 60505 --with-eval

bash examples/libero/pld/tools/launch_residual_sac.sh \
  --session pld_t5_robometer_pbrs_300k_239 \
  --config examples/libero/pld/configs/reward_model/libero_spatial_task5_residual_sac_robometer_pbrs.yaml \
  --actor-gpu 2 --learner-gpu 3 --policy-gpu 2 --policy-port 61505 \
  --eval-gpu 2 --eval-policy-port 61505 \
  --trainer-port 60514 --broadcast-port 60515 --with-eval

bash examples/libero/pld/tools/launch_residual_sac.sh \
  --session pld_t8_robometer_pbrs_300k_239 \
  --config examples/libero/pld/configs/reward_model/libero_spatial_task8_residual_sac_robometer_pbrs.yaml \
  --actor-gpu 4 --learner-gpu 5 --policy-gpu 4 --policy-port 61508 \
  --eval-gpu 4 --eval-policy-port 61508 \
  --trainer-port 60524 --broadcast-port 60525 --with-eval

bash examples/libero/pld/tools/launch_residual_sac.sh \
  --session pld_t9_robometer_pbrs_300k_239 \
  --config examples/libero/pld/configs/reward_model/libero_spatial_task9_residual_sac_robometer_pbrs.yaml \
  --actor-gpu 6 --learner-gpu 7 --policy-gpu 6 --policy-port 61509 \
  --eval-gpu 6 --eval-policy-port 61509 \
  --trainer-port 60534 --broadcast-port 60535 --with-eval
```

## Speed issues and fixes

### 1. `init_temperature=0.01` made the early residual policy too cold

Symptom:

- New runs with `0.01` started with very low entropy temperature.
- Sparse and PBRS groups showed unstable early eval behavior compared with
  historical successful residual SAC runs.

Fix:

- Set all 30 formal reward_model YAMLs back to `init_temperature: 1.0`.
- Do not override this from CLI. The YAML is the source of truth.

Verification:

- Latest runs show temperature decaying smoothly from `1.0`, for example around
  50k steps:
  - Robo-Dopamine: `train/temperature` around `0.16`.
  - RoboMeter: `train/temperature` around `0.15`.
  - Sparse runs are further along, so temperature is lower.

### 2. Async eval policy port collision

Symptom:

- When many groups are launched, the default async eval policy port `8999` can
  collide.
- This caused async eval startup issues.

Fix:

- Launch with `--eval-policy-port` equal to `--policy-port`.
- `launch_residual_sac.sh` then reuses the main OpenPI reference-policy server
  for async eval and does not start a second policy server.

Example:

```bash
--policy-port 61504 \
--eval-policy-port 61504
```

### 3. GPU placement can dominate actor speed

Symptom:

- Some GPU pairings on these machines can make LIBERO/OpenPI actor execution very
  slow.
- The worst observed case was task4 Robo-Dopamine before swapping: actor was
  around sub-1 env/s to a few env/s, with very large `env_step_chunk_sec`.

Fix:

- Swap the 0/1 pair for task4 on 234:
  - actor/policy/eval on GPU 0
  - learner/RM on GPU 1
- Use the same principle on 204 sparse task4:
  - actor/policy/eval on GPU 1
  - learner on GPU 0

Rule of thumb:

- If actor is below about `10 env/s` and reward is not failing, try swapping the
  actor/policy/eval side with the learner/RM side for that task.

### 4. RoboMeter official HTTP/multipart path was too slow

Symptom:

- The old RoboMeter stack could be much slower because the adapter forwarded
  repeated image prefixes through an official eval-server style HTTP/multipart
  `.npy` path.
- This made reward latency high and could backpressure actor.

Fix:

- Run RoboMeter in the native backend:

```text
BACKEND=native
QUERY_MODE=prefix_per_query
VIEW_MODE=first
MAX_HISTORY_FRAMES=8
FORWARD_BATCH_SIZE=8
```

Why this is semantically OK:

- `prefix_per_query` builds each query from frames with index `<= query_index`.
- It does not use future frames such as `S58...S64` to predict progress at `S57`.
- For LIBERO, RoboMeter is run with the main camera only: `image_rgb_0`.

### 5. Reward processor batch size should not wait for large batches

Symptom:

- Actor-side reward relabeling can become backpressured if the reward processor
  waits for larger batches or if per-query latency is high.

Fix:

- Set reward-model YAMLs to actor-side:

```yaml
reward:
  async:
    batch_size: 1
```

This does not disable model-side batching. It only prevents the actor-side reward
processor from waiting for a multi-chunk batch before committing progress.

Observed final behavior:

- RoboMeter pending stayed around `0-2`.
- Robo-Dopamine pending stayed near the configured cap `126-128`, but actor speed
  remained above `12 env/s` and `failed=0`.

### 6. LeRobot reservation processes can be killed

Symptom:

- Periodic `lerobot/.venv/bin/python3` GPU reservation/stress processes can hold
  GPU memory and block experiments.

Fix:

```bash
pkill -f '[l]erobot/.venv/bin/python3' || true
```

After cleanup, no `lerobot/.venv/bin/python3` GPU process was found on 204, 234,
or 239.

## Monitoring commands

The generic monitor may label these VLA-RL PLD processes as unknown GPU
processes, so the reliable check is to read JSONL logs directly.

Important files per task:

```text
actor_metrics.jsonl
metrics.jsonl
progress_events.jsonl
eval_queue.jsonl
eval_summary.jsonl
```

Example manual checks:

```bash
# Check current tmux sessions.
tmux ls | grep -E 'pld|rm|robometer|robodopamine|sparse'

# Check LeRobot reservation processes.
pgrep -af '[l]erobot/.venv/bin/python3' || true

# Check latest actor metrics for a task.
tail -n 1 examples/libero/pld/outputs/reward_model/libero_spatial_residual_sac_robometer_pbrs_300k/task04/actor_metrics.jsonl

# Check latest learner metrics for a task.
tail -n 1 examples/libero/pld/outputs/reward_model/libero_spatial_residual_sac_robometer_pbrs_300k/task04/metrics.jsonl
```

The 60-second speed snapshots in this note were computed by comparing the last
`env_steps` and `update_steps` in JSONL logs before and after a 60-second sleep.
This avoids relying on noisy process lists or on a single logged speed field.

Minimal version of the sampler:

```bash
python3 - <<'PY'
import json, pathlib, time

base = pathlib.Path("examples/libero/pld/outputs/reward_model/libero_spatial_residual_sac_robometer_pbrs_300k")
tasks = [4, 5, 8, 9]

def last_json(path):
    if not path.exists():
        return {}
    for line in reversed(path.read_text(errors="ignore").splitlines()[-50:]):
        try:
            return json.loads(line)
        except Exception:
            pass
    return {}

def snap():
    out = {}
    for task in tasks:
        d = base / f"task{task:02d}"
        actor = last_json(d / "actor_metrics.jsonl")
        learner = last_json(d / "metrics.jsonl")
        out[task] = {
            "env_steps": actor.get("env_steps", 0),
            "update_steps": learner.get("update_steps", learner.get("learner/update_steps", 0)),
            "replay_size": learner.get("replay_size", learner.get("learner/replay_size")),
            "temperature": learner.get("train/temperature", learner.get("learner/temperature")),
            "reward_processor": actor.get("reward_processor", {}),
        }
    return out

s0 = snap()
t0 = time.time()
time.sleep(60)
s1 = snap()
t1 = time.time()

for task in tasks:
    env_s = (s1[task]["env_steps"] - s0[task]["env_steps"]) / (t1 - t0)
    upd_s = (s1[task]["update_steps"] - s0[task]["update_steps"]) / (t1 - t0)
    print(task, "env/s", round(env_s, 2), "update/s", round(upd_s, 2), s1[task])
PY
```

## Final early verification snapshot

Sampling window:

```text
2026-06-29 02:03:55 - 02:04:57 CST
```

### 204 sparse

| task | env steps | actor delta speed | learner update speed | replay | temp | eval latest | reward failed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 78,789 | 12.85 env/s | 1.90 up/s | 15,838 | 0.037 | 0.70 | 0 |
| 5 | 93,308 | 15.95 env/s | 2.61 up/s | 18,719 | 0.013 | 0.48 | 0 |
| 8 | 94,840 | 15.85 env/s | 2.56 up/s | 19,103 | 0.012 | 0.76 | 0 |
| 9 | 92,587 | 14.92 env/s | 2.66 up/s | 18,668 | 0.012 | 0.84 | 0 |

### 234 Robo-Dopamine PBRS

| task | env steps | actor delta speed | learner update speed | replay | temp | eval latest | reward pending | reward failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 48,295 | 12.98 env/s | 1.83 up/s | 9,594 | 0.165 | 0.58 | 128 | 0 |
| 5 | 49,229 | 12.98 env/s | 1.83 up/s | 9,742 | 0.163 | 0.38 | 127 | 0 |
| 8 | 49,389 | 13.40 env/s | 1.85 up/s | 9,836 | 0.163 | 0.66 | 126 | 0 |
| 9 | 49,216 | 13.15 env/s | 1.81 up/s | 9,792 | 0.163 | 0.94 | 127 | 0 |

### 239 RoboMeter PBRS

| task | env steps | actor delta speed | learner update speed | replay | temp | eval latest | reward pending | reward failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 51,031 | 14.03 env/s | 2.06 up/s | 10,261 | 0.156 | 0.70 | 1 | 0 |
| 5 | 52,343 | 14.81 env/s | 2.10 up/s | 10,492 | 0.152 | 0.50 | 1 | 0 |
| 8 | 51,214 | 15.73 env/s | 2.08 up/s | 10,322 | 0.152 | 0.82 | 2 | 0 |
| 9 | 52,127 | 14.73 env/s | 2.05 up/s | 10,503 | 0.150 | 0.86 | 1 | 0 |

## Current judgment

The rerun successfully passed the early link check:

- All 12 groups launched from the corrected `init_temperature=1.0` YAMLs.
- Old `0.01` outputs were archived and standard output paths now point to the
  current mainline rerun.
- Actor speeds are above the practical target of `10 env/s`.
- Learners are updating normally.
- Reward failures are `0`.
- Async eval is producing results.
- RoboMeter is on the native, single-view, prefix-per-query path and no future
  frames are used for earlier progress queries.

These runs can be left to finish the full `300k` budget.
