# LIBERO Spatial RLT Stage 2 Reward-Model Launch Guide

This document records the launch convention for LIBERO spatial RLT stage 2
experiments with sparse rewards and Robo-Dopamine dense rewards.

Default convention:

- A sparse run does not start or use a reward-model service.
- Every reward-model run owns one dedicated Robo-Dopamine progress server.
- Do not share a reward server across multiple reward-model runs unless the
  experiment name explicitly says it is a shared-server throughput test.
- Reward server ports are fixed in the reward-model YAML files for traceability.
  Do not override `reward.remote.url` from the command line for standard runs.

## Paths

Run all commands from the VLA-RL checkout:

```bash
cd /vla/users/niejunnan/codebase/VLA-RL
```

Reference policy checkpoint used by the current benchmark:

```text
/vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000_pytorch
```

Robo-Dopamine model used by the current benchmark:

```text
/vla/users/niejunnan/assets/Robo-Dopamine-GRM-2.0-4B-Preview
```

Goal dataset root:

```text
/vla/users/niejunnan/datasets/libero_lerobot
```

## Resource Policy

The port plan below is global for the host: it is safe even if all 10 LIBERO
spatial tasks are launched on the same machine at the same time. No trainer,
broadcast, policy, eval-policy, or reward-server port is reused across tasks or
reward settings.

GPU capacity is a separate constraint. Robo-Dopamine-GRM-2.0-4B-Preview usually
consumes about 80GB+ GPU memory. On an 8-GPU node, the recommended layout is to
run at most two task pairs at once:

```text
even task slot:
  sparse actor/policy/eval -> GPU0
  sparse learner           -> GPU1
  PBRS actor/policy/eval   -> GPU2
  PBRS learner/reward      -> GPU3

odd task slot:
  sparse actor/policy/eval -> GPU4
  sparse learner           -> GPU5
  PBRS actor/policy/eval   -> GPU6
  PBRS learner/reward      -> GPU7
```

Run LIBERO spatial tasks in batches:

```text
batch 1: task0 + task1
batch 2: task2 + task3
batch 3: task4 + task5
batch 4: task6 + task7
batch 5: task8 + task9
```

The command examples below follow this two-task-per-node GPU convention, but the
ports remain unique across all 10 tasks. If a larger node can actually host all
10 reward servers, keep the same ports and only adjust GPU ids.

Always pass `--run-dir` to `launch_rlt.sh`. The launcher has a legacy default
run directory, so omitting `--run-dir` can make concurrent jobs write to the same
output directory even when ports are unique.

## Per-Task Port Matrix

| task | sparse trainer/broadcast | sparse policy/eval-policy | PBRS trainer/broadcast | PBRS policy/eval-policy | PBRS reward server |
| --- | --- | --- | --- | --- | --- |
| 0 | `61000/61001` | `41000/41001` | `62000/62001` | `42000/42001` | `50052` |
| 1 | `61010/61011` | `41010/41011` | `62010/62011` | `42010/42011` | `50053` |
| 2 | `61020/61021` | `41020/41021` | `62020/62021` | `42020/42021` | `50054` |
| 3 | `61030/61031` | `41030/41031` | `62030/62031` | `42030/42031` | `50055` |
| 4 | `61040/61041` | `41040/41041` | `62040/62041` | `42040/42041` | `50056` |
| 5 | `61050/61051` | `41050/41051` | `62050/62051` | `42050/42051` | `50057` |
| 6 | `61060/61061` | `41060/41061` | `62060/62061` | `42060/42061` | `50058` |
| 7 | `61070/61071` | `41070/41071` | `62070/62071` | `42070/42071` | `50059` |
| 8 | `61080/61081` | `41080/41081` | `62080/62081` | `42080/42081` | `50060` |
| 9 | `61090/61091` | `41090/41091` | `62090/62091` | `42090/42091` | `50061` |

The full host-level port set is:

```text
41000 41001 41010 41011 41020 41021 41030 41031 41040 41041
41050 41051 41060 41061 41070 41071 41080 41081 41090 41091
42000 42001 42010 42011 42020 42021 42030 42031 42040 42041
42050 42051 42060 42061 42070 42071 42080 42081 42090 42091
50052 50053 50054 50055 50056 50057 50058 50059 50060 50061
61000 61001 61010 61011 61020 61021 61030 61031 61040 61041
61050 61051 61060 61061 61070 61071 61080 61081 61090 61091
62000 62001 62010 62011 62020 62021 62030 62031 62040 62041
62050 62051 62060 62061 62070 62071 62080 62081 62090 62091
```

## Run Directory Rule

Sparse:

```text
/vla/users/niejunnan/codebase/VLA-RL/outputs/libero_spatial_task{TASK}_self_conditioned_prefix_512_chunk5_sparse_rlt
```

PBRS:

```text
/vla/users/niejunnan/codebase/VLA-RL/outputs/libero_spatial_task{TASK}_self_conditioned_prefix_512_chunk5_reward_model_pbrs_rlt
```

## Start A Dedicated Robo-Dopamine Server

Use one server per reward-model run. Change only `--port`, GPU, and tmux session
according to the task matrix.

Task 0, reward server on GPU3:

```bash
tmux new-session -d -s rlt_spatial_task0_robodopamine_50052 -n reward \
  "bash -lc 'cd /vla/users/niejunnan/codebase/VLA-RL && CUDA_VISIBLE_DEVICES=3 /vla/users/niejunnan/envs/robo-dopamine/bin/python scripts/serve_robodopamine_progress_http.py --robodopamine-root /vla/users/niejunnan/codebase/Robo-Dopamine --model-path /vla/users/niejunnan/assets/Robo-Dopamine-GRM-2.0-4B-Preview --host 127.0.0.1 --port 50052 --cuda-visible-devices 3 --image-keys image_rgb_0 image_rgb_1 --view-mode two_view_copy_main --image-preprocess none --goal-image-preprocess none --batch-size 8 --goal-dataset /vla/users/niejunnan/datasets/libero_lerobot'"
```

Task 1, reward server on GPU7:

```bash
tmux new-session -d -s rlt_spatial_task1_robodopamine_50053 -n reward \
  "bash -lc 'cd /vla/users/niejunnan/codebase/VLA-RL && CUDA_VISIBLE_DEVICES=7 /vla/users/niejunnan/envs/robo-dopamine/bin/python scripts/serve_robodopamine_progress_http.py --robodopamine-root /vla/users/niejunnan/codebase/Robo-Dopamine --model-path /vla/users/niejunnan/assets/Robo-Dopamine-GRM-2.0-4B-Preview --host 127.0.0.1 --port 50053 --cuda-visible-devices 7 --image-keys image_rgb_0 image_rgb_1 --view-mode two_view_copy_main --image-preprocess none --goal-image-preprocess none --batch-size 8 --goal-dataset /vla/users/niejunnan/datasets/libero_lerobot'"
```

Task 2, reward server on GPU3:

```bash
tmux new-session -d -s rlt_spatial_task2_robodopamine_50054 -n reward \
  "bash -lc 'cd /vla/users/niejunnan/codebase/VLA-RL && CUDA_VISIBLE_DEVICES=3 /vla/users/niejunnan/envs/robo-dopamine/bin/python scripts/serve_robodopamine_progress_http.py --robodopamine-root /vla/users/niejunnan/codebase/Robo-Dopamine --model-path /vla/users/niejunnan/assets/Robo-Dopamine-GRM-2.0-4B-Preview --host 127.0.0.1 --port 50054 --cuda-visible-devices 3 --image-keys image_rgb_0 image_rgb_1 --view-mode two_view_copy_main --image-preprocess none --goal-image-preprocess none --batch-size 8 --goal-dataset /vla/users/niejunnan/datasets/libero_lerobot'"
```

Task 3, reward server on GPU7:

```bash
tmux new-session -d -s rlt_spatial_task3_robodopamine_50055 -n reward \
  "bash -lc 'cd /vla/users/niejunnan/codebase/VLA-RL && CUDA_VISIBLE_DEVICES=7 /vla/users/niejunnan/envs/robo-dopamine/bin/python scripts/serve_robodopamine_progress_http.py --robodopamine-root /vla/users/niejunnan/codebase/Robo-Dopamine --model-path /vla/users/niejunnan/assets/Robo-Dopamine-GRM-2.0-4B-Preview --host 127.0.0.1 --port 50055 --cuda-visible-devices 7 --image-keys image_rgb_0 image_rgb_1 --view-mode two_view_copy_main --image-preprocess none --goal-image-preprocess none --batch-size 8 --goal-dataset /vla/users/niejunnan/datasets/libero_lerobot'"
```

Task 4, reward server on GPU3:

```bash
tmux new-session -d -s rlt_spatial_task4_robodopamine_50056 -n reward \
  "bash -lc 'cd /vla/users/niejunnan/codebase/VLA-RL && CUDA_VISIBLE_DEVICES=3 /vla/users/niejunnan/envs/robo-dopamine/bin/python scripts/serve_robodopamine_progress_http.py --robodopamine-root /vla/users/niejunnan/codebase/Robo-Dopamine --model-path /vla/users/niejunnan/assets/Robo-Dopamine-GRM-2.0-4B-Preview --host 127.0.0.1 --port 50056 --cuda-visible-devices 3 --image-keys image_rgb_0 image_rgb_1 --view-mode two_view_copy_main --image-preprocess none --goal-image-preprocess none --batch-size 8 --goal-dataset /vla/users/niejunnan/datasets/libero_lerobot'"
```

Task 5, reward server on GPU7:

```bash
tmux new-session -d -s rlt_spatial_task5_robodopamine_50057 -n reward \
  "bash -lc 'cd /vla/users/niejunnan/codebase/VLA-RL && CUDA_VISIBLE_DEVICES=7 /vla/users/niejunnan/envs/robo-dopamine/bin/python scripts/serve_robodopamine_progress_http.py --robodopamine-root /vla/users/niejunnan/codebase/Robo-Dopamine --model-path /vla/users/niejunnan/assets/Robo-Dopamine-GRM-2.0-4B-Preview --host 127.0.0.1 --port 50057 --cuda-visible-devices 7 --image-keys image_rgb_0 image_rgb_1 --view-mode two_view_copy_main --image-preprocess none --goal-image-preprocess none --batch-size 8 --goal-dataset /vla/users/niejunnan/datasets/libero_lerobot'"
```

Task 6, reward server on GPU3:

```bash
tmux new-session -d -s rlt_spatial_task6_robodopamine_50058 -n reward \
  "bash -lc 'cd /vla/users/niejunnan/codebase/VLA-RL && CUDA_VISIBLE_DEVICES=3 /vla/users/niejunnan/envs/robo-dopamine/bin/python scripts/serve_robodopamine_progress_http.py --robodopamine-root /vla/users/niejunnan/codebase/Robo-Dopamine --model-path /vla/users/niejunnan/assets/Robo-Dopamine-GRM-2.0-4B-Preview --host 127.0.0.1 --port 50058 --cuda-visible-devices 3 --image-keys image_rgb_0 image_rgb_1 --view-mode two_view_copy_main --image-preprocess none --goal-image-preprocess none --batch-size 8 --goal-dataset /vla/users/niejunnan/datasets/libero_lerobot'"
```

Task 7, reward server on GPU7:

```bash
tmux new-session -d -s rlt_spatial_task7_robodopamine_50059 -n reward \
  "bash -lc 'cd /vla/users/niejunnan/codebase/VLA-RL && CUDA_VISIBLE_DEVICES=7 /vla/users/niejunnan/envs/robo-dopamine/bin/python scripts/serve_robodopamine_progress_http.py --robodopamine-root /vla/users/niejunnan/codebase/Robo-Dopamine --model-path /vla/users/niejunnan/assets/Robo-Dopamine-GRM-2.0-4B-Preview --host 127.0.0.1 --port 50059 --cuda-visible-devices 7 --image-keys image_rgb_0 image_rgb_1 --view-mode two_view_copy_main --image-preprocess none --goal-image-preprocess none --batch-size 8 --goal-dataset /vla/users/niejunnan/datasets/libero_lerobot'"
```

Task 8, reward server on GPU3:

```bash
tmux new-session -d -s rlt_spatial_task8_robodopamine_50060 -n reward \
  "bash -lc 'cd /vla/users/niejunnan/codebase/VLA-RL && CUDA_VISIBLE_DEVICES=3 /vla/users/niejunnan/envs/robo-dopamine/bin/python scripts/serve_robodopamine_progress_http.py --robodopamine-root /vla/users/niejunnan/codebase/Robo-Dopamine --model-path /vla/users/niejunnan/assets/Robo-Dopamine-GRM-2.0-4B-Preview --host 127.0.0.1 --port 50060 --cuda-visible-devices 3 --image-keys image_rgb_0 image_rgb_1 --view-mode two_view_copy_main --image-preprocess none --goal-image-preprocess none --batch-size 8 --goal-dataset /vla/users/niejunnan/datasets/libero_lerobot'"
```

Task 9, reward server on GPU7:

```bash
tmux new-session -d -s rlt_spatial_task9_robodopamine_50061 -n reward \
  "bash -lc 'cd /vla/users/niejunnan/codebase/VLA-RL && CUDA_VISIBLE_DEVICES=7 /vla/users/niejunnan/envs/robo-dopamine/bin/python scripts/serve_robodopamine_progress_http.py --robodopamine-root /vla/users/niejunnan/codebase/Robo-Dopamine --model-path /vla/users/niejunnan/assets/Robo-Dopamine-GRM-2.0-4B-Preview --host 127.0.0.1 --port 50061 --cuda-visible-devices 7 --image-keys image_rgb_0 image_rgb_1 --view-mode two_view_copy_main --image-preprocess none --goal-image-preprocess none --batch-size 8 --goal-dataset /vla/users/niejunnan/datasets/libero_lerobot'"
```

After starting a server, wait for its port before launching PBRS training:

```bash
/vla/users/niejunnan/envs/serl_torch/bin/python \
  examples/libero/rlt/tools/wait_for_tcp.py \
  --host 127.0.0.1 \
  --ports 50052 \
  --timeout-sec 900
```

Replace `50052` with the task-specific reward server port.

## Sparse Stage 2 Launch Template

Sparse runs do not need a reward server.

```bash
bash examples/libero/rlt/tools/launch_rlt.sh \
  --config examples/libero/rlt/configs/reward_model/libero_spatial_task0_self_cond_512_stage2.yaml \
  --session rlt_spatial_task0_sparse \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --policy-gpu 0 \
  --with-eval \
  --eval-gpu 0 \
  --trainer-port 61000 \
  --broadcast-port 61001 \
  --policy-port 41000 \
  --eval-policy-port 41001 \
  --run-dir /vla/users/niejunnan/codebase/VLA-RL/outputs/libero_spatial_task0_self_conditioned_prefix_512_chunk5_sparse_rlt \
  --policy-checkpoint /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000_pytorch
```

Change task id, session name, GPUs, and ports according to the matrix.

## PBRS Stage 2 Launch Template

Start the dedicated reward server first, then launch PBRS. The PBRS YAML already
contains the task-specific `reward.remote.url`.

```bash
bash examples/libero/rlt/tools/launch_rlt.sh \
  --config examples/libero/rlt/configs/reward_model/libero_spatial_task0_self_cond_512_reward_model_pbrs_stage2.yaml \
  --session rlt_spatial_task0_pbrs \
  --actor-gpu 2 \
  --learner-gpu 3 \
  --policy-gpu 2 \
  --with-eval \
  --eval-gpu 2 \
  --trainer-port 62000 \
  --broadcast-port 62001 \
  --policy-port 42000 \
  --eval-policy-port 42001 \
  --run-dir /vla/users/niejunnan/codebase/VLA-RL/outputs/libero_spatial_task0_self_conditioned_prefix_512_chunk5_reward_model_pbrs_rlt \
  --policy-checkpoint /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000_pytorch
```

## Task Launch Matrix

Use the two templates above with the following concrete values.

| task | sparse config | sparse session | sparse GPUs | sparse ports | PBRS config | PBRS session | PBRS GPUs | PBRS ports | reward session |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `libero_spatial_task0_self_cond_512_stage2.yaml` | `rlt_spatial_task0_sparse` | actor/policy/eval `0`, learner `1` | trainer/broadcast `61000/61001`, policy/eval `41000/41001` | `libero_spatial_task0_self_cond_512_reward_model_pbrs_stage2.yaml` | `rlt_spatial_task0_pbrs` | actor/policy/eval `2`, learner/reward `3` | trainer/broadcast `62000/62001`, policy/eval `42000/42001` | `rlt_spatial_task0_robodopamine_50052` |
| 1 | `libero_spatial_task1_self_cond_512_stage2.yaml` | `rlt_spatial_task1_sparse` | actor/policy/eval `4`, learner `5` | trainer/broadcast `61010/61011`, policy/eval `41010/41011` | `libero_spatial_task1_self_cond_512_reward_model_pbrs_stage2.yaml` | `rlt_spatial_task1_pbrs` | actor/policy/eval `6`, learner/reward `7` | trainer/broadcast `62010/62011`, policy/eval `42010/42011` | `rlt_spatial_task1_robodopamine_50053` |
| 2 | `libero_spatial_task2_self_cond_512_stage2.yaml` | `rlt_spatial_task2_sparse` | actor/policy/eval `0`, learner `1` | trainer/broadcast `61020/61021`, policy/eval `41020/41021` | `libero_spatial_task2_self_cond_512_reward_model_pbrs_stage2.yaml` | `rlt_spatial_task2_pbrs` | actor/policy/eval `2`, learner/reward `3` | trainer/broadcast `62020/62021`, policy/eval `42020/42021` | `rlt_spatial_task2_robodopamine_50054` |
| 3 | `libero_spatial_task3_self_cond_512_stage2.yaml` | `rlt_spatial_task3_sparse` | actor/policy/eval `4`, learner `5` | trainer/broadcast `61030/61031`, policy/eval `41030/41031` | `libero_spatial_task3_self_cond_512_reward_model_pbrs_stage2.yaml` | `rlt_spatial_task3_pbrs` | actor/policy/eval `6`, learner/reward `7` | trainer/broadcast `62030/62031`, policy/eval `42030/42031` | `rlt_spatial_task3_robodopamine_50055` |
| 4 | `libero_spatial_task4_self_cond_512_stage2.yaml` | `rlt_spatial_task4_sparse` | actor/policy/eval `0`, learner `1` | trainer/broadcast `61040/61041`, policy/eval `41040/41041` | `libero_spatial_task4_self_cond_512_reward_model_pbrs_stage2.yaml` | `rlt_spatial_task4_pbrs` | actor/policy/eval `2`, learner/reward `3` | trainer/broadcast `62040/62041`, policy/eval `42040/42041` | `rlt_spatial_task4_robodopamine_50056` |
| 5 | `libero_spatial_task5_self_cond_512_stage2.yaml` | `rlt_spatial_task5_sparse` | actor/policy/eval `4`, learner `5` | trainer/broadcast `61050/61051`, policy/eval `41050/41051` | `libero_spatial_task5_self_cond_512_reward_model_pbrs_stage2.yaml` | `rlt_spatial_task5_pbrs` | actor/policy/eval `6`, learner/reward `7` | trainer/broadcast `62050/62051`, policy/eval `42050/42051` | `rlt_spatial_task5_robodopamine_50057` |
| 6 | `libero_spatial_task6_self_cond_512_stage2.yaml` | `rlt_spatial_task6_sparse` | actor/policy/eval `0`, learner `1` | trainer/broadcast `61060/61061`, policy/eval `41060/41061` | `libero_spatial_task6_self_cond_512_reward_model_pbrs_stage2.yaml` | `rlt_spatial_task6_pbrs` | actor/policy/eval `2`, learner/reward `3` | trainer/broadcast `62060/62061`, policy/eval `42060/42061` | `rlt_spatial_task6_robodopamine_50058` |
| 7 | `libero_spatial_task7_self_cond_512_stage2.yaml` | `rlt_spatial_task7_sparse` | actor/policy/eval `4`, learner `5` | trainer/broadcast `61070/61071`, policy/eval `41070/41071` | `libero_spatial_task7_self_cond_512_reward_model_pbrs_stage2.yaml` | `rlt_spatial_task7_pbrs` | actor/policy/eval `6`, learner/reward `7` | trainer/broadcast `62070/62071`, policy/eval `42070/42071` | `rlt_spatial_task7_robodopamine_50059` |
| 8 | `libero_spatial_task8_self_cond_512_stage2.yaml` | `rlt_spatial_task8_sparse` | actor/policy/eval `0`, learner `1` | trainer/broadcast `61080/61081`, policy/eval `41080/41081` | `libero_spatial_task8_self_cond_512_reward_model_pbrs_stage2.yaml` | `rlt_spatial_task8_pbrs` | actor/policy/eval `2`, learner/reward `3` | trainer/broadcast `62080/62081`, policy/eval `42080/42081` | `rlt_spatial_task8_robodopamine_50060` |
| 9 | `libero_spatial_task9_self_cond_512_stage2.yaml` | `rlt_spatial_task9_sparse` | actor/policy/eval `4`, learner `5` | trainer/broadcast `61090/61091`, policy/eval `41090/41091` | `libero_spatial_task9_self_cond_512_reward_model_pbrs_stage2.yaml` | `rlt_spatial_task9_pbrs` | actor/policy/eval `6`, learner/reward `7` | trainer/broadcast `62090/62091`, policy/eval `42090/42091` | `rlt_spatial_task9_robodopamine_50061` |

## RoboMeter Stage 2 Reward Model

RoboMeter is exposed to RLT through the same `remote_progress` contract as
Robo-Dopamine: the trainer calls `predict_progress(request)` and receives
absolute progress values. The RLT reward code does not call RoboMeter directly.
Use a two-process reward stack per RoboMeter run:

1. RoboMeter official eval server, running in the RoboMeter environment.
2. VLA-RL RoboMeter adapter, running in the VLA-RL environment and exposing
   `predict_progress` on the port recorded in the YAML.

RoboMeter does not use an expert goal image. The adapter sends the task prompt
and a server-side cached trajectory prefix to RoboMeter. For LIBERO RLT the
default view mode is `average_two`: `image_rgb_0` and `image_rgb_1` are evaluated
as separate videos and their progress values are averaged.

RoboMeter adapter ports use `50152..50161` so they do not conflict with the
Robo-Dopamine ports `50052..50061`. The suggested RoboMeter eval-server ports
are `8410..8419`.

| task | RoboMeter YAML | eval server port | adapter port |
| --- | --- | --- | --- |
| 0 | `libero_spatial_task0_self_cond_512_reward_model_robometer_pbrs_stage2.yaml` | `8410` | `50152` |
| 1 | `libero_spatial_task1_self_cond_512_reward_model_robometer_pbrs_stage2.yaml` | `8411` | `50153` |
| 2 | `libero_spatial_task2_self_cond_512_reward_model_robometer_pbrs_stage2.yaml` | `8412` | `50154` |
| 3 | `libero_spatial_task3_self_cond_512_reward_model_robometer_pbrs_stage2.yaml` | `8413` | `50155` |
| 4 | `libero_spatial_task4_self_cond_512_reward_model_robometer_pbrs_stage2.yaml` | `8414` | `50156` |
| 5 | `libero_spatial_task5_self_cond_512_reward_model_robometer_pbrs_stage2.yaml` | `8415` | `50157` |
| 6 | `libero_spatial_task6_self_cond_512_reward_model_robometer_pbrs_stage2.yaml` | `8416` | `50158` |
| 7 | `libero_spatial_task7_self_cond_512_reward_model_robometer_pbrs_stage2.yaml` | `8417` | `50159` |
| 8 | `libero_spatial_task8_self_cond_512_reward_model_robometer_pbrs_stage2.yaml` | `8418` | `50160` |
| 9 | `libero_spatial_task9_self_cond_512_reward_model_robometer_pbrs_stage2.yaml` | `8419` | `50161` |

Task 0 example, RoboMeter eval server on GPU3:

```bash
tmux new-session -d -s rlt_spatial_task0_robometer_eval_8410 -n robometer_eval \
  "bash -lc 'cd /vla/users/niejunnan/workspace/robometer && export CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_ENDPOINT=https://hf-mirror.com HF_HOME=/vla/users/niejunnan/assets/hf_cache HF_HUB_CACHE=/vla/users/niejunnan/assets/hf_cache/hub TOKENIZERS_PARALLELISM=false TRANSFORMERS_NO_TF=1 USE_TF=0 USE_FLAX=0 && /vla/users/niejunnan/workspace/robometer/.venv/bin/python robometer/evals/eval_server.py model_path=/vla/users/niejunnan/assets/Robometer-4B server_url=0.0.0.0 server_port=8410 num_gpus=1 max_workers=1'"
```

Task 0 adapter, exposing the YAML-configured `predict_progress` port `50152`:

```bash
tmux new-session -d -s rlt_spatial_task0_robometer_adapter_50152 -n reward_adapter \
  "bash -lc 'cd /vla/users/niejunnan/codebase/VLA-RL && /vla/users/niejunnan/envs/serl_torch/bin/python scripts/serve_robometer_progress_http.py --robometer-url http://127.0.0.1:8410 --host 127.0.0.1 --port 50152 --image-keys image_rgb_0 image_rgb_1 --view-mode average_two --max-history-frames 8 --use-frame-steps'"
```

Wait for the adapter before launching RoboMeter PBRS training:

```bash
/vla/users/niejunnan/envs/serl_torch/bin/python \
  examples/libero/rlt/tools/wait_for_tcp.py \
  --host 127.0.0.1 \
  --ports 50152 \
  --timeout-sec 900
```

Then launch stage 2 with the RoboMeter YAML:

```bash
bash examples/libero/rlt/tools/launch_rlt.sh \
  --config examples/libero/rlt/configs/reward_model/libero_spatial_task0_self_cond_512_reward_model_robometer_pbrs_stage2.yaml \
  --session rlt_spatial_task0_robometer_pbrs \
  --actor-gpu 2 \
  --learner-gpu 3 \
  --policy-gpu 2 \
  --with-eval \
  --eval-gpu 2 \
  --trainer-port 63000 \
  --broadcast-port 63001 \
  --policy-port 43000 \
  --eval-policy-port 43001 \
  --run-dir /vla/users/niejunnan/codebase/VLA-RL/outputs/libero_spatial_task0_self_conditioned_prefix_512_chunk5_reward_model_robometer_pbrs_rlt \
  --policy-checkpoint /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000_pytorch
```

For other tasks, keep the task-specific YAML and adapter port from the table.
Use separate trainer/broadcast/policy/eval-policy ports if running RoboMeter and
Robo-Dopamine experiments on the same host at the same time.

## Monitoring

Check tmux sessions:

```bash
tmux ls | grep -E 'rlt_spatial_task|robodopamine'
```

Check reward server ports:

```bash
ss -ltnp | grep -E ':(50052|50053|50054|50055|50056|50057|50058|50059|50060|50061)\b'
```

Check actor-side reward health:

```bash
tail -n 5 /vla/users/niejunnan/codebase/VLA-RL/outputs/libero_spatial_task0_self_conditioned_prefix_512_chunk5_reward_model_pbrs_rlt/actor_metrics.jsonl
```

Check per-chunk absolute progress events:

```bash
tail -n 5 /vla/users/niejunnan/codebase/VLA-RL/outputs/libero_spatial_task0_self_conditioned_prefix_512_chunk5_reward_model_pbrs_rlt/progress_events.jsonl
```

For reward-model runs, the key fields are:

```text
reward/committed
reward/submitted
reward/pending
reward/failed
reward/last_latency_sec
```

Healthy runs should have `reward/failed = 0`, and `reward/submitted - reward/committed` close to the pending queue size.
`progress_events.jsonl` records the absolute progress returned by the remote
reward model at chunk boundaries, plus `previous_progress`, `env_reward`,
`computed_reward`, `discount`, `executed_steps`, and `latency_sec`. Use this file
for offline reward-shaping analysis.

## Current Running-Job Note

The initial task0/task1 smoke launch used one shared reward server on port
`50052`. That is valid only as a throughput smoke test. Standard benchmark runs
should use the dedicated-port mapping in this document.
