# Residual SAC Experiment Matrix

This document records the formal LIBERO spatial reward-model matrix currently
represented by YAML files.

## Tasks

```text
libero_spatial task4
libero_spatial task5
libero_spatial task8
libero_spatial task9
```

## Reward-Model Methods

```text
Robo-Dopamine PBRS
RoboMeter PBRS
```

Both methods use absolute progress from the reward model and apply
`env_plus_potential_delta` in the learner.

## Seeds

The run label maps to `env.seed` as follows:

| Run label | `env.seed` |
|---|---:|
| `s0` | 7 |
| `s17` | 17 |
| `s27` | 27 |

`global_seed` remains `0` in these configs so this matrix isolates environment
seed changes.

## 24 Reward-Model Runs

| Method | Task | s0 config | s17 config | s27 config |
|---|---:|---|---|---|
| Robo-Dopamine PBRS | 4 | `reward_model/libero_spatial_task4_robodopamine_pbrs` | `reward_model/libero_spatial_task4_robodopamine_pbrs_s17` | `reward_model/libero_spatial_task4_robodopamine_pbrs_s27` |
| Robo-Dopamine PBRS | 5 | `reward_model/libero_spatial_task5_robodopamine_pbrs` | `reward_model/libero_spatial_task5_robodopamine_pbrs_s17` | `reward_model/libero_spatial_task5_robodopamine_pbrs_s27` |
| Robo-Dopamine PBRS | 8 | `reward_model/libero_spatial_task8_robodopamine_pbrs` | `reward_model/libero_spatial_task8_robodopamine_pbrs_s17` | `reward_model/libero_spatial_task8_robodopamine_pbrs_s27` |
| Robo-Dopamine PBRS | 9 | `reward_model/libero_spatial_task9_robodopamine_pbrs` | `reward_model/libero_spatial_task9_robodopamine_pbrs_s17` | `reward_model/libero_spatial_task9_robodopamine_pbrs_s27` |
| RoboMeter PBRS | 4 | `reward_model/libero_spatial_task4_robometer_pbrs` | `reward_model/libero_spatial_task4_robometer_pbrs_s17` | `reward_model/libero_spatial_task4_robometer_pbrs_s27` |
| RoboMeter PBRS | 5 | `reward_model/libero_spatial_task5_robometer_pbrs` | `reward_model/libero_spatial_task5_robometer_pbrs_s17` | `reward_model/libero_spatial_task5_robometer_pbrs_s27` |
| RoboMeter PBRS | 8 | `reward_model/libero_spatial_task8_robometer_pbrs` | `reward_model/libero_spatial_task8_robometer_pbrs_s17` | `reward_model/libero_spatial_task8_robometer_pbrs_s27` |
| RoboMeter PBRS | 9 | `reward_model/libero_spatial_task9_robometer_pbrs` | `reward_model/libero_spatial_task9_robometer_pbrs_s17` | `reward_model/libero_spatial_task9_robometer_pbrs_s27` |

## Sparse Baselines

Sparse configs currently exist for the main baseline seed:

```text
libero_spatial_task4_sparse
libero_spatial_task5_sparse
libero_spatial_task8_sparse
libero_spatial_task9_sparse
```

If sparse baselines need the same 3-seed treatment, create matching sparse YAMLs
rather than overriding `env.seed` on the command line.

## Launch Pattern

Use the same launcher for every row:

```bash
bash examples/libero/residual_sac/tools/launch_residual_sac.sh \
  --mode chunk \
  --config-name <config> \
  --learner-gpu <learner_gpu> \
  --actor-gpu <actor_gpu> \
  --env-gpu <actor_gpu> \
  --eval-env-gpu <actor_gpu> \
  --policy-gpu <actor_gpu> \
  --reward-model-gpu <learner_or_reward_gpu> \
  --policy-server managed \
  --reward-model true \
  --with-eval-env \
  --learner-gpu-memory-guard-fraction 0
```

For sparse runs set `--reward-model false` and omit `--reward-model-gpu`.

## Reporting Rule

Report results by task, method, and seed. Use local JSONL as the source of truth
for rollout curves and async eval; use SwanLab for quick monitoring and plots.
Do not merge curves from different seeds unless the config names and output roots
prove they used the same base policy, reward checkpoint, goal dataset, and SAC
hyperparameters.
