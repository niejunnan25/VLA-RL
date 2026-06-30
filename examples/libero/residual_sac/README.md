# LIBERO Residual SAC

This example is the current recommended LIBERO residual SAC mainline in VLA-RL.
It is a SERL Torch-aligned implementation for chunk-level residual learning on
top of a frozen OpenPI base policy.

## What Runs

One formal run is a set of cooperating processes:

| Component | Role |
|---|---|
| learner | Owns SAC networks, replay, checkpoints, SwanLab/W&B-compatible logging, and async eval queueing. |
| actor | Queries OpenPI, samples residual actions, steps LIBERO in chunks, and sends transitions to the learner. |
| train env server | Serves the LIBERO training environment over local HTTP RPC. |
| eval env server | Serves a separate LIBERO environment for async eval. |
| OpenPI policy server | Serves the frozen base action chunks from `pi0_10000`. |
| reward server | Optional. Serves Robo-Dopamine or RoboMeter absolute progress values. |

The launcher starts these processes in one shell invocation and writes service
logs under the run output root.

## Action Semantics

The actor uses chunk-level residual SAC:

```text
base_actions = OpenPI(obs)
residual_actions = residual_agent(obs_with_base_actions)
final_actions = base_actions + alpha * residual_actions
env.step_chunk(final_actions)
```

Replay stores chunk transitions rather than dense per-step transitions. The
default LIBERO spatial configs use `chunk_horizon: 5` and `alpha: 0.2`.

## Reward Variants

| Variant | YAML location | Reward source | Transform |
|---|---|---|---|
| Sparse | `configs/libero_spatial_task*_sparse.yaml` | LIBERO env sparse success reward | `env_only` |
| Robo-Dopamine PBRS | `configs/reward_model/libero_spatial_task*_robodopamine_pbrs*.yaml` | Robo-Dopamine absolute progress | `env_plus_potential_delta` |
| RoboMeter PBRS | `configs/reward_model/libero_spatial_task*_robometer_pbrs*.yaml` | RoboMeter absolute progress | `env_plus_potential_delta` |

The reward servers return absolute progress. The training loop applies the YAML
reward transform. For PBRS this means the replay reward is the sparse env reward
plus a potential-delta shaping term.

## Config Matrix

The formal spatial tasks are:

```text
task4, task5, task8, task9
```

Sparse configs currently provide the baseline seed used by the first formal
runs:

```text
libero_spatial_task4_sparse.yaml
libero_spatial_task5_sparse.yaml
libero_spatial_task8_sparse.yaml
libero_spatial_task9_sparse.yaml
```

Reward-model configs provide three run labels for each task/model:

| Label | `env.seed` | Filename suffix |
|---|---:|---|
| `s0` | 7 | no suffix in the filename; `wandb.exp_name` ends in `_s0` |
| `s17` | 17 | `_s17.yaml` |
| `s27` | 27 | `_s27.yaml` |

Example:

```text
configs/reward_model/libero_spatial_task5_robodopamine_pbrs.yaml
configs/reward_model/libero_spatial_task5_robodopamine_pbrs_s17.yaml
configs/reward_model/libero_spatial_task5_robodopamine_pbrs_s27.yaml
```

The YAML owns the task id, environment seed, reward semantics, ports, output
root, SwanLab run name, async eval cadence, and training budget. Prefer changing
YAML over command-line overrides for formal experiments.

## Install and Assets

Read the repository-level setup first:

- `docs/setup.md`
- `docs/assets.md`
- `.env.example`

The minimal editable install is:

```bash
pip install -e ".[dev,wandb,residual_sac]"
```

The OpenPI, LIBERO, Robo-Dopamine, and RoboMeter stacks remain external services.
Configure their paths through environment variables rather than importing them
directly into the actor or learner.

## Smoke Test

Run a sparse smoke before a formal run:

```bash
bash examples/libero/residual_sac/tools/smoke_residual_sac.sh \
  --task 4 \
  --reward sparse \
  --actor-gpu 0 \
  --learner-gpu 1
```

Run a reward-model smoke:

```bash
bash examples/libero/residual_sac/tools/smoke_residual_sac.sh \
  --task 4 \
  --reward robometer \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --reward-model-gpu 1
```

The smoke disables online logging, uses a short budget, disables torch compile,
and writes to a timestamped output root under `outputs/smoke`.

## Formal Sparse Run

```bash
cd /path/to/VLA-RL

bash examples/libero/residual_sac/tools/launch_residual_sac.sh \
  --mode chunk \
  --config-name libero_spatial_task4_sparse \
  --learner-gpu 1 \
  --actor-gpu 0 \
  --env-gpu 0 \
  --eval-env-gpu 0 \
  --policy-gpu 0 \
  --policy-server managed \
  --reward-model false \
  --with-eval-env
```

## Formal Robo-Dopamine PBRS Run

```bash
cd /path/to/VLA-RL

bash examples/libero/residual_sac/tools/launch_residual_sac.sh \
  --mode chunk \
  --config-name reward_model/libero_spatial_task4_robodopamine_pbrs \
  --learner-gpu 1 \
  --actor-gpu 0 \
  --env-gpu 0 \
  --eval-env-gpu 0 \
  --policy-gpu 0 \
  --reward-model-gpu 1 \
  --policy-server managed \
  --reward-model true \
  --with-eval-env \
  --learner-gpu-memory-guard-fraction 0
```

Use `_s17` or `_s27` config names for the additional seeded runs.

## Formal RoboMeter PBRS Run

```bash
cd /path/to/VLA-RL

bash examples/libero/residual_sac/tools/launch_residual_sac.sh \
  --mode chunk \
  --config-name reward_model/libero_spatial_task4_robometer_pbrs \
  --learner-gpu 1 \
  --actor-gpu 0 \
  --env-gpu 0 \
  --eval-env-gpu 0 \
  --policy-gpu 0 \
  --reward-model-gpu 1 \
  --policy-server managed \
  --reward-model true \
  --with-eval-env \
  --learner-gpu-memory-guard-fraction 0
```

RoboMeter is latency-sensitive. Keep the OpenPI policy server and the RoboMeter
reward server on separate GPUs when possible.

## Logs and Metrics

For a run root such as `.../task04`, check:

```text
actor/episode_logs.jsonl
actor/actor_timers.jsonl
learner/metrics.jsonl
learner/summary.json
learner/async_eval_results.jsonl
services/*.log
rollout/
```

Cloud logging uploads a small metric set under `rollout/*`, `learner/*`, and
`eval/*`. Local JSONL files are the source of truth for debugging.

Healthy actor speed is usually above `10 env/s` after startup. If actor speed is
much lower, inspect GPU placement, policy server latency, reward server latency,
and LIBERO env reset timing before tuning SAC hyperparameters.

## Common Failures

| Symptom | Likely cause | First check |
|---|---|---|
| Port already in use | Another run has the same YAML ports active. | `tmux ls`, `lsof -i :<port>` |
| Actor writes no episodes | Env or policy service did not become ready. | `services/train_env.log`, `services/policy.log` |
| Reward model stalls actor | Reward server is too slow or sharing a busy GPU. | `services/reward_model.log` |
| Async eval does not finish | Eval env server or checkpoint queue issue. | `learner/async_eval_worker.log` |
| SwanLab run missing | Logging disabled or credentials/env missing. | `learner/metrics.jsonl` first, then cloud logging env |

GPU placement notes are maintained in
`examples/libero/residual_sac/docs/2026-06-29-gpu-scheduling-notes.md`.
