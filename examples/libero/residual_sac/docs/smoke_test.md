# Residual SAC Smoke Test

The smoke test checks connectivity, not final learning performance. It should
be run on every new machine before starting a 300k-step formal experiment.

## What It Checks

A smoke run should prove that:

- the LIBERO train env server starts;
- the LIBERO eval env server starts;
- the OpenPI policy server starts;
- the actor can query OpenPI and call `env.step_chunk()`;
- the learner receives replay and starts updates;
- async eval can run at least one short eval;
- reward-model RPC works when `--reward robodopamine` or `--reward robometer`
  is selected.

## Sparse Smoke

```bash
cd /path/to/VLA-RL

bash examples/libero/residual_sac/tools/smoke_residual_sac.sh \
  --task 4 \
  --reward sparse \
  --actor-gpu 0 \
  --learner-gpu 1
```

## Robo-Dopamine Smoke

```bash
cd /path/to/VLA-RL

bash examples/libero/residual_sac/tools/smoke_residual_sac.sh \
  --task 4 \
  --reward robodopamine \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --reward-model-gpu 1
```

## RoboMeter Smoke

```bash
cd /path/to/VLA-RL

bash examples/libero/residual_sac/tools/smoke_residual_sac.sh \
  --task 4 \
  --reward robometer \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --reward-model-gpu 1
```

## Useful Environment Overrides

```bash
MAX_ENV_STEPS=3000 \
TRAINING_STARTS=20 \
ASYNC_EVAL_EVERY_EPISODES=5 \
ASYNC_EVAL_EPISODES=2 \
bash examples/libero/residual_sac/tools/smoke_residual_sac.sh ...
```

Use `DRY_RUN=1` to check command composition without launching processes:

```bash
DRY_RUN=1 bash examples/libero/residual_sac/tools/smoke_residual_sac.sh \
  --task 4 \
  --reward robometer \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --reward-model-gpu 1
```

## Expected Output

The script writes to:

```text
examples/libero/residual_sac/outputs/smoke/<reward>_taskXX_<timestamp>
```

After startup, inspect:

```text
actor/episode_logs.jsonl
learner/metrics.jsonl
learner/async_eval_results.jsonl
services/*.log
```

The smoke is healthy when actor env steps increase, replay size increases,
learner update steps appear after warmup, and async eval writes at least one
result. For reward-model smoke tests, `services/reward_model.log` should show
successful progress requests.

## Do Not Use Smoke Results as Method Results

Smoke runs disable online logging by default, disable torch compile, use short
budgets, and reduce eval cadence. They are plumbing tests only. Formal results
should come from the YAML-defined 300k-step runs.
