# VLA-RL

VLA-RL is a sample-efficient reinforcement-learning infrastructure for
post-training vision-language-action policies.

Milestone 4 defines the core contracts, wires OpenPI and LIBERO as the first
real policy/env boundaries, integrates RLT Stage 2 as the first real algorithm,
and adds an Agentlace-first actor-learner runtime. PLD, residual SAC, and
StarVLA are still future targets.

## v0 Targets

- Policies: OpenPI, StarVLA
- Environments: LIBERO
- Algorithms: RLT, PLD, residual SAC

## Current Scope

- `vla_rl.data`: shared schemas passed between environments, policies,
  algorithms, replay, and runtime.
- `vla_rl.policies`: policy backend interfaces, fake policy, and OpenPI adapter.
- `vla_rl.envs`: environment backend interfaces, fake environment, and LIBERO
  remote client.
- `vla_rl.algorithms`: algorithm interface and fake algorithm.
- `vla_rl.algorithms.rlt`: RLT Stage 2 actor/critic and frozen encoder feature
  processing.
- `vla_rl.runtime`: local debug runners, Agentlace actor-learner runtime,
  checkpointing, synchronous evaluation, and HTTP RPC utilities.
- `recipes`: composable configuration files.

## Local Smoke

```bash
python scripts/train.py --config recipes/fake_local.yaml
pytest -q
```

## LIBERO + OpenPI Integration Shape

Start the external LIBERO server through the compatibility wrapper:

```bash
python scripts/serve_libero_env.py \
  --serl-torch-root /vla/users/niejunnan/codebase/serl_torch-rlt-merge \
  --host 127.0.0.1 \
  --port 23000 \
  --gpu-id 0
```

Then run a local actor-learner recipe. This uses OpenPI for reference actions
and features, LIBERO over RPC for rollout, and `FakeAlgorithm` as the placeholder
algorithm:

```bash
python scripts/train.py --config recipes/libero_spatial_openpi_fake_algorithm.yaml
```

The real recipe requires the `niejunnan25/openpi` extractor fork and Torch
OpenPI checkpoint paths configured in `recipes/config/policy/openpi_*.yaml`.

## RLT Stage 2 Smoke

```bash
python scripts/train.py --config recipes/libero_spatial_openpi_rlt_smoke.yaml
```

This smoke recipe uses OpenPI + LIBERO + a frozen Stage 1 RLTokenEncoder and
trains the local RLT actor/critic. It is single-process and intended for systems
checks, not final experiment throughput.

## RLT Stage 2 Formal Single-Process Run

The single-process runner is kept as a debug path. For real training, prefer
the Agentlace runtime below.

Start separate LIBERO servers for training and synchronous evaluation:

```bash
python scripts/serve_libero_env.py \
  --serl-torch-root /vla/users/niejunnan/codebase/serl_torch-rlt-merge \
  --host 127.0.0.1 \
  --port 23000 \
  --gpu-id 0

python scripts/serve_libero_env.py \
  --serl-torch-root /vla/users/niejunnan/codebase/serl_torch-rlt-merge \
  --host 127.0.0.1 \
  --port 23001 \
  --gpu-id 0
```

Run the 300k task-4 recipe:

```bash
python scripts/train.py --config recipes/libero_spatial_task4_openpi_rlt_300k.yaml
```

The run directory contains `config.yaml`, `metrics.jsonl`, `summary.json`, and
`checkpoints/latest.pt`. Resume by setting `runtime.resume_from` in the recipe
or by editing a copy of the YAML to point at `checkpoints/latest.pt`.

## RLT Stage 2 Agentlace Run

The default real-training path is Agentlace actor-learner. The helper starts a
LIBERO env server, learner, and actor in one tmux session:

```bash
scripts/launch_agentlace_rlt.sh \
  --session vlarl_rlt_task4 \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --env-port 23000 \
  --trainer-port 5488 \
  --broadcast-port 5489 \
  --run-dir outputs/libero_spatial_task4_openpi_rlt_agentlace
```

For a short smoke, pass overrides after `--`:

```bash
scripts/launch_agentlace_rlt.sh \
  --session vlarl_rlt_task4_smoke \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --run-dir /tmp/vlarl_rlt_agentlace_smoke \
  -- \
  runtime.max_env_steps=200 \
  runtime.max_update_steps=200
```

The actor sends compact RLT transitions to the learner rather than raw images.
The learner owns replay, updates, metrics, and checkpoints.
