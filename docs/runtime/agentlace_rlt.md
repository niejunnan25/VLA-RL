# Agentlace RLT Runbook

This runbook describes the current production-style RLT Stage 2 path in
VLA-RL. It uses Agentlace for actor-learner communication, OpenPI for the
frozen base VLA, LIBERO as the remote environment, and the RLT actor/critic as
the learner-side algorithm.

## Runtime Layout

- Learner: run from the `serl_torch` conda environment.
- Actor/OpenPI: run from
  `/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3`.
- LIBERO env server: launched through the external
  `/vla/users/niejunnan/codebase/serl_torch-rlt-merge/examples/libero/tools/serve_env.sh`
  wrapper.
- OpenPI fork: the configured `openpi_root` must expose
  `PI0Pytorch.sample_actions_with_features()`. The default fork path is
  `/vla/users/niejunnan/codebase/openpi-rlt-github`.

The 234 smoke environment required these packages:

- `serl_torch`: `numpydantic`, `tyro`.
- OpenPI venv: `agentlace`, `lz4`.

## Launcher Parameters

`scripts/launch_agentlace_rlt.sh` starts one tmux session with three windows:

- `env`: LIBERO env server.
- `learner`: Agentlace learner and RLT optimizer.
- `actor`: LIBERO rollout, OpenPI inference, RLT feature processing, and action
  execution.

Important options:

- `--actor-gpu`: GPU used by actor and OpenPI. Default convention is GPU0.
- `--learner-gpu`: GPU used by learner. Default convention is GPU1.
- `--env-gpu`: GPU used by LIBERO env server. Defaults to actor GPU.
- `--env-port`: HTTP port for the LIBERO env server.
- `--trainer-port`: Agentlace trainer request/data port.
- `--broadcast-port`: Agentlace network broadcast port.
- `--run-dir`: directory for `metrics.jsonl`, `actor_metrics.jsonl`,
  summaries, and checkpoints.
- `--python`: learner Python executable.
- `--actor-python`: actor/OpenPI Python executable.
- `--`: all arguments after this marker are OmegaConf dotlist overrides passed
  to both actor and learner.

Use distinct port triples for concurrent runs.

## Commands

### 200-step Smoke

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/VLA-RL

rm -rf /tmp/vlarl_m4_agentlace_smoke

bash scripts/launch_agentlace_rlt.sh \
  --session vlarl_m4_agentlace_smoke \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --env-port 23100 \
  --trainer-port 5518 \
  --broadcast-port 5519 \
  --run-dir /tmp/vlarl_m4_agentlace_smoke \
  -- runtime.max_env_steps=200 \
     runtime.max_update_steps=200 \
     runtime.checkpoint_interval_env_steps=100
```

### 1k Sanity

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/VLA-RL

rm -rf /tmp/vlarl_m4_agentlace_1k_sanity

bash scripts/launch_agentlace_rlt.sh \
  --session vlarl_m4_agentlace_1k_sanity \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --env-port 23110 \
  --trainer-port 5528 \
  --broadcast-port 5529 \
  --run-dir /tmp/vlarl_m4_agentlace_1k_sanity \
  -- runtime.max_env_steps=1000 \
     runtime.max_update_steps=1000 \
     runtime.checkpoint_interval_env_steps=500
```

### 300k Task-4 Training

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/VLA-RL

bash scripts/launch_agentlace_rlt.sh \
  --session vlarl_task4_rlt_300k \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --env-port 23120 \
  --trainer-port 5538 \
  --broadcast-port 5539 \
  --run-dir outputs/libero_spatial_task4_openpi_rlt_agentlace
```

The default recipe already sets `runtime.max_env_steps=300000` and
`runtime.max_update_steps=300000`.

## Metrics and Acceptance

The learner writes `metrics.jsonl` and `summary.json`. The actor writes
`actor_metrics.jsonl` and `actor_summary.json`.

The learner keeps the Agentlace server alive after it reaches
`runtime.max_update_steps` until the actor has also reached
`runtime.max_env_steps` or reports its final actor summary. This keeps actor
rollout and learner checkpoint accounting aligned for sanity checks.

For smoke and sanity checks, verify:

- `actor_summary.json.env_steps` reaches the requested env-step target.
- `actor_summary.json.received_policy_state` is `true`.
- `summary.json.update_steps` reaches the requested update target.
- `summary.json.replay_size > 0`.
- `checkpoints/latest.pt` and `checkpoints/final.pt` exist.
- `train/critic_loss`, `train/actor_loss`, and Q metrics in `metrics.jsonl`
  are finite.

Useful speed estimates:

- Actor wall env/s: final actor env steps divided by the actor wall-clock span.
- Actor active env/s: sum of `executed_steps` divided by sum of
  `chunk_time_sec`.
- Learner wall update/s: final update steps divided by learner wall-clock span.
- Learner active update/s: inverse of mean `update_time_sec`.

After a run, stop the tmux session if needed and confirm no stale GPU compute
processes remain.
