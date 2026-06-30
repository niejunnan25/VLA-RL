# Assets and Checkpoints

VLA-RL does not vendor large model checkpoints, LIBERO datasets, or reward-model
assets. Keep them outside the repository and point launch scripts to them with
environment variables.

## Recommended Directory Layout

The validated machines use a layout like this:

```text
/vla/users/niejunnan/assets/
  openpi-assets/
    serl_torch_ckpt/pi0_10000/
  Robo-Dopamine-GRM-2.0-4B-Preview/
  Robometer-4B/
  robometer_processed_datasets/
  sentence-transformers/all-MiniLM-L12-v2/
  hf_cache/

/vla/users/niejunnan/datasets/
  libero_lerobot/

/vla/users/niejunnan/codebase/
  openpi-modified/
  Robo-Dopamine/

/vla/users/niejunnan/workspace/
  robometer/
```

This is a machine-local convention, not a repository requirement.

## Path Variables

Use these variables to adapt the examples to another machine:

| Variable | Purpose |
|---|---|
| `OPENPI_ROOT` | OpenPI checkout used to serve the frozen base policy. |
| `OPENPI_CONDA_ENV` | Conda env used by OpenPI service scripts. |
| `POLICY_CONFIG` | OpenPI policy config, usually `pi0_libero_baseline_10_bs32_150000`. |
| `POLICY_DIR` | OpenPI checkpoint directory, usually the `pi0_10000` checkpoint. |
| `LIBERO_ROOT` | LIBERO checkout if the env server needs an explicit path. |
| `LIBERO_DATASETS_ROOT` | LIBERO benchmark datasets. |
| `REWARD_GOAL_DATASET` | Lerobot-style LIBERO demos used to fetch goal images for reward models. |
| `REWARD_MODEL_REPO` | Robo-Dopamine repository root. |
| `REWARD_MODEL_CONDA_ENV` | Robo-Dopamine Python environment. |
| `REWARD_MODEL_PATH` | Robo-Dopamine checkpoint path. |
| `ROBOMETER_ROOT` | RoboMeter repository root. |
| `ROBOMETER_MODEL_PATH` | RoboMeter checkpoint path. |
| `ROBOMETER_PYTHON_CMD` | Python command used to run RoboMeter. |
| `ROBOMETER_PROCESSED_DATASETS_PATH` | RoboMeter processed benchmark assets. |
| `ROBOMETER_SENTENCE_MODEL_PATH` | Sentence-transformer model used by RoboMeter tools. |
| `HF_HOME` / `HF_HUB_CACHE` | Hugging Face cache paths. |

Copy `.env.example` and edit these variables for your machine.

## OpenPI Base Policy

The residual SAC experiments currently assume:

```text
POLICY_CONFIG=pi0_libero_baseline_10_bs32_150000
POLICY_DIR=/path/to/openpi-assets/serl_torch_ckpt/pi0_10000
```

The OpenPI service is started through
`examples/libero/residual_sac/tools/serve_openpi_10000_policy.sh`. The actor
only sees the service endpoint defined by the YAML `policy.host` and
`policy.port` fields.

## LIBERO Goal Dataset

Reward-model PBRS uses absolute progress values. When a reward model needs a
goal image, the adapter loads the final frame from the corresponding task demos.
Set:

```bash
export REWARD_GOAL_DATASET=/path/to/libero_lerobot
```

The adapter uses the task prompt from the environment and the configured task id
to select the matching goal. Keep this dataset version fixed across experiments.

## Robo-Dopamine Checkpoint

The default formal configuration uses:

```text
Robo-Dopamine-GRM-2.0-4B-Preview
```

Set:

```bash
export REWARD_MODEL_REPO=/path/to/Robo-Dopamine
export REWARD_MODEL_CONDA_ENV=/path/to/envs/robo-dopamine
export REWARD_MODEL_PATH=/path/to/Robo-Dopamine-GRM-2.0-4B-Preview
```

The service script also sets `VLLM_ATTENTION_BACKEND=TORCH_SDPA` by default.

## RoboMeter Checkpoint

The default formal configuration uses:

```text
Robometer-4B
```

Set:

```bash
export ROBOMETER_ROOT=/path/to/robometer
export ROBOMETER_MODEL_PATH=/path/to/Robometer-4B
export ROBOMETER_PYTHON_CMD=/path/to/robometer/.venv/bin/python
```

For LIBERO training the recommended backend is `native`; it avoids an extra
RoboMeter eval-server hop and keeps reward latency low.

## Reproducibility Rule

Do not change checkpoint paths, goal dataset versions, or policy configs inside
a result directory after a run starts. If any asset changes, create a new YAML or
new output root so SwanLab curves, local JSONL logs, and checkpoints remain
traceable to the exact assets used.
