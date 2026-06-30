# Setup

This document describes the minimum setup needed to run the current LIBERO
residual SAC mainline in VLA-RL. The repository intentionally keeps simulators,
VLA policies, and reward models as external services. Do not try to force all
of them into one Python environment.

## Recommended System

The validated development machines use:

- Ubuntu 20.04 or newer.
- NVIDIA GPUs with a CUDA-capable driver.
- Python 3.10.
- Conda or another environment manager that can create isolated environments.
- Git and a working C/C++ build toolchain for packages with native extensions.

The examples assume a Linux server with local GPUs. macOS is useful for editing
and plotting, but not for running LIBERO training.

## Environment Layout

Use separate environments for separate dependency stacks:

| Component | Recommended environment | Why |
|---|---|---|
| VLA-RL / residual SAC | `serl_torch` | Owns actor, learner, replay, logging, and launch scripts. |
| OpenPI | OpenPI-managed env | Policy checkpoints and preprocessing are OpenPI-specific. |
| Robo-Dopamine | Robo-Dopamine env | GRM inference uses its own model/vLLM stack. |
| RoboMeter | RoboMeter `.venv` or `uv run` | RoboMeter has its own evaluation/inference dependencies. |
| LIBERO | Available to `serl_torch` env | The env server imports LIBERO and robosuite. |

The runtime connection between these components is HTTP/WebSocket, not Python
imports across projects.

## Create the VLA-RL Environment

```bash
conda create -n serl_torch python=3.10 -y
conda activate serl_torch
```

Install PyTorch for your CUDA version first. The exact command depends on your
driver/CUDA stack. For example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install VLA-RL in editable mode:

```bash
cd /path/to/VLA-RL
pip install -e ".[dev,wandb,residual_sac]"
```

The `residual_sac` extra installs runtime packages used by the standalone
residual SAC package, including Hydra, Pillow, OpenCV, Gym/Gymnasium,
websockets/msgpack helpers, tqdm, and Transformers for the ResNet encoder.

## Install LIBERO

LIBERO must be importable in the environment that starts the env server. The
validated path uses the LIBERO checkout and dependencies from the SERL Torch
stack. If you maintain a separate LIBERO checkout, set:

```bash
export LIBERO_ROOT=/path/to/LIBERO
export LIBERO_DATASETS_ROOT=/path/to/libero/datasets
```

The residual SAC launcher starts the LIBERO env server through:

```text
examples/libero/residual_sac/tools/serve_env.sh
```

That server eventually imports `examples/libero/residual_sac/env/task_env.py`,
so missing LIBERO/robosuite packages will show up during service startup.

## Install OpenPI

OpenPI serves the frozen base policy. The residual SAC launcher can manage the
OpenPI server through:

```text
examples/libero/residual_sac/tools/serve_openpi_10000_policy.sh
```

Set these variables if your paths differ from the machine defaults:

```bash
export OPENPI_ROOT=/path/to/openpi-modified
export OPENPI_CONDA_ENV=openpi-modified
export POLICY_CONFIG=pi0_libero_baseline_10_bs32_150000
export POLICY_DIR=/path/to/openpi-assets/serl_torch_ckpt/pi0_10000
```

The policy server exposes OpenPI action inference to the actor through a local
WebSocket client. Training code should not directly import OpenPI unless you are
debugging the policy backend itself.

## Install Robo-Dopamine

Robo-Dopamine runs as a reward-model service. Use its own environment and
checkpoint directory:

```bash
export REWARD_MODEL_REPO=/path/to/Robo-Dopamine
export REWARD_MODEL_CONDA_ENV=/path/to/envs/robo-dopamine
export REWARD_MODEL_PATH=/path/to/Robo-Dopamine-GRM-2.0-4B-Preview
export REWARD_GOAL_DATASET=/path/to/libero_lerobot
```

The launcher starts it through:

```text
examples/libero/residual_sac/tools/serve_robodopamine_progress.sh
```

The training side only receives absolute progress values and applies the reward
transform configured in YAML, usually `env_plus_potential_delta` for PBRS.

## Install RoboMeter

RoboMeter also runs as a reward-model service. It may use its own `.venv` under
the RoboMeter checkout or `uv run python` if that is how the project is managed:

```bash
export ROBOMETER_ROOT=/path/to/robometer
export ROBOMETER_MODEL_PATH=/path/to/Robometer-4B
export ROBOMETER_PYTHON_CMD=/path/to/robometer/.venv/bin/python
export ROBOMETER_BACKEND=native
```

The launcher starts it through:

```text
examples/libero/residual_sac/tools/serve_robometer_progress_stack.sh
```

For LIBERO experiments the RoboMeter adapter uses single-view prefix-per-query
inference with a bounded history window. See
`examples/libero/residual_sac/docs/reward_models.md` for details.

## Configure Local Paths

Copy the path template and edit it for your machine:

```bash
cp .env.example .env
${EDITOR:-vi} .env
set -a
source .env
set +a
```

These variables are read by launch scripts. YAML files still define the actual
experiment semantics: task id, reward source, reward transform, seed, ports,
training budget, logging names, and output roots.

## Check the Environment

Run:

```bash
python scripts/check_env.py
```

Use strict mode before formal runs:

```bash
python scripts/check_env.py --strict
```

Strict mode fails if core imports or configured asset paths are missing. It does
not start GPU services or download checkpoints.

## Smoke Test

Before a 300k-step experiment, run a small residual SAC smoke:

```bash
bash examples/libero/residual_sac/tools/smoke_residual_sac.sh \
  --task 4 \
  --reward sparse \
  --actor-gpu 0 \
  --learner-gpu 1
```

For reward-model smoke tests, set `--reward robodopamine` or
`--reward robometer` and pass `--reward-model-gpu` if it should not share the
learner GPU.

## Formal Runs

After the smoke passes, launch formal 300k-step runs through the YAML files in:

```text
examples/libero/residual_sac/configs
examples/libero/residual_sac/configs/reward_model
```

The residual SAC README contains exact commands and monitoring guidance.
