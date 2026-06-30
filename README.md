# VLA-RL

VLA-RL is a sample-efficient reinforcement-learning infrastructure for
post-training vision-language-action policies.

VLA-RL keeps large VLA models and simulation environments outside the RL
process. They run as external services, while VLA-RL owns the lightweight
actor/critic training loops, replay, checkpoints, and example recipes. RLT is
the first clean SERL-style training line; PLD and residual SAC are separate
example lines rather than variants hidden behind a universal runner.


## Start Here

Use these documents in order when setting up a fresh checkout:

1. `docs/setup.md`: Python environment, editable install, external service
   repositories, and sanity checks.
2. `docs/assets.md`: checkpoint and dataset layout, including the environment
   variables used to override machine-local paths.
3. `.env.example`: copy this to `.env` or source the same variables in your
   shell before launching experiments.
4. `examples/libero/residual_sac/README.md`: the current recommended LIBERO
   residual SAC training path, including sparse, Robo-Dopamine PBRS, and
   RoboMeter PBRS recipes.
5. `examples/libero/residual_sac/docs/smoke_test.md`: a short connectivity run
   before starting a formal 300k-step experiment.

The current recommended mainline is:

```text
examples/libero/residual_sac
```

It mirrors the validated SERL Torch LIBERO residual SAC stack while keeping the
training code inside this repository. The older `examples/libero/rlt` and
`examples/libero/pld` examples remain useful references, but new LIBERO residual
reward-model experiments should start from `examples/libero/residual_sac`.

Run a local environment check after installation:

```bash
python scripts/check_env.py --strict
```

The strict mode checks local Python imports and the asset paths supplied through
environment variables. It does not download models or start GPU services.

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
- `vla_rl.algorithms.pld`: PLD Stage 1 residual action policy, residual
  observation processing, SAC updates, Cal-QL-style critic pretraining, and
  offline/online replay mixing.
- `residual_sac`: standalone SERL-style residual SAC package used by
  `examples/libero/residual_sac`. This package intentionally does not depend on
  the older `vla_rl.algorithms.pld` implementation because the residual SAC
  example is a strict alignment path for LIBERO chunk-level residual training.
- `vla_rl.runtime`: thin transport helpers, checkpointing, and HTTP/RPC utilities. Algorithm training loops live in examples.
- `examples/*/configs`: example-owned configuration files.

## Quick Checks

Use unit tests for library-level checks and short real example runs for end-to-end validation:

```bash
pytest -q
```

Real algorithm entrypoints live under their own examples.

## RLT Stage 2 SERL-Style Run

The default real-training path is the LIBERO RLT example. The actor and learner
loops live directly in `examples/libero/rlt/scripts/train_stage2.py`; Agentlace is used only as
transport for RLT transitions and actor-weight broadcasts. The helper
starts a LIBERO env server, OpenPI reference-policy server, learner, and actor in
one tmux session:

```bash
examples/libero/rlt/tools/launch_rlt.sh \
  --session vlarl_rlt_task4 \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --env-port 23000 \
  --policy-port 8899 \
  --trainer-port 5488 \
  --broadcast-port 5489 \
  --run-dir outputs/libero_spatial_task4_openpi_rlt
```

For a short smoke, pass overrides after `--`:

```bash
examples/libero/rlt/tools/launch_rlt.sh \
  --session vlarl_rlt_task4_smoke \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --run-dir /tmp/vlarl_rlt_smoke \
  -- \
  runtime.max_env_steps=1000
```

The actor sends RLT transitions to the learner rather than raw images.
The learner owns replay, updates, metrics, and checkpoints.

See `docs/runtime/agentlace_rlt.md` for the full RLT runbook and the exact
actor/learner data path.

W&B logging follows the lightweight HIL-SERL pattern: the
learner owns one W&B-compatible run and uploads only `rollout/*`, `learner/*`,
and `eval/*` metrics. Formal LIBERO configs enable it by default:
VLA-RL tries SwanLab first and falls back to W&B if SwanLab is unavailable.
Install the logging dependencies for formal online runs:

```bash
pip install -e ".[wandb]"
```

Override run metadata with:

```bash
wandb.project=vla-rl wandb.exp_name=task4_rlt_seed0
```

## PLD Stage 1 / Residual RL

VLA-RL implements PLD Stage 1 only: residual RL on top of a frozen OpenPI base
policy. Stage 2 hybrid data collection and Stage 3 VLA SFT are intentionally
out of scope for this milestone.

Collect successful base-policy replay:

```bash
examples/libero/pld/scripts/collect_base_success_replay.py \
  --config examples/libero/pld/configs/libero_spatial_task4_openpi_pld.yaml \
  --target-successes 50
```

Run PLD through its example-local launcher:

```bash
examples/libero/pld/tools/launch_pld.sh \
  --session vlarl_pld_task4_smoke \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --run-dir /tmp/vlarl_pld_smoke \
  -- \
  runtime.max_env_steps=1000 \
  runtime.calql_pretrain_steps=10 \
  runtime.training_starts=10
```

See `docs/algorithms/pld.md` for the PLD Stage 1 runbook and scope boundary.
The default PLD recipe follows the existing `serl_torch` PLD configs and uses
a frozen HuggingFace ResNet-18 image encoder, not torchvision.

## Evaluation Entrypoints

RLT and PLD keep evaluation local to their examples:

```bash
python examples/libero/rlt/scripts/eval_stage2.py --config <config> --checkpoint <checkpoint> --episodes 10 --output-dir <dir>
python examples/libero/pld/scripts/eval.py --config <config> --checkpoint <checkpoint> --episodes 10 --output-dir <dir>
```

PLD formal runs can use `examples/libero/pld/tools/launch_pld_after_collect.sh` to collect base-success replay before starting actor/learner training.


## Residual SAC Mainline

The LIBERO residual SAC example starts a full actor/learner/service stack from
one launcher:

```bash
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

See `examples/libero/residual_sac/README.md` for the reward-model variants,
seeded YAML matrix, log locations, SwanLab metrics, and GPU placement rules.
Use `examples/libero/residual_sac/tools/smoke_residual_sac.sh` before launching
a formal run on a new machine.
