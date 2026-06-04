# VLA-RL

VLA-RL is a sample-efficient reinforcement-learning infrastructure for
post-training vision-language-action policies.

VLA-RL keeps large VLA models and simulation environments outside the RL
process. They run as external services, while VLA-RL owns the lightweight
actor/critic training loops, replay, checkpoints, and example recipes. RLT is
the first clean SERL-style training line; PLD and residual SAC are separate
example lines rather than variants hidden behind a universal runner.

## v0 Targets

- Policies: OpenPI, StarVLA
- Environments: LIBERO
- Algorithms: RLT, PLD, residual SAC

## Current Scope

- `vla_rl.data`: shared schemas passed between environments, policies,
  algorithms, replay, and runtime.
- `vla_rl.policies`: policy backend interfaces, fake policy, reference-policy client, and service-side reference-policy factory.
- `vla_rl.envs`: environment backend interfaces, fake environment, and LIBERO
  remote client.
- `vla_rl.algorithms`: algorithm interface and fake algorithm.
- `vla_rl.algorithms.rlt`: RLT Stage 2 actor/critic and frozen encoder feature
  processing.
- `vla_rl.algorithms.pld`: PLD Stage 1 residual action policy, residual
  observation processing, SAC updates, Cal-QL-style critic pretraining, and
  offline/online replay mixing.
- `vla_rl.runtime`: thin transport helpers, checkpointing, and HTTP/RPC utilities. Algorithm training loops live in examples.
- `examples/*/configs`: example-owned configuration files. Root `recipes/` is intentionally not a public entrypoint.

## Debug Smoke

The old single-process runner is now a debug example, not a framework-level
training entrypoint:

```bash
python examples/fake_debug/train.py --config examples/fake_debug/configs/fake_local.yaml
pytest -q
```

Real algorithm entrypoints live under their own examples.

## RLT Stage 2 SERL-Style Run

The default real-training path is the LIBERO RLT example. The actor and learner
loops live directly in `examples/libero_rlt/train.py`; Agentlace is used only as
transport for compact replay transitions and actor-weight broadcasts. The helper
starts a LIBERO env server, OpenPI reference-policy server, learner, and actor in
one tmux session:

```bash
examples/libero_rlt/tools/launch_rlt.sh \
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
examples/libero_rlt/tools/launch_rlt.sh \
  --session vlarl_rlt_task4_smoke \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --run-dir /tmp/vlarl_rlt_smoke \
  -- \
  runtime.max_env_steps=200 \
  runtime.max_update_steps=200
```

The actor sends compact RLT transitions to the learner rather than raw images.
The learner owns replay, updates, metrics, and checkpoints.

See `docs/runtime/agentlace_rlt.md` for the full RLT runbook and the exact
actor/learner data path.

## PLD Stage 1 / Residual RL

VLA-RL implements PLD Stage 1 only: residual RL on top of a frozen OpenPI base
policy. Stage 2 hybrid data collection and Stage 3 VLA SFT are intentionally
out of scope for this milestone.

Collect successful base-policy replay:

```bash
examples/libero_pld/collect_base_success_replay.py \
  --config examples/libero_pld/configs/libero_spatial_task4_openpi_pld.yaml \
  --target-successes 50
```

Run PLD through its example-local launcher:

```bash
examples/libero_pld/tools/launch_pld.sh \
  --session vlarl_pld_task4_smoke \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --run-dir /tmp/vlarl_pld_smoke \
  -- \
  runtime.max_env_steps=200 \
  runtime.max_update_steps=200 \
  runtime.calql_pretrain_steps=10 \
  runtime.training_starts=10
```

See `docs/algorithms/pld.md` for the PLD Stage 1 runbook and scope boundary.
The default PLD recipe follows the existing `serl_torch` PLD configs and uses
a frozen HuggingFace ResNet-18 image encoder, not torchvision.
