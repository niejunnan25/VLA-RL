# PLD Stage 1 / Residual RL

VLA-RL implements only the first stage of PLD: task-specific residual
reinforcement learning on top of a frozen OpenPI base policy. It does not
implement PLD Stage 2 hybrid data collection or Stage 3 VLA SFT distillation.

The residual policy predicts a bounded delta action and executes:

```text
final_action = base_action + alpha * residual_action
```

The faithful default uses `chunk_horizon=1`.

The default PLD recipe follows the HIL-SERL-style residual RL setup used as the
implementation reference: HuggingFace `transformers.ResNetModel`
(`microsoft/resnet-18`) with a frozen backbone, spatial learned embeddings, a
256-dim bottleneck, and a 64-dim projection for vector observations (`proprio`,
base action chunk, and `alpha`). It does not use `torchvision.models.resnet*`.
The small CNN encoder remains available for unit tests and lightweight
debugging only.

Install the optional PLD dependency in the learner environment if it is not
already present:

```bash
pip install -e ".[pld]"
```

## Offline Base-Success Replay

Start the LIBERO env server, then collect successful base-policy rollouts:

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3 \
  examples/libero/pld/scripts/collect_base_success_replay.py \
  --config examples/libero/pld/configs/libero_spatial_task4_openpi_pld.yaml \
  --target-successes 50
```

The output defaults to:

```text
outputs/pld_base_success/libero_spatial_task4_pi0_libero
```

Each transition stores PLD observations, final executed actions, rewards,
terminal flags, discounts, and Monte Carlo returns for Cal-QL-style critic
pretraining.

The PLD loader expects this VLA-RL Transition replay format. It does not load old
`serl_torch` prepared replay dictionaries directly.

## Training

PLD training lives in `examples/libero/pld/scripts/train.py`. The actor and learner
loops make base warmup, offline/online replay mixing, Cal-QL-style critic
pretraining, residual action selection, checkpointing, and replay
transport visible in the PLD example itself.

## Boundaries

This implementation does not depend on `serl_torch` at runtime. Prior
SERL-style PLD code is used only as a reference for algorithm semantics and
engineering style.
