# LIBERO PLD

This example is the PLD Stage-1 / residual-RL line in VLA-RL. It is intentionally
separate from `examples/libero_rlt`: PLD has its own base-success collection,
residual observation construction, offline/online replay mix, and Cal-QL-style
critic pretraining.

The current implementation reuses the stable VLA-RL primitives for reference
policy access, LIBERO env access, compact replay, and checkpoints. It does not
run through the RLT example or a shared universal runner.

## Collect Base-Success Replay

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

python examples/libero_pld/collect_base_success_replay.py \
  --config examples/libero_pld/configs/libero_spatial_task4_openpi_pld.yaml \
  --target-successes 50
```

## Train

Run PLD with the example-local launcher:

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

bash examples/libero_pld/tools/launch_pld.sh \
  --session vlarl_pld_task4_smoke \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --run-dir /tmp/vlarl_pld_task4_smoke \
  -- runtime.max_env_steps=200 runtime.max_update_steps=200
```
