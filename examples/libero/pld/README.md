# LIBERO PLD

This example is the PLD Stage-1 / residual-RL line in VLA-RL. It is separate from `examples/libero/rlt`: PLD has its own base-success collection, residual observation construction, offline/online replay mix, and Cal-QL-style critic pretraining.

The implementation reuses stable VLA-RL primitives for reference-policy access, LIBERO env access, replay, and checkpoints. It does not run through the RLT example or a shared universal runner.

PLD now uses `vla_rl.envs.libero.LiberoLocalEnvBackend` by default, so LIBERO runs in the actor / collector / evaluator process with `/vla/users/niejunnan/envs/serl_torch/bin/python`. It does not launch a separate LIBERO env RPC server. The OpenPI reference policy is still served through RPC because its dependency stack is intentionally isolated from the training environment.

## Collect Base-Success Replay

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

/vla/users/niejunnan/envs/serl_torch/bin/python examples/libero/pld/scripts/collect_base_success_replay.py \
  --config examples/libero/pld/configs/libero_spatial_task4_openpi_pld.yaml \
  --target-successes 50
```

For a fast connectivity check, use `--target-successes 1`. Formal PLD runs should collect the intended base-success replay before training.

## Train

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

bash examples/libero/pld/tools/launch_pld.sh \
  --session vlarl_pld_task4_smoke \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --run-dir /tmp/vlarl_pld_task4_smoke \
  -- \
  runtime.max_env_steps=1000
```

## Collect Then Train

```bash
bash examples/libero/pld/tools/launch_pld_after_collect.sh \
  --session vlarl_pld_task4_after_collect \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --target-successes 50 \
  --run-dir outputs/libero_spatial_task4_openpi_pld
```

For a connectivity smoke, set `--target-successes 1` and override `runtime.max_env_steps=1000`. Formal PLD runs should keep offline replay and Cal-QL-style pretraining enabled.

## Evaluation

```bash
/vla/users/niejunnan/envs/serl_torch/bin/python examples/libero/pld/scripts/eval.py \
  --config examples/libero/pld/configs/libero_spatial_task4_openpi_pld.yaml \
  --checkpoint /tmp/vlarl_pld_task4_smoke/checkpoints/final.pt \
  --episodes 10 \
  --output-dir /tmp/vlarl_pld_task4_eval
```

The evaluator writes success, return, length, and residual-magnitude summaries. Video saving is optional and disabled by default.

The checked-in config and launcher defaults use local cluster paths for the validated LIBERO environment, OpenPI fork, OpenPI checkpoint, and ResNet checkpoint. Override `env.serl_torch_root`, `--policy-root`, `--policy-checkpoint`, or matching OmegaConf fields on another machine.
