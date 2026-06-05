# LIBERO RLT Stage 1

Stage 1 trains the frozen-VLA feature bottleneck used by RLT Stage 2. It is an offline prefix-token reconstruction job:

```text
OpenPI LIBERO dataloader
  -> frozen OpenPI predict_action_with_features(...)
  -> features["prefix"]
  -> RLTokenEncoder
  -> RLTokenDecoder
  -> MSE(prefix_reconstruction, prefix)
```

Only `RLTokenEncoder` and `RLTokenDecoder` receive gradients. The OpenPI policy is loaded from an external checkout and remains frozen.

## Output Contract

`examples/libero/rlt/scripts/train_stage1.py` writes checkpoints that Stage 2 can load directly through `feature.encoder_path`:

```text
encoder_state_dict
decoder_state_dict
optimizer_state_dict
scheduler_state_dict
global_step
config.rlt.input_dim
config.rlt.rl_token_dim
config.rlt.num_encoder_layers
config.rlt.num_decoder_layers
config.rlt.num_heads
config.rlt.ff_dim
config.rlt.dropout
config.rlt.max_tokens
```

The main artifact is `final_model.pt`. `rlt_encoder.pt` is also written for lightweight inspection.

## Smoke Command

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3 \
  examples/libero/rlt/scripts/train_stage1.py \
  --config examples/libero/rlt/configs/stage1_libero_openpi_rlt.yaml \
  vla.openpi_root=/vla/users/niejunnan/codebase/openpi-rlt-github \
  vla.lerobot_home=/vla/users/niejunnan/datasets \
  training.output_dir=/tmp/vlarl_rlt_stage1_smoke \
  training.steps=20 \
  training.batch_size=1 \
  training.num_workers=0
```

This 20-step command only validates dataset loading, prefix feature extraction, loss computation, and checkpoint compatibility. It does not produce a useful encoder.

The default `vla.num_steps=1` is intentional for Stage 1. Prefix features are computed before OpenPI action denoising, so extra denoising steps only add cost. Do not set it to `0`: the OpenPI action sampler computes `dt = -1 / num_steps`, and a zero-step action would also make the `predict_action_with_features()` API ambiguous. Stage 2 should still use normal reference-policy inference settings.

## Full Online Stage 1 Run

For full Stage 1 runs, train the encoder/decoder directly from frozen OpenPI prefix features. This path keeps the OpenPI policy frozen and does not write an intermediate prefix cache.

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3 \
  -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  examples/libero/rlt/scripts/train_stage1.py \
  --config examples/libero/rlt/configs/stage1_libero_openpi_rlt.yaml \
  training.output_dir=/vla/users/niejunnan/outputs/vlarl/rlt_stage1/libero10_scene6_ours_online_ddp \
  training.steps=20000 \
  training.batch_size=16 \
  training.save_every=2000 \
  training.log_every=50 \
  training.lr=4.0e-4
```

## Stage 2 Smoke With New Checkpoint

```bash
bash examples/libero/rlt/tools/launch_rlt.sh \
  --session vlarl_rlt_stage1_to_stage2_smoke \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --env-gpu 0 \
  --policy-gpu 0 \
  --env-port 23210 \
  --policy-port 8909 \
  --trainer-port 5588 \
  --broadcast-port 5589 \
  --run-dir /tmp/vlarl_rlt_stage1_to_stage2_smoke \
  --python /vla/miniconda3/envs/serl_torch/bin/python \
  -- \
  feature.encoder_path=/tmp/vlarl_rlt_stage1_smoke/final_model.pt \
  runtime.max_env_steps=1000 \
  runtime.max_update_steps=1000
```
