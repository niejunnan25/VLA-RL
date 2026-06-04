# LIBERO RLT

This example is the main RLT Stage-2 training path in VLA-RL. The training loop lives in VLA-RL, while OpenPI or another VLA runs as a frozen reference-policy service.

## Data Flow

```text
obs -> ReferencePolicyClient.predict_actions_and_prefix()
    -> base_actions, prefix_tokens, proprio
    -> base_actions[:chunk_size]
    -> encode_rlt_obs(prefix_tokens, base_actions, proprio)
    -> RLTAgent.sample_action(rlt_state) -> actions
    -> env.step_chunk(actions)
    -> Transition -> ReplayBuffer -> RLTAgent.update
```

The reference-policy server must expose `predict_action_with_features()`, returning reference actions and prefix features in one call. The actor loop consumes the narrower `predict_actions_and_prefix()` client method.

RLT v0 uses a single `chunk_size`: actor output length, environment execution length, critic action input, replay action, and BC target all use the same action chunk. Terminal tails carry `action_mask` so padded action dimensions do not affect BC loss or actor Q loss.

## 1000-Step Smoke

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

bash examples/libero/rlt/tools/launch_rlt.sh \
  --session vlarl_rlt_task4_smoke \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --env-port 23200 \
  --policy-port 8899 \
  --trainer-port 5568 \
  --broadcast-port 5569 \
  --run-dir /tmp/vlarl_rlt_task4_smoke \
  -- \
  runtime.max_env_steps=1000 \
  runtime.max_update_steps=1000
```

For formal runs, remove the short-step overrides or set them to the target budget. The checked-in config and launcher defaults use local cluster paths for the validated LIBERO server, OpenPI fork, OpenPI checkpoint, and an external RLT Stage-1 encoder checkpoint. Override `--serl-torch-root`, `--policy-root`, `--policy-checkpoint`, or `feature.encoder_path` on another machine.

## Multi-Task Example

```bash
bash examples/libero/rlt/tools/launch_libero90_8tasks_example.sh \
  --dry-run \
  -- \
  runtime.max_env_steps=1000 \
  runtime.max_update_steps=1000
```

Remove `--dry-run` to start one tmux session per task.

## Evaluation

```bash
python examples/libero/rlt/scripts/eval_stage2.py \
  --config examples/libero/rlt/configs/libero_spatial_task4_openpi_rlt.yaml \
  --checkpoint /tmp/vlarl_rlt_task4_smoke/checkpoints/final.pt \
  --episodes 10 \
  --output-dir /tmp/vlarl_rlt_task4_eval
```

The evaluator writes `eval_summary.json` and `eval_episodes.jsonl`. Video saving is optional and disabled by default.

## Stage 1 Cache, Training, and Stage 2 Commands

Prepare the full LIBERO prefix-feature cache with OpenPI on 8 GPUs:

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3 \
  -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  examples/libero/rlt/scripts/prepare_stage1_cache.py \
  --config examples/libero/rlt/configs/stage1_libero_openpi_rlt.yaml \
  cache.output_dir=/vla/users/niejunnan/cache/vlarl/rlt_stage1/libero_openpi_prefix_512 \
  cache.batch_size=8 \
  cache.num_workers=2 \
  rlt.max_tokens=512
```

Train the Stage 1 RLToken encoder/decoder from the cached prefix features with 8-card DDP. `training.batch_size=16` is per GPU, so the global batch size is `16 * 8 = 128`:

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/vla/miniconda3/envs/serl_torch/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  examples/libero/rlt/scripts/train_stage1.py \
  --config examples/libero/rlt/configs/stage1_libero_openpi_rlt.yaml \
  training.source=cache \
  cache.input_dir=/vla/users/niejunnan/cache/vlarl/rlt_stage1/libero_openpi_prefix_512 \
  training.output_dir=/vla/users/niejunnan/outputs/vlarl/rlt_stage1/libero_openpi_prefix_512_ddp_v2 \
  training.steps=20000 \
  training.batch_size=16 \
  training.save_every=2000 \
  training.log_every=50 \
  training.lr=4.0e-4
```

Run Stage 2 online RLT with the Stage 1 checkpoint:

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

bash examples/libero/rlt/tools/launch_rlt.sh \
  --session vlarl_rlt_stage2_task4 \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --env-gpu 0 \
  --policy-gpu 0 \
  --env-port 23210 \
  --policy-port 8909 \
  --trainer-port 5588 \
  --broadcast-port 5589 \
  --run-dir /tmp/vlarl_rlt_stage2_task4 \
  --python /vla/miniconda3/envs/serl_torch/bin/python \
  -- \
  feature.encoder_path=/vla/users/niejunnan/outputs/vlarl/rlt_stage1/libero_openpi_prefix_512_ddp_v2/final_model.pt \
  runtime.max_env_steps=1000 \
  runtime.max_update_steps=1000 \
  runtime.checkpoint_period=2000
```

The cache stores prefix features as `float16` on disk. Stage 1 training moves cached prefixes to GPU as `float32`, matching the original online Stage 1 path where `_truncate_prefix(...)` also converts OpenPI prefix features to `float32`. Distributed Stage 1 defaults to `fp32` math because `bf16` autocast with DDP is unstable on the current 225 environment; single-GPU cache training can still use `bf16` autocast by default.
