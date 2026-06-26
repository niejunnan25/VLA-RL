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

The reference-policy server must expose `predict_action_with_features()` for policy-prior features. For self-conditioned RLT experiments, the OpenPI server also exposes `predict_action_with_self_conditioned_features()`. The actor loop consumes the narrower `predict_actions_and_prefix()` client method and passes the configured `feature_source` through the reference-policy client.

## Feature Sources

`feature.source` controls Stage 1 feature extraction; `feature.online_source` controls Stage 2 online inference. With `online_source: auto`, the online path follows `source`, except `expert_conditioned_prefix` maps to `self_conditioned_prefix` because online rollout has no expert action.

| name | Stage 1 condition | Stage 2 online path |
| --- | --- | --- |
| `policy_prior_prefix` | normal VLA inference prefix, no action condition | same |
| `self_conditioned_prefix` | sample reference action, then joint forward with that reference action | same |
| `expert_conditioned_prefix` | dataset expert/demo action joint forward | maps to `self_conditioned_prefix` under `auto` |

`rlt.max_tokens: 512` uses the first 512 prefix tokens; `rlt.max_tokens: null` uses the full prefix. Stage 2 first follows the Stage 1 checkpoint's recorded `config.rlt.max_tokens`; YAML `rlt.max_tokens` is only the fallback for legacy checkpoints.

Main experiment names:

```text
policy_prior_prefix_512
policy_prior_prefix_full
self_conditioned_prefix_512
self_conditioned_prefix_full
```

`expert_conditioned_prefix_512` is a yixin-style Stage 1/offline baseline, not an online rollout feature source.

RLT v0 uses `chunk_size` as the action chunk length: actor output, environment execution, critic action input, replay action, and BC target all use the same action chunk. `subsample_stride` controls window replay density; for example `chunk_size=10, subsample_stride=2` yields windows at `0,2,4,6,8` using `action_chunk[p:] + next_action_chunk[:p]`. Terminal chunks only flush windows that can be constructed within the episode.

## Reward Models

The default reward path is the environment sparse reward:

```yaml
reward:
  type: sparse
  source: env
```

Remote reward models are exposed to RLT as absolute progress services. The RLT
actor submits completed chunk transitions to an async reward worker, and the
worker commits only fully relabeled transitions to learner replay. This keeps
the learner from seeing placeholder rewards while avoiding step-level VLA
backfill. The first chunk queries both the start and end boundary; later
chunks send only the latest boundary. Disable window replay when using remote progress rewards:

```yaml
rlt:
  subsample_stride: 0

reward:
  type: env_plus_potential_delta
  source: remote_progress
  scale: 1.0
  initial_progress: query_start
  remote:
    url: http://127.0.0.1:50052
    method: predict_progress
    timeout: 120.0
  async:
    max_pending_chunks: 64
  trajectory:
    image_keys: [image_rgb_0, image_rgb_1]
```

Supported reward transforms are `sparse`, `progress_abs`, `progress_delta`,
`potential_delta`, and `env_plus_potential_delta`. Potential-based rewards use
the transition discount consumed by the learner. Actor metrics report
`submitted_transitions` for chunks handed to the reward processor and
`replay_transitions` for transitions actually committed to learner replay.

## 1000-Step Smoke

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

bash examples/libero/rlt/tools/launch_rlt.sh \
  --session vlarl_rlt_task4_smoke \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --policy-port 8899 \
  --trainer-port 5568 \
  --broadcast-port 5569 \
  --run-dir /tmp/vlarl_rlt_task4_smoke \
  -- \
  runtime.max_env_steps=1000
```

For formal runs, remove the short-step overrides or set them to the target budget. The checked-in config and launcher defaults use the local LIBERO backend in `/vla/users/niejunnan/envs/serl_torch`, the OpenPI fork, OpenPI checkpoint, and an external RLT Stage-1 encoder checkpoint. Override `--policy-root`, `--policy-checkpoint`, or `feature.encoder_path` on another machine.

## Multi-Task Example

```bash
bash examples/libero/rlt/tools/launch_libero90_8tasks_example.sh \
  --dry-run \
  -- \
  runtime.max_env_steps=1000
```

Remove `--dry-run` to start one tmux session per task.

## Evaluation

```bash
python examples/libero/rlt/scripts/eval_stage2.py \
  --config examples/libero/rlt/configs/reward_model/libero_spatial_task4_self_cond_512_stage2.yaml \
  --checkpoint /tmp/vlarl_rlt_task4_smoke/checkpoints/final.pt \
  --episodes 10 \
  --output-dir /tmp/vlarl_rlt_task4_eval
```

The evaluator writes `eval_summary.json` and `eval_episodes.jsonl`. Video saving is optional and disabled by default.

### Train With Async Eval

`launch_rlt.sh --with-eval` starts a dedicated eval policy service in the same tmux session. With the default local LIBERO backend, the eval worker creates its own in-process env. The learner queues eval checkpoints every `runtime.async_eval.every_episodes` completed training episodes. A separate eval worker writes `eval_summary.jsonl` plus `eval_runs/*`.

```bash
bash examples/libero/rlt/tools/launch_rlt.sh \
  --session vlarl_rlt_task6_eval \
  --config examples/libero/rlt/configs/reward_model/libero_spatial_task6_self_cond_512_stage2.yaml \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --policy-gpu 0 \
  --with-eval \
  --eval-gpu 2 \
  --policy-port 8964 \
  --eval-policy-port 9064 \
  --trainer-port 5667 \
  --broadcast-port 5668 \
  --run-dir /vla/users/niejunnan/codebase/VLA-RL/outputs/rlt_task6_eval \
  --python /vla/users/niejunnan/envs/serl_torch/bin/python \
  --policy-python /vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3 \
  --policy-root /vla/users/niejunnan/codebase/openpi-rlt-github \
  --policy-config pi0_libero \
  --policy-checkpoint /vla/users/niejunnan/assets/openpi-assets/checkpoints/pi0_libero_pytorch \
  -- \
  runtime.async_eval.every_episodes=50 \
  runtime.async_eval.episodes=50 \
  runtime.max_env_steps=600000
```

## Stage 1 and Stage 2 Commands

Train the Stage 1 RLToken encoder/decoder online from frozen OpenPI prefix features. This path loads OpenPI in the OpenPI-capable Python environment and does not use a prefix cache:

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3 \
  -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  examples/libero/rlt/scripts/train_stage1.py \
  --config examples/libero/rlt/configs/libero_10_task6_self_cond_512_stage1.yaml \
  training.output_dir=/vla/users/niejunnan/outputs/vlarl/rlt_stage1/libero10_scene6_ours_online_ddp \
  training.steps=20000 \
  training.batch_size=16 \
  training.save_every=2000 \
  training.log_every=50 \
  training.lr=4.0e-4 \
  feature.source=policy_prior_prefix \
  feature.online_source=auto \
  rlt.max_tokens=512
```

Run Stage 2 online RLT with the Stage 1 checkpoint:

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

bash examples/libero/rlt/tools/launch_rlt.sh \
  --session vlarl_rlt_stage2_task4 \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --policy-gpu 0 \
  --policy-port 8909 \
  --trainer-port 5588 \
  --broadcast-port 5589 \
  --run-dir /tmp/vlarl_rlt_stage2_task4 \
  --python /vla/users/niejunnan/envs/serl_torch/bin/python \
  -- \
  feature.encoder_path=/vla/users/niejunnan/outputs/vlarl/rlt_stage1/libero10_scene6_ours_online_ddp/final_model.pt \
  feature.source=policy_prior_prefix \
  feature.online_source=auto \
  runtime.max_env_steps=1000 \
  runtime.checkpoint_period=2000
```
