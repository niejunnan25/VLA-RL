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
python examples/libero/rlt/scripts/eval.py \
  --config examples/libero/rlt/configs/libero_spatial_task4_openpi_rlt.yaml \
  --checkpoint /tmp/vlarl_rlt_task4_smoke/checkpoints/final.pt \
  --episodes 10 \
  --output-dir /tmp/vlarl_rlt_task4_eval
```

The evaluator writes `eval_summary.json` and `eval_episodes.jsonl`. Video saving is optional and disabled by default.
