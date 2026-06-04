# LIBERO RLT

This example is the main RLT Stage-2 training path in VLA-RL. It keeps the
training loop in VLA-RL and treats OpenPI or other VLAs as frozen reference
policy services.

## Data Flow

```text
obs -> ReferencePolicyClient.extract_features()
    -> PolicyFeatures(reference_actions, embeddings["prefix"])
    -> RLTFeatureProcessor -> z_rl
    -> RLTAgent.act -> action chunk
    -> env.step_chunk(action_chunk[:execute_horizon])
    -> compact replay -> RLTAgent.update
```

The reference-policy server must expose `predict_action_with_features()`, which
returns reference actions and prefix features in one call.

## Smoke Command

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

bash examples/libero_rlt/tools/launch_rlt.sh \
  --session vlarl_rlt_task4_smoke \
  --actor-gpu 0 \
  --learner-gpu 1 \
  --env-port 23200 \
  --policy-port 8899 \
  --trainer-port 5568 \
  --broadcast-port 5569 \
  --run-dir /tmp/vlarl_rlt_task4_smoke \
  -- runtime.max_env_steps=200 runtime.max_update_steps=200
```

For long runs, remove the short-step overrides or set them to the target budget.
