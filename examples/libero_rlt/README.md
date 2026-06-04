# LIBERO RLT

This example is the main RLT Stage-2 training path in VLA-RL. It keeps the
training loop in VLA-RL and treats OpenPI or other VLAs as frozen reference
policy services.

## Data Flow

```text
obs -> ReferencePolicyClient.predict_actions_and_prefix()
    -> base_actions, prefix_tokens, proprio
    -> base_actions[:execute_horizon]
    -> encode_rlt_obs(prefix_tokens, base_actions, proprio)
    -> RLTAgent.sample_action(rlt_state) -> actions
    -> env.step_chunk(actions)
    -> compact replay -> RLTAgent.update
```

The reference-policy server must expose `predict_action_with_features()`, which
returns reference actions and prefix features in one call. The actor loop uses
the narrower `predict_actions_and_prefix()` client method so the RLT data path
does not pass around a generic feature object.

RLT optimizes exactly the action prefix that is executed in the environment:
`reference_action`, actor output, critic action input, and BC loss all use
`execute_horizon` actions.

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

The checked-in config and launcher defaults use local cluster paths for the
validated LIBERO server, OpenPI fork, OpenPI checkpoint, and an external RLT
Stage-1 encoder checkpoint. Override `--serl-torch-root`, `--policy-root`,
`--policy-checkpoint`, or `feature.encoder_path` when using a different machine
or a newly trained Stage-1 encoder.
