# RLT Stage 2

VLA-RL implements RL-Token (RLT) Stage 2 as a first-class SERL-style example.
Stage 2 loads a frozen RLTokenEncoder checkpoint and trains only the RLT
actor/critic. The frozen VLA runs as a reference policy that returns reference
actions and prefix features.

## Data Flow

```text
LIBERO Observation
  -> ReferencePolicyClient.predict_actions_and_prefix
  -> base_actions[:execute_horizon] + prefix_tokens + proprio
  -> encode_rlt_obs
  -> rlt_obs: z_rl + reference_action + proprio
  -> RLTAgent.sample_action
  -> action chunk with execute_horizon actions
  -> env.step_chunk(actions)
  -> compact replay transition
  -> RLTAgent.update
```

The canonical `examples/libero_rlt` path builds RLT observations explicitly in
the actor loop. `RLTStateBuilder` is kept only for fake/debug local runners.
The reference-policy client/server lives under `vla_rl.policies`, while
trainable actor/critic code lives under `vla_rl.algorithms.rlt`.

The model-side hook is:

```python
predict_actions_and_prefix(...)
```

It returns base VLA actions, prefix tokens, and proprio. Future StarVLA/JoyRA
integrations should expose the same semantics while RLT training continues to
keep the frozen reference policy separate from the trainable actor/critic.

## Algorithm Semantics

- `RLTAgent.sample_action()` outputs `(execute_horizon, action_dim)`.
- `reference_action`, actor output, critic action input, BC loss, replay action,
  and `env.step_chunk()` all use the same `execute_horizon` action prefix.
- `chunk_size` belongs to the frozen VLA reference chunk, not to the RLT
  actor/critic action space.
- Transition discount is `gamma ** executed_steps`; terminal transitions do not bootstrap.
- The actor loss is `-Q + bc_reg_coeff * MSE(action, reference_action)`.
- The actor/critic state currently uses `z_rl` only; `proprio` remains part of
  the explicit observation schema for future variants.

## Main Example

The canonical LIBERO RLT path is centered at:

```text
examples/libero_rlt/train.py
examples/libero_rlt/configs/libero_spatial_task4_openpi_rlt.yaml
examples/libero_rlt/tools/launch_rlt.sh
```

A 200-step smoke run is:

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

For long runs, remove the short-step overrides or set
`runtime.max_env_steps` / `runtime.max_update_steps` to the target budget.
