# RLT Stage 2

VLA-RL implements RL-Token (RLT) Stage 2 as a first-class SERL-style example.
Stage 2 loads a frozen RLTokenEncoder checkpoint and trains only the RLT
actor/critic. The frozen VLA runs as a reference policy that returns reference
actions and prefix features.

## Data Flow

```text
LIBERO Observation
  -> ReferencePolicyClient.predict_actions_and_prefix
  -> base_actions[:chunk_size] + prefix_tokens + proprio
  -> encode_rlt_obs
  -> rlt_obs: z_rl + reference_action + proprio
  -> RLTAgent.sample_action
  -> action chunk with chunk_size actions
  -> env.step_chunk(actions)
  -> compact chunk replay transition
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

- `RLTAgent.sample_action()` outputs `(chunk_size, action_dim)`.
- RLT v0 intentionally has no separate `execute_horizon`: actor output, env execution, critic action input, replay action chunk, and BC target all use `chunk_size`.
- The actor stores one replay transition per executed chunk. This keeps the online path simple and avoids per-step VLA backfill.
- Terminal tails shorter than `chunk_size` carry `action_mask`; actor BC loss and actor Q loss ignore padded action dimensions.
- Chunk transition discount is `gamma ** executed_steps`; terminal chunks do not bootstrap.
- The actor loss is `-Q + bc_reg_coeff * masked_mse(action, reference_action)`.
- The actor/critic state currently uses `z_rl` only; `proprio` remains part of
  the explicit observation schema for future variants.

## Main Example

The canonical LIBERO RLT path is centered at:

```text
examples/libero_rlt/train.py
examples/libero_rlt/configs/libero_spatial_task4_openpi_rlt.yaml
examples/libero_rlt/tools/launch_rlt.sh
```

A 1000-step smoke run is:

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
  -- runtime.max_env_steps=1000 runtime.max_update_steps=1000
```

For long runs, remove the short-step overrides or set
`runtime.max_env_steps` / `runtime.max_update_steps` to the target budget.
