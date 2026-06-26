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

The canonical `examples/libero/rlt` path builds RLT observations explicitly in
the actor loop using `vla_rl.algorithms.rlt.encode_rlt_obs`. The reference-policy
client/server lives under `vla_rl.policies`, while trainable actor/critic code
lives under `vla_rl.algorithms.rlt`.

The model-side OpenPI hooks are `predict_action_with_features()` and
`predict_action_with_self_conditioned_features()`. The training-side
`ReferencePolicyClient.predict_actions_and_prefix()` returns base VLA actions,
prefix tokens, and proprio to the actor loop.

## Algorithm Semantics

- `RLTAgent.sample_action()` outputs `(chunk_size, action_dim)`.
- RLT v0 has no separate execution horizon: actor output, env execution, critic action input, replay action chunk, and BC target all use `chunk_size`.
- `subsample_stride` controls window replay density. With `chunk_size=10` and `subsample_stride=2`, each pair of adjacent chunks can produce windows at `0,2,4,6,8` using `action_chunk[p:] + next_action_chunk[:p]`.
- Terminal chunks only flush windows that can be constructed without crossing into a new episode.
- Chunk transition discount is `gamma ** chunk_size` for non-terminal windows; terminal chunks do not bootstrap.
- The actor loss is `-Q + bc_reg_coeff * masked_mse(action, reference_action)`.
- The actor/critic state currently uses `z_rl` only; `proprio` remains part of
  the explicit observation schema for future variants.

## Main Example

The canonical LIBERO RLT path is centered at:

```text
examples/libero/rlt/scripts/train_stage2.py
examples/libero/rlt/configs/libero_spatial_task4_openpi_rlt.yaml
examples/libero/rlt/tools/launch_rlt.sh
```

A 1000-step smoke run is:

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
  -- runtime.max_env_steps=1000
```

For long runs, remove the short-step override or set `runtime.max_env_steps` to the target budget.
