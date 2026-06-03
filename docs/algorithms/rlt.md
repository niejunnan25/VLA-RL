# RLT Stage 2

VLA-RL integrates RL-Token (RLT) Stage 2 as the first real algorithm path.
Stage 1 training is not migrated yet; Stage 2 loads a frozen RLTokenEncoder
checkpoint and trains only the RLT actor/critic.

## Data Flow

```text
LIBERO Observation
  -> OpenPIBackend.sample_actions / extract_features
  -> RLTFeatureProcessor
  -> agent_obs: z_rl + reference_action + proprio
  -> RLTAgent.act
  -> full action chunk
  -> runtime executes first execute_horizon actions
  -> replay Transition/CompactTransition(agent_obs, next_agent_obs, discount)
  -> RLTAgent.update
```

`OpenPIBackend` remains a base VLA adapter. It returns prefix/suffix VLA hidden
states and reference actions. RLT-specific `z_rl` construction happens in
`RLTFeatureProcessor`, not inside the OpenPI backend.

## Algorithm Semantics

- `RLTAgent.act()` outputs a full `(chunk_size, action_dim)` action chunk.
- The runtime executes only the first `execute_horizon` actions.
- Replay stores the full action chunk for critic/actor training.
- Transition discount is `gamma ** executed_steps`; terminal transitions do not
  bootstrap.
- The actor loss is `-Q + bc_reg_coeff * MSE(action, reference_action)`.
- The actor/critic state currently uses `z_rl` only; `proprio` is kept in
  `agent_obs` for compatibility and future variants.

## Smoke Recipe

```bash
python scripts/serve_libero_env.py \
  --serl-torch-root /vla/users/niejunnan/codebase/serl_torch-rlt-merge \
  --host 127.0.0.1 \
  --port 23000 \
  --gpu-id 0

python scripts/train.py --config recipes/libero_spatial_openpi_rlt_smoke.yaml
```

The default smoke recipe targets LIBERO spatial task 4, `pi0_libero`, and the
legacy Stage 1 encoder checkpoint:

```text
/vla/users/yixin/openpi/checkpoints/rlt_stage1_single_cos_l6/checkpoint_12000.pt
```

The Pi0 checkpoint used by the smoke recipe is the yixin checkpoint because it
contains the OpenPI assets/norm stats required by `create_trained_policy()`.

## Formal Single-Process Training

The single-process runtime is retained for debugging and local regression
checks. It adds:

- run directories with `config.yaml`, `metrics.jsonl`, and `summary.json`;
- `checkpoints/step_*.pt`, `checkpoints/latest.pt`, and `checkpoints/final.pt`;
- resume from a checkpoint, without replay-buffer persistence;
- synchronous evaluation on an independent LIBERO env server.

The default formal recipe is:

```bash
python scripts/train.py --config recipes/libero_spatial_task4_openpi_rlt_300k.yaml
```

By default it uses `127.0.0.1:23000` for training and `127.0.0.1:23001` for
evaluation. Start two `scripts/serve_libero_env.py` processes before running
the recipe. For short acceptance tests, edit a copy of the recipe or override
the YAML values to use `runtime.max_env_steps=200`, `runtime.max_update_steps=200`,
and `eval.enabled=false`.

Checkpoint resume restores the RLT actor, critics, target critics, optimizers,
and step counters. Replay is intentionally not saved in Milestone 3, so resumed
runs collect fresh online replay before continuing updates.

## Agentlace Training

Milestone 4 makes Agentlace the default real-training runtime. It splits RLT
Stage 2 into:

- learner: owns RLT actor/critic updates, compact replay, metrics, and
  checkpoints;
- actor: owns LIBERO rollout, OpenPI inference, RLT feature processing, and
  action execution.

The actor streams compact transitions containing `z_rl`, `next_z_rl`,
`reference_action`, `next_reference_action`, action chunk, reward, terminal
flags, `executed_steps`, and `discount`. Raw observations and images stay on
the actor side. Learner publishes actor-only weights, and actor updates its
local policy through `load_policy_state_dict()`.

Default recipe:

```bash
scripts/launch_agentlace_rlt.sh \
  --config recipes/libero_spatial_task4_openpi_rlt_agentlace.yaml \
  --actor-gpu 0 \
  --learner-gpu 1
```

For the full Agentlace runbook, environment split, port conventions, smoke
commands, and metrics checks, see `docs/runtime/agentlace_rlt.md`.
