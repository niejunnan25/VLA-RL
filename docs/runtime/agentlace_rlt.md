# Agentlace RLT Runbook

This runbook covers the RLT Stage-2 actor/learner split in VLA-RL. The training
loop is intentionally owned by the LIBERO example:

```text
examples/libero/rlt/scripts/train_stage2.py
examples/libero/rlt/tools/launch_rlt.sh
```

`vla_rl.runtime.agentlace` is only the transport layer: it imports Agentlace,
creates trainer configs, adapts replay to a data store, and sanitizes
metrics. It does not own RLT rollout or update semantics.

## Runtime Layout

- LIBERO env server: external process, launched through the existing LIBERO
  server script.
- Reference-policy server: external process, usually OpenPI, exposing
  `predict_action_with_features()`.
- Learner: owns RLT actor/critic updates, replay, checkpoints, and
  metrics.
- Actor: owns LIBERO rollout, reference-policy calls, RLT feature processing,
  and action execution.

The actor streams transitions containing `z_rl`, `next_z_rl`,
`reference_action`, `next_reference_action`, action chunk, reward, terminal
flags, `executed_steps`, and `discount`. Raw observations and images stay on the
actor side.

## Reference Policy Server

```bash
python scripts/serve_reference_policy.py \
  --policy openpi \
  --policy-root /vla/users/niejunnan/codebase/openpi-rlt-github \
  --config-name pi0_libero \
  --checkpoint-path /vla/users/niejunnan/assets/openpi-assets/checkpoints/pi0_libero_pytorch \
  --action-dim 7 \
  --device cuda \
  --host 127.0.0.1 \
  --port 8899
```

Recipes should use `vla_rl.policies.ReferencePolicyClient` with the matching
server URL. This keeps the actor independent of OpenPI, StarVLA, or any other
provider-specific Python environment.

## 1000-step Smoke

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/VLA-RL

rm -rf /tmp/vlarl_rlt_task4_smoke

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

## Metrics and Acceptance

The learner writes `metrics.jsonl` and `summary.json`. The actor writes
`actor_metrics.jsonl` and `actor_summary.json`.

For smoke and sanity checks, verify:

- `actor_summary.json.env_steps` reaches the requested env-step target.
- `actor_summary.json.received_policy_state` is `true`.
- `summary.json.update_steps` reaches the requested update target.
- `summary.json.replay_size > 0`.
- `checkpoints/latest.pt` and `checkpoints/final.pt` exist.
- `train/critic_loss`, `train/actor_loss`, and Q metrics are finite.
