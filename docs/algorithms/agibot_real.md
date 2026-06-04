# AgiBot Real-Robot Examples

VLA-RL now has two AgiBot real-robot examples:

- `examples/agibot_real/rlt`: RL-Token Stage 2 with critical-phase handover.
- `examples/agibot_real/pld`: residual SAC / PLD Stage 1.

Both examples use the shared `examples.agibot_real.service` environment wrapper. The default configs use `env.backend=fake` for smoke tests. Switch to `env.backend=local` only when the AgiBot robot service and reset procedure are ready.

## RLT data flow

`AgiBotEnvBackend -> OpenPI reference-policy server -> prefix/reference_action -> local RLTokenEncoder -> RLTAgent.sample_action -> env.step_chunk`.

When `critical_phase.enabled=true`, the actor executes the frozen VLA action before handover and sends replay only after the operator presses the critical key or the handover classifier crosses the configured threshold.

## PLD data flow

`AgiBotEnvBackend -> frozen reference policy -> base_actions -> PLDObservationBuilder -> PLDSACAgent.sample_action -> env.step_chunk`.

This is Stage-1 residual RL only. Processor backfill, raw rollout recycle, and async eval from the old serl_torch AgiBot code are intentionally not included in this first VLA-RL port.
