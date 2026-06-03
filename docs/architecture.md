# VLA-RL Architecture

VLA-RL is organized around four replaceable boundaries:

- `PolicyBackend`: wraps a base VLA policy such as OpenPI or StarVLA.
- `EnvBackend`: wraps robot environments such as LIBERO.
- `Algorithm`: implements action selection and learning.
- `Runner`: coordinates environment rollout, policy features, replay data, and
  learner updates.

The system center is the interface contract, not any specific policy,
environment, or algorithm.

## v0 Scope

The first real implementation target is:

- Policies: OpenPI, StarVLA
- Environment: LIBERO
- Algorithms: RLT, PLD, residual SAC

Milestone 1 keeps fake components for deterministic tests, and adds the first
real boundaries:

- `OpenPIBackend` for Torch OpenPI policies.
- `LiberoRemoteEnvBackend` for LIBERO over HTTP RPC.
- `LocalActorLearnerRunner` plus in-memory replay for single-process smoke
  tests.
- `RLTAgent` plus `RLTFeatureProcessor` for the first real algorithm path.

The LIBERO server itself remains external for now. `scripts/serve_libero_env.py`
delegates to the validated server in `serl_torch-rlt-merge`, keeping VLA-RL's
first version focused on framework contracts rather than environment vendoring.

## Data Flow

```text
EnvBackend.reset/step
  -> Observation
  -> PolicyBackend.sample_actions / extract_features
  -> optional FeatureProcessor
  -> Algorithm.act
  -> ActionChunk
  -> EnvBackend.step
  -> Transition
  -> ReplayBuffer
  -> Algorithm.update
```

All components exchange shared schema objects from `vla_rl.data.schema`.

## Runtime Levels

- `LocalRunner`: minimal one-transition-at-a-time loop for fake smoke tests.
- `LocalActorLearnerRunner`: local replay-backed loop that executes action
  chunks through `EnvBackend.step_chunk`, records `Transition.discount` as
  `gamma ** executed_steps`, updates the algorithm from replay, writes run
  artifacts, checkpoints algorithm state, and can run synchronous evaluation on
  a separate environment backend.

Distributed actor/learner, async eval, checkpoint servers, and algorithm-specific
distributed training loops are intentionally deferred to later milestones.
