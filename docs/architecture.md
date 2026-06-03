# VLA-RL Architecture

VLA-RL is a thin bridge package for frozen VLA reference-policy inference. It is
not the long-term training framework for RLT, PLD, residual RL, or other
sample-efficient frozen-VLA RL methods. Training loops, replay semantics,
checkpointing, and experiment recipes should live in `serl_torch` examples.

## Boundaries

The intended boundaries are:

- Model repositories such as OpenPI, StarVLA, and JoyRA expose frozen policy
  capabilities, for example `predict_action_with_features()`. They do not own
  RL training.
- VLA-RL provides small client/server adapters and schema tests for reference
  actions and VLA features.
- `serl_torch` owns actor/learner loops, replay buffers, update logic, and
  benchmark recipes.

This keeps VLA dependencies isolated in their own Python environments while
preserving SERL-style explicit training scripts in the RL repository.

## Current Bridge Contract

For RLT, a reference-policy server returns:

```text
Observation
  -> reference_actions
  -> features["prefix"]
```

The training side turns this into:

```text
prefix features -> frozen RLTokenEncoder -> z_rl
reference_actions -> RLT actor condition and BC target
```

The bridge should stay minimal: it may know how to call OpenPI or another VLA
provider, but it should not add new generic runners, algorithm registries, or
replay systems.

## Deprecated Direction

Earlier local experiments added generic `Algorithm`, `Runner`, local
actor/learner, and Agentlace runtime abstractions inside VLA-RL. Those were useful
for smoke tests, but they should not continue as the main training stack. New RLT
and PLD work should be implemented in `serl_torch`, with VLA-RL used only where a
standalone reference-policy bridge is useful.
