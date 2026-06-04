# VLA-RL v0 Runbook

This document records the v0 operating model for VLA-RL. The goal is a small,
SERL-style framework for sample-efficient RL with frozen VLA policies. VLA-RL
trains small heads; it does not train or import the large VLA model.

## Why VLA Is an External Service

OpenPI, StarVLA, JoyRA, and future VLA repositories have incompatible model
dependencies, checkpoints, and feature extraction hooks. VLA-RL treats them as
reference-policy services instead of Python libraries imported by the learner.

The service side owns:

- model loading and checkpoint format;
- model-specific preprocessing and normalization;
- GPU placement for the frozen VLA;
- model hooks such as `predict_action_with_features()`.

The training side sees only `PolicyFeatures` through `ReferencePolicyClient`.
This keeps the actor and learner environment small and lets a new VLA be added
by writing a service adapter, not by changing the RL training loop.

## Why Env Is an External Service

Simulation and robot stacks also have incompatible dependencies. LIBERO,
RobotWin, IsaacLab, and real-robot stacks should live behind env services or
thin remote clients. VLA-RL consumes a stable interface:

```text
reset() -> Observation
step(action) -> Observation, reward, done, truncated, info
step_chunk(actions) -> Observation, reward, done, truncated, info
```

The RL process should not import simulator internals, rendering backends, ROS,
MuJoCo, or task-specific environment packages unless the example explicitly
runs a local debug env.

## RLT Data Flow

RLT Stage 2 trains a small actor/critic on top of a frozen VLA and a frozen
RLToken encoder.

```text
obs
  -> ReferencePolicyClient.predict_action_with_features(obs)
  -> PolicyFeatures(reference_actions, embeddings["prefix"], proprio)
  -> RLTFeatureProcessor(prefix) -> z_rl
  -> RLTAgent.act(z_rl, reference_actions) -> action_chunk
  -> env.step_chunk(action_chunk[:execute_horizon])
  -> CompactTransition(obs_features, action_chunk, reward, next_features, done)
  -> CompactReplayBuffer.sample()
  -> RLTAgent.update(batch)
```

The example owns the actor and learner loops in `examples/libero_rlt/train.py`.
`vla_rl.algorithms.rlt` owns only the trainable heads, feature processor, and
update logic.

## PLD Data Flow

PLD Stage 1 trains a residual policy on top of a frozen VLA reference action.

```text
obs
  -> ReferencePolicyClient.sample_actions(obs) -> base_action
  -> PLDFeatureProcessor(obs, base_action, alpha) -> residual_obs
  -> PLDSACAgent.act(residual_obs) -> residual_action
  -> final_action = base_action + alpha * residual_action
  -> env.step_chunk(final_action)
  -> CompactTransition(residual_obs, residual_action, reward, next_residual_obs, done)
  -> online/offline replay mix
  -> PLDSACAgent.update(batch)
```

Formal PLD uses successful base-policy replay for offline replay and
Cal-QL-style critic pretraining. The example owns this sequence in
`examples/libero_pld/train.py`; the public PLD package owns only the residual
agent, networks, feature construction, and offline replay helpers.

## Formal Runs vs Connectivity Smokes

A formal experiment should follow the method recipe. For PLD this means first
collecting base-success replay, then enabling offline replay and Cal-QL-style
critic pretraining. For RLT this means using a real Stage 1 RLToken encoder
checkpoint and the intended train/eval budget.

A connectivity smoke only checks that services and training loops talk to each
other. It may use small batch sizes, short budgets, disabled async eval, and no
offline replay. A PLD connectivity smoke should set:

```text
runtime.require_offline=false
runtime.offline_replay_path=null
runtime.calql_pretrain_steps=0
runtime.training_starts=1
runtime.batch_size=1
```

Smoke results verify plumbing, not sample efficiency or paper-level behavior.

## Public vs Example-Local

Public VLA-RL code is stable code that more than one example can depend on:

- data schemas and replay records;
- reference-policy and env clients;
- RLT/PLD agents, networks, feature processors, and update logic;
- checkpoint and transport helpers.

Example-local code is allowed to be task- and method-specific:

- actor and learner loops;
- launch scripts and tmux layouts;
- ports, GPU placement, paths, and smoke overrides;
- task configs and runbook commands.

If a piece of code is only needed by one algorithm, keep it in that algorithm's
example until it becomes a stable shared primitive.

## Adding StarVLA, JoyRA, or DSRL

Add new model support by implementing the model-side reference-policy service
hook, then connect it through `ReferencePolicyClient`. Do not import the model
repository into the actor or learner process.

Add a new algorithm by creating a new example with visible actor and learner
loops. Promote only the small heads, feature processors, replay schemas, or
update logic that are stable enough to share. Avoid root registries and a
universal runner.
