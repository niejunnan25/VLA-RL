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
- model hooks such as `predict_actions_and_prefix()` or
  `predict_action_with_features()`.

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
  -> ReferencePolicyClient.predict_actions_and_prefix(obs)
  -> base_actions[:chunk_size] + prefix_tokens + proprio
  -> encode_rlt_obs(prefix_tokens, base_actions, proprio) -> rlt_state
  -> RLTAgent.sample_action(rlt_state) -> actions
  -> env.step_chunk(actions)
  -> Transition(obs=learner_obs, action=actions, reward=reward, next_obs=next_learner_obs, done=done)
  -> ReplayBuffer.sample()
  -> RLTAgent.update(batch)
```

The example owns the actor and learner loops in `examples/libero/rlt/scripts/train_stage2.py`.
`vla_rl.algorithms.rlt` owns the trainable heads and update logic.
`RLTStateBuilder` remains only for fake/debug local runners; the formal LIBERO
path uses the explicit `encode_rlt_obs` flow above.

## PLD Data Flow

PLD Stage 1 trains a residual policy on top of a frozen VLA reference action.

```text
obs
  -> ReferencePolicyClient.sample_actions(obs) -> base_actions
  -> PLDObservationBuilder(obs, base_actions, alpha) -> pld_obs
  -> PLDSACAgent.sample_action(pld_obs) -> final_actions
  -> final_actions = base_actions + alpha * residual_actions
  -> env.step_chunk(final_actions)
  -> Transition(obs=pld_obs, action=final_actions, reward=reward, next_obs=next_pld_obs, done=done)
  -> online/offline replay mix
  -> PLDSACAgent.update(batch)
```

Formal PLD uses successful base-policy replay for offline replay and
Cal-QL-style critic pretraining. The example owns this sequence in
`examples/libero/pld/scripts/train.py`; the public PLD package owns only the residual
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

## W&B Logging

VLA-RL follows the lightweight HIL-SERL logging pattern. The learner owns the
single W&B-compatible run. Actor metrics are sent to the learner with the same
Agentlace `send-stats` request used for local JSONL metrics, and the learner
uploads them together with learner update metrics and timer diagnostics.

Local files are always written first:

- learner: `metrics.jsonl`, `summary.json`, `checkpoints/`;
- actor: `actor_metrics.jsonl`, `actor_summary.json`.

W&B logging is enabled by default in formal LIBERO RLT/PLD configs. Internally,
VLA-RL tries SwanLab first, then falls back to native W&B if SwanLab is not
installed. Formal online runs should install the logging extra:

```bash
pip install -e ".[wandb]"
```

Override the run name or project with:

```bash
wandb.project=vla-rl \
wandb.exp_name=task4_rlt_seed0
```

Disable online upload for local smoke/debug runs with:

```bash
wandb.mode=disabled
```

The uploaded payload is intentionally small and mirrors HIL-SERL: learner
update metrics (`train/*`), timer averages (`timer/*`), and actor episode
environment summaries (`environment/episode/*`). VLA-RL does not upload replay
sizes, speed diagnostics, observations, replay batches, raw actions, videos, or
model-specific large arrays. Those richer diagnostics remain available in the
local JSONL files.

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
