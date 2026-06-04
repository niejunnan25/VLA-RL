# VLA-RL Architecture

VLA-RL is the main training framework for sample-efficient frozen-VLA
reinforcement learning. It follows the SERL style: examples own the visible
training loop, common library code stays thin, and large VLA models remain
frozen reference-policy providers.

The project is built for methods such as RLT, PLD, residual RL, and related
small-head RL algorithms. The trainable parts live in VLA-RL; model repositories
such as OpenPI, StarVLA, and JoyRA expose only inference hooks.

## Boundaries

- Model side: OpenPI, StarVLA, JoyRA, and other VLA repositories run as frozen
  reference-policy services. They expose inference hooks such as
  `predict_action_with_features()` and do not own replay, actor/learner loops,
  checkpoints, or RL losses.
- Environment side: LIBERO, RobotWin, and real robot stacks live behind env
  services or thin env clients. VLA-RL consumes `Observation` and `step_chunk()`
  semantics instead of importing simulator internals into training loops.
- Algorithm side: `vla_rl.algorithms.*` contains trainable small heads, feature
  processors, critics, and update rules. Algorithm subpackages are imported
  directly; the root algorithm package intentionally exposes only stable base
  utilities.
- Example side: `examples/` owns experiment entrypoints and visible training
  loops. RLT, PLD, and future methods should each have their own example-local
  actor/learner loop instead of sharing a universal runner.
- Runtime side: `vla_rl.runtime` provides transport, checkpoint, and RPC helpers
  only. It is support code, not an orchestration framework.


## Public Interfaces

The root packages are intentionally small. They are the surfaces that examples
may depend on without pulling model or simulator repositories into the RL
process.

- `vla_rl.data`: stable schemas plus compact and mixed replay utilities. It may
  contain replay records and samplers, but not debug runners or algorithm loops.
- `vla_rl.policies`: training-side policy interfaces and
  `ReferencePolicyClient`. Service-side adapters such as OpenPI live in
  subpackages and are imported only by reference-policy servers.
- `vla_rl.envs`: env interfaces, fake envs, and remote env clients. Importing it
  must not require LIBERO, robosuite, MuJoCo, or other simulator packages.
- `vla_rl.algorithms`: base algorithm utilities only. RLT and PLD details are
  imported from their subpackages.
- `vla_rl.runtime`: checkpoint, Agentlace transport helpers, and HTTP/RPC
  utilities. It must not own rollout or learner semantics.

Current v0 algorithm subpackages are public at their own namespace level:

- `vla_rl.algorithms.rlt`: RLT actor, critic, RLToken encoder/decoder,
  feature processor, and update logic.
- `vla_rl.algorithms.pld`: residual action spec, PLD feature processor,
  actor/critic/encoder modules, PLD SAC update logic, and PLD offline replay
  helpers.

## Example-Local Code

Training scripts, launch scripts, and task recipes stay in examples. An example
is allowed to know about a concrete task, ports, GPUs, tmux layout, smoke
overrides, and the exact actor/learner sequence. Common code should move into
`vla_rl/` only when it is a stable primitive shared by multiple examples.

Use this rule when adding new methods:

- Add a method-specific example first.
- Keep its actor and learner loops visible in that example.
- Share only small, stable pieces such as schemas, clients, agents, replay
  records, checkpoint helpers, and transport helpers.
- Do not add root registries or a universal runner to make unrelated algorithms
  look the same.

## RLT Mainline

The RLT LIBERO example uses the following explicit data path:

```text
observation
  -> reference_policy.predict_action_with_features()
  -> PolicyFeatures(reference_actions, embeddings["prefix"])
  -> RLTFeatureProcessor(prefix) -> z_rl
  -> RLTAgent.act(z_rl, reference_actions) -> action chunk
  -> env.step_chunk(action_chunk[:execute_horizon])
  -> compact replay transition
  -> RLTAgent.update(batch)
```

The model-side interface is intentionally small:

```python
predict_action_with_features(observation, ...) -> {
    "actions": ...,
    "features": {"prefix": ...},
}
```

VLA-RL converts this to `PolicyFeatures` and keeps the RLT actor/critic update
inside the framework.

## PLD Mainline

PLD is a separate example, not a mode inside the RLT runner. It has its own
base-success collection, residual observation construction, offline/online replay
mix, and Cal-QL-style critic pretraining. It shares only stable primitives such
as policy clients, environment clients, replay records, and checkpoints.

## Design Rule

Do not rebuild RLinf-style universal orchestration here. When adding a method,
prefer a clear example-local training script over a new generic registry or
runner layer. Promote code to `vla_rl/` only after it has become a stable
primitive shared by more than one example.
