# VLA-RL Architecture

VLA-RL is the main training framework for sample-efficient frozen-VLA
reinforcement learning. It follows the SERL style: examples own the visible
training loop, common library code stays thin, and large VLA models remain
frozen reference-policy providers.

The project is built for methods such as RLT, PLD, residual RL, and related
small-head RL algorithms. The trainable parts live in VLA-RL; model repositories
such as OpenPI, StarVLA, and JoyRA expose only inference hooks.

## Boundaries

- Model repositories expose frozen policy capabilities such as
  `predict_action_with_features()`. They do not own replay, actor/learner loops,
  checkpoints, or RL losses.
- `vla_rl.policies` contains reference-policy clients and lightweight wrappers
  around those external VLA providers. This is the boundary between model
  environments and the RL framework.
- `vla_rl.algorithms` contains only trainable algorithm logic: small actors,
  critics, feature processors, and update rules.
- `vla_rl.data`, `vla_rl.envs`, and `vla_rl.runtime` provide stable primitives
  for replay records, environment adapters, transport, and checkpoints. Runtime
  helpers are support code, not a universal runner abstraction.
- `examples/` contains the real experiment recipes. Each algorithm gets a
  self-contained example whose training flow can be read without chasing a
  generic framework stack.

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
