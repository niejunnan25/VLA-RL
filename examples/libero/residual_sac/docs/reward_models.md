# Reward Model Adapters

Residual SAC treats reward models as progress services. The actor sends rollout
frames and task metadata to a local HTTP adapter. The adapter returns absolute
progress values. The training code then applies the reward transform specified
in YAML.

## Common Interface

The training side expects:

```text
predict_progress(request) -> progress values
```

The returned values are absolute progress estimates. The configured transform
decides how to turn them into replay rewards.

For the current formal PBRS configs:

```yaml
reward:
  transform: env_plus_potential_delta
  discount: ${sac.discount}
```

This keeps model-specific inference details outside the SAC algorithm.

## Robo-Dopamine

Launcher path:

```text
examples/libero/residual_sac/tools/serve_robodopamine_progress.sh
```

Important variables:

| Variable | Meaning |
|---|---|
| `REWARD_MODEL_REPO` / `ROBODOPAMINE_ROOT` | Robo-Dopamine repository. |
| `REWARD_MODEL_CONDA_ENV` | Robo-Dopamine Python environment. |
| `REWARD_MODEL_PATH` | GRM checkpoint, usually `Robo-Dopamine-GRM-2.0-4B-Preview`. |
| `REWARD_GOAL_DATASET` | LIBERO demo dataset used to fetch goal images. |
| `REWARD_BATCH_SIZE` | Progress batch size passed to the service. |

The LIBERO adapter uses two-view inputs for Robo-Dopamine and copies the main
view when a third view is required by the model-side format. The reward model
returns absolute progress; PBRS is applied in VLA-RL.

## RoboMeter

Launcher path:

```text
examples/libero/residual_sac/tools/serve_robometer_progress_stack.sh
```

Important variables:

| Variable | Meaning |
|---|---|
| `ROBOMETER_ROOT` | RoboMeter repository. |
| `ROBOMETER_MODEL_PATH` | RoboMeter checkpoint, usually `Robometer-4B`. |
| `ROBOMETER_PYTHON_CMD` | Python command used to run RoboMeter. |
| `ROBOMETER_BACKEND` | `native` is recommended for training latency. |
| `FORWARD_BATCH_SIZE` | RoboMeter forward batch size. |
| `MAX_HISTORY_FRAMES` | Maximum prefix frames sampled per query. |

For LIBERO training the current adapter uses:

```text
BACKEND=native
QUERY_MODE=prefix_per_query
VIEW_MODE=first
ROBOMETER_IMAGE_KEYS=image_rgb_0
MAX_HISTORY_FRAMES=8
```

`prefix_per_query` means each queried timestep gets its own prefix sample, and
the adapter batches those samples for one model forward. This matches the
RoboMeter policy-learning style while avoiding the repeated full-history HTTP
transfer used by older two-layer adapter paths.

## Why Services Are Separate

Robo-Dopamine, RoboMeter, OpenPI, and LIBERO have different dependency stacks.
Keeping them as services gives three practical benefits:

1. The actor/learner environment remains small and stable.
2. Reward model failures are visible in service logs and do not silently change
   the SAC update code.
3. GPU placement can be adjusted independently for policy, reward, learner, and
   environment processes.

## Reward Transforms

Current formal reward-model configs use PBRS:

```text
env_plus_potential_delta
```

Sparse-only configs use:

```text
env_only
```

Other transforms may be useful for ablation, but they should be represented by
separate YAML files so each SwanLab run remains traceable.

## Latency Expectations

Reward-model latency should not block actor rollout. Healthy formal runs have
actor speeds above `10 env/s`; RoboMeter native adapter latency is typically a
few hundred milliseconds per progress request, and the async reward path should
keep the actor moving.

If actor speed drops:

1. Check `services/reward_model.log` for request latency and errors.
2. Check whether reward model and OpenPI policy are sharing a busy GPU.
3. Check whether actor/env/policy placement is abnormal for that machine.
4. Check queue backlog and replay commit timing before changing SAC settings.

## Failure Policy

Formal YAMLs set `reward.fail_on_error: true`. A reward service failure should
fail fast rather than silently falling back to sparse reward. This is intentional
for benchmark runs.
