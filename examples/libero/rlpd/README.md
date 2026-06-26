# LIBERO Standard RLPD

This module implements standard single-step RLPD for LIBERO in VLA-RL. It is Torch-native and does not import the SERL/JAX trainer.

The online policy is a SAC policy over environment actions:

```text
action = SAC(obs)
```

There is no frozen OpenPI base action in the online actor path, and the replay observation does not contain `base_action_chunk` or `alpha`.

## Algorithm

The learner samples mixed RLPD batches:

```text
batch = online_replay * (1 - offline_ratio) + offline_replay * offline_ratio
```

The current implementation uses `vla_rl.algorithms.rlpd.SACAgent`, a visual DrQ/SAC agent with image/proprio observations and environment actions. By default it follows the standard image-based RLPD path: online/offline mixed replay, DrQ random crop with edge padding, and a shared visual encoder for actor and critic.
The 10 LIBERO spatial configs set runtime.random_steps: 0, so actor rollout never uses uniform random warmup actions. The actor still samples from the SAC stochastic policy. The runtime.training_starts: 2500 threshold only delays learner updates until enough online replay exists.

The default visual path is:

```text
image_rgb_0, image_rgb_1 -> shared ResNet/encoder -> SAC actor and critic heads
```

For the shared encoder case, the actor update stops gradients through the encoder and freezes critic parameters while still differentiating Q with respect to the sampled action. The encoder is therefore primarily updated by critic/DrQ losses, matching the usual DrQ-style visual RL setup.

## Single-Step Contract

This implementation is deliberately standard single-step RLPD. The YAML does not expose horizon fields because there is no supported chunked/RL-residual mode in this package.

The actor sends exactly one action per environment call and asserts that `env.step_chunk()` executes exactly one step. If the environment returns `executed_steps != 1`, training stops immediately.

## Offline Replay

The default YAML keeps `runtime.require_offline: true`. The learner will fail early unless `runtime.offline_replay_path` points to an RLPD replay directory. The loader asserts:

```text
transition.executed_steps == 1
transition.action.shape == [action_dim]
transition.obs does not contain base_action_chunk or alpha
transition.obs contains proprio and configured image keys
```

Do not use PLD/residual replay here. PLD replay contains `base_action_chunk`, which is intentionally rejected.

## Run One Task

The standard workflow for one LIBERO task is:

1. Convert the matching expert demonstrations into an RLPD offline replay buffer.
2. Start the LIBERO env server, learner, and actor with the same task YAML.

The example below runs `libero_spatial/task_id=4`.

### Step 1: Activate Environment

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate /vla/users/niejunnan/envs/serl_torch
cd /vla/users/niejunnan/codebase/VLA-RL
```

### Step 2: Build Expert Offline Replay

RLPD mixes online replay with an offline replay buffer. For LIBERO, the offline buffer should usually come from expert demonstrations. Do not match the local LeRobot directories by sorted order: `task_id` order in LIBERO is not the same as directory order.

The converter resolves the official LIBERO prompt from `env.task_suite_name` and `env.task_id`, then matches that prompt against each LeRobot repo's `meta/tasks.jsonl`.

```bash
python examples/libero/rlpd/scripts/convert_lerobot_expert_replay.py \
  --config examples/libero/rlpd/configs/libero_spatial_task4_rlpd.yaml \
  --lerobot-root /vla/users/niejunnan/datasets/libero_lerobot
```

For `libero_spatial/task_id=4`, this must match:

```text
pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
```

The default output path is `outputs/rlpd_offline/libero_spatial_task4_expert`, matching the YAML's `runtime.offline_replay_path`.

### Step 3: Launch RLPD Training

```bash
bash examples/libero/rlpd/tools/launch_rlpd.sh \
  --config examples/libero/rlpd/configs/libero_spatial_task4_rlpd.yaml \
  --session vlarl_libero_spatial4_rlpd \
  --actor-gpu 0 \
  --learner-gpu 0 \
  --env-gpu 0 \
  --env-port 23100 \
  --trainer-port 5588 \
  --broadcast-port 5589 \
  --run-dir outputs/libero_spatial_task4_rlpd
```

The default launcher placement is intentionally conservative for one-task runs: actor, learner, LIBERO env, and async eval can all be placed on the same GPU. For multi-task sweeps, run one task per GPU or override the `--actor-gpu`, `--learner-gpu`, `--env-gpu`, and `--eval-gpu` flags explicitly.

The launcher starts only the LIBERO env server, learner, and actor. It does not start an OpenPI/reference-policy server because the SAC policy outputs actions directly.

### Running Other Spatial Tasks

Use the matching YAML for each task:

```text
examples/libero/rlpd/configs/libero_spatial_task0_rlpd.yaml
...
examples/libero/rlpd/configs/libero_spatial_task9_rlpd.yaml
```

For example, to run task 7, replace every `task4` / `spatial4` occurrence in the commands above with `task7` / `spatial7`, and use a separate session, run directory, and ports.


## Async Evaluation

RLPD uses the same JSONL queue runtime as RLT, but the eval worker runs the SAC policy directly and does not require an OpenPI/reference-policy server.

Enable async evaluation through the launcher:

```bash
bash examples/libero/rlpd/tools/launch_rlpd.sh \
  --config examples/libero/rlpd/configs/libero_spatial_task4_rlpd.yaml \
  --session vlarl_libero_spatial4_rlpd \
  --actor-gpu 0 \
  --learner-gpu 0 \
  --env-gpu 0 \
  --env-port 23100 \
  --with-eval \
  --eval-gpu 0 \
  --eval-env-port 23110 \
  --trainer-port 5588 \
  --broadcast-port 5589 \
  --run-dir outputs/libero_spatial_task4_rlpd
```

With `--with-eval`, the launcher starts a second LIBERO env server for evaluation. This keeps actor collection and eval rollouts from sharing the same remote simulator process.

The learner writes eval requests and results under `runtime.run_dir`:

```text
eval_queue.jsonl
eval_summary.jsonl
eval_worker.log
eval_checkpoints/
eval_runs/
```

The relevant YAML section is `runtime.async_eval`. It is disabled by default and can be enabled by the launcher or by an explicit YAML experiment config. When enabling it manually, set `runtime.async_eval.env_url` to a dedicated eval env server; the launcher does this automatically for `--with-eval`.

## Reward Model

Reward model relabeling uses the shared `vla_rl.rewards` remote-progress processor. Add a `reward` section to the YAML when needed, for example:

```yaml
reward:
  type: env_plus_potential_delta
  source: remote_progress
  scale: 1.0
  initial_progress: query_start
  on_error: fallback_sparse
  remote:
    url: http://127.0.0.1:50052
    method: predict_progress
    timeout: 120.0
    retries: 1
    retry_sleep: 0.5
  trajectory:
    image_keys: [image_rgb_0, image_rgb_1]
  async:
    max_pending_chunks: 64
```
