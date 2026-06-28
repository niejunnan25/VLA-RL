# RoboMeter Reward Path Notes

This note records the RoboMeter reward-model path used by the LIBERO PLD / residual SAC reward-model runs, and explains why the optimized path is faster without changing the online reward semantics.

## Current RoboMeter Path

The validated RoboMeter PBRS runs use a local reward adapter process:

```text
actor / reward processor
  -> VLA-RL pickle HTTP RPC: predict_progress(request)
  -> scripts/serve_robometer_progress_http.py
  -> RoboMeter model forward in the same Python process
```

The current RoboMeter adapter is launched with the important options below:

```text
--backend native
--forward-batch-size 8
--image-keys image_rgb_0
--view-mode first
--query-mode prefix_per_query
--max-history-frames 8
```

For LIBERO, the reward model uses only the main camera view, `image_rgb_0`. The PLD policy still uses both policy observation views where configured; this note only describes the reward-model input path.

## HTTP vs Native Backend

There are two different HTTP boundaries that should not be confused.

### Boundary 1: Training to Reward Adapter

This boundary still exists. The actor / async reward processor calls the reward adapter through VLA-RL local pickle HTTP RPC:

```text
reward.remote.url -> predict_progress(request)
```

This is intentional. It keeps the training process independent from the reward model implementation and allows RoboMeter, Robo-Dopamine, RoboReward, or another reward model to expose the same absolute-progress interface.

### Boundary 2: Adapter to RoboMeter Official Server

This boundary was removed in the optimized path.

The older HTTP backend looked like this:

```text
actor / reward processor
  -> VLA-RL reward adapter
  -> RoboMeter official eval_server.py over HTTP
  -> /evaluate_batch_npy multipart request
  -> RoboMeter model forward
```

That path was useful at first because it isolated the RoboMeter environment and reused the official evaluation server. It also avoided importing RoboMeter directly into the VLA-RL adapter process.

The cost is that every reward request has to be converted into `.npy` multipart payloads and sent through another HTTP server. For prefix-per-query reward inference, that is expensive because each query becomes a separate prefix sample and many image frames are repeatedly serialized.

The current native backend instead runs RoboMeter directly inside `serve_robometer_progress_http.py` using the RoboMeter Python environment:

```text
actor / reward processor
  -> VLA-RL reward adapter HTTP RPC
  -> RoboMeter batch_collator + model forward directly
```

So the current path does not remove the training-to-reward RPC. It removes the second adapter-to-official-server HTTP/multipart hop.

## Prefix Semantics

The online RL path must not use `sequence_multi_query` for reward labeling. That mode can put future frames into the model input for earlier query points.

The correct online mode is `prefix_per_query`. For query index `q`, the adapter constructs the model input only from frames with indices `<= q`:

```python
prefix_available = [idx for idx in available_indices if int(idx) <= query_index]
sampled_indices = _select_sampled_indices(
    prefix_available,
    [prefix_available[0], query_index],
    max_frames=self.max_history_frames,
)
```

This means a query for `S57` cannot see `S58...S64`. Each query is evaluated from its own history prefix.

With `max_history_frames=8`, a query at step 64 is evaluated from up to 8 sampled frames from `S0...S64`, always including the first available frame and `S64` itself. Earlier query points use their own shorter prefixes.

## What `forward_batch_size=8` Means

`forward_batch_size=8` does not mean one long sequence with 8 query points. It means the native RoboMeter backend may batch up to 8 independent prefix samples in one model forward.

For example, if the reward processor asks for progress at 8 boundary indices, the adapter builds up to 8 prefix samples:

```text
sample 1: frames <= q1 -> P(q1)
sample 2: frames <= q2 -> P(q2)
...
sample 8: frames <= q8 -> P(q8)
```

This preserves online semantics. It is slower than `sequence_multi_query`, but it avoids future-frame leakage.

In the current runs, `reward.async.batch_size=8` is a maximum. The worker only waits `max_wait_ms=5`, so it often processes batch size 1 or a small batch instead of waiting for a full batch of 8. This is expected when the reward model is keeping up with the actor.

## Why The Optimized Path Is Faster

The observed improvement comes from engineering changes, not from changing reward semantics:

- Native RoboMeter inference removes the second HTTP/multipart `.npy` serialization path.
- LIBERO RoboMeter reward inference uses one main view instead of averaging two views.
- `forward_batch_size=8` allows small groups of prefix samples to share one model forward when the reward queue has enough work.
- The runs place RoboMeter reward servers and OpenPI actor/policy processes on separate GPUs, avoiding the earlier resource contention.
- The reward processor no longer accumulates a large pending backlog in the current RoboMeter runs.

On the 2026-06-28 check, the optimized RoboMeter PBRS runs on 239 showed reward latency around `0.25s` median and `0.35-0.38s` mean in recent progress events, with reward pending usually `0-2`. Actor throughput was about `14-15.6 env/s` in the short-window sample.

The earlier slow path showed RoboMeter request latency around `9-12s` for some batch-8 requests. That behavior is not intrinsic to `prefix_per_query`; it was caused by the previous deployment/path/resource state.

## Semantic Audit Result

The optimized RoboMeter path is acceptable for online reward-model RL as long as these invariants hold:

- `--query-mode prefix_per_query` is used.
- Each query prefix is sampled only from frames `<= query_index`.
- `--image-keys image_rgb_0` is intentional for LIBERO reward-model inference.
- `reward.type` remains the training-side choice, e.g. `env_plus_potential_delta`; RoboMeter only returns absolute progress.
- The reward adapter is treated as the model-specific implementation boundary, while the trainer only consumes absolute progress.

No future-frame leakage was found in the current optimized RoboMeter path.

## Follow-Up Notes

The launch command currently carries several important reward-server settings, especially `backend`, `query_mode`, `view_mode`, `forward_batch_size`, and `max_history_frames`. For experiment traceability, these should be kept in launch logs and, when practical, mirrored in a structured config or run metadata.

Robo-Dopamine task4/5/8/9 configs were updated to use `reward.async.batch_size=8`, but already-running processes keep their original in-memory config. Restarting those runs is required before that change affects runtime behavior.
