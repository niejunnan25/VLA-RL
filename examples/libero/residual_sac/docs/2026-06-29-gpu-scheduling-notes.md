# 2026-06-29 GPU Scheduling Notes for LIBERO Residual SAC

This note records the GPU placement that was observed to work for the current LIBERO residual SAC / reward-model training runs. The goal is to make future launches reproducible without repeatedly trying GPU order permutations.

## Scope

This document covers two concrete cases:

- `204`: `examples/libero/residual_sac` sparse residual SAC runs launched on 2026-06-29 around 22:15 CST.
- `239`: existing RoboMeter PBRS / PLD reward-model stack inspected on 2026-06-29 around 23:28 CST.

The machines both expose GPUs as H20 devices with the same index-to-bus ordering:

| GPU index | PCI bus id |
|---:|---|
| 0 | `00000000:F1:09.0` |
| 1 | `00000000:F3:09.0` |
| 2 | `00000000:F5:09.0` |
| 3 | `00000000:F7:09.0` |
| 4 | `00000000:F9:09.0` |
| 5 | `00000000:FB:09.0` |
| 6 | `00000000:FD:09.0` |
| 7 | `00000000:FF:09.0` |

## 204 Residual SAC Sparse Runs

### Final Working Mapping

The working output roots are under:

```text
/vla/users/niejunnan/codebase/VLA-RL/examples/libero/residual_sac/outputs/libero_spatial_residual_sac_sparse_serl_aligned
```

| task | learner GPU | actor/env/eval/policy GPU | output suffix | observed status |
|---|---:|---:|---|---|
| task4 | 0 | 1 | `task04_residual_sac_serl_aligned_20260629_2221_swapped` | fixed by swapping the pair |
| task5 | 3 | 2 | `task05_residual_sac_serl_aligned_20260629_221537` | default order worked |
| task8 | 5 | 4 | `task08_residual_sac_serl_aligned_20260629_221537` | default order worked |
| task9 | 7 | 6 | `task09_residual_sac_serl_aligned_20260629_221537` | default order worked |

### What Happened

The first launch used the usual convention:

```text
pair (0,1): actor/env/eval/policy on 0, learner on 1
pair (2,3): actor/env/eval/policy on 2, learner on 3
pair (4,5): actor/env/eval/policy on 4, learner on 5
pair (6,7): actor/env/eval/policy on 6, learner on 7
```

This was fine for task5/8/9, but task4 on pair `(0,1)` was abnormal: the env server was receiving very slow requests and no `actor_timers.jsonl` / `episode_logs.jsonl` appeared for several minutes. After swapping that pair to `learner=0` and `actor/env/eval/policy=1`, task4 immediately started writing episodes and reached normal actor speed.

### Verified Speed After Fix

A 90-second live sample after the task4 swap showed:

| task | actor env/s | learner status |
|---|---:|---|
| task4 swapped | ~18.9 env/s | reached heartbeat, ~4.0 update/s after warmup |
| task5 | ~15.5 env/s | normal, ~4.3 update/s |
| task8 | ~15.5 env/s | normal, ~3.9 update/s |
| task9 | ~14.4 env/s | normal, ~2.5-3.3 update/s |

The practical threshold is: actor should usually be above `10 env/s`. If it is below that after startup, or if no actor timer/episode logs appear after a few minutes, swap the two GPUs inside that task pair.

### Recommended 204 Launch Rule

For `examples/libero/residual_sac` sparse runs:

1. Start with actor/env/eval/policy on the lower/even GPU and learner on the higher/odd GPU for pairs `(2,3)`, `(4,5)`, `(6,7)`.
2. For pair `(0,1)` on 204, prefer the swapped order first: `learner=0`, `actor/env/eval/policy=1`.
3. If any task has actor speed below `10 env/s`, no actor timers, or no episode logs after 3-5 minutes, stop only that task and swap the two GPUs in its pair.
4. Do not judge learner speed before replay warmup. With chunk-level replay, `training_starts=1000` means roughly 1000 chunk transitions, so learner heartbeat may only appear around several thousand actor env steps.

### Files to Check During Monitoring

For each task output root:

```text
actor/actor_timers.jsonl
actor/episode_logs.jsonl
learner/train_residual_chunk.log
learner/async_eval_results.jsonl
```

Useful signals:

- `actor/actor_timers.jsonl`: latest `env_steps` should advance; `timer.total` around `0.23-0.29s` per chunk was normal in this run.
- `actor/episode_logs.jsonl`: `recent_success_rate_20` confirms rollout is progressing.
- `learner/train_residual_chunk.log`: look for `replay warmup complete` and `learner heartbeat`.

## 239 RoboMeter PBRS / PLD Stack

### Live Service Mapping Observed on 239

At inspection time, the main RoboMeter PBRS training processes were no longer active, but the tmux sessions and services remained. The live service placement was:

| task | RoboMeter RM GPU | RM port | OpenPI policy GPU | policy port | note |
|---|---:|---:|---:|---:|---|
| task4 | 0 | 50156 | 1 | 61504 | RM and policy separated |
| task5 | 3 | 50157 | 2 | 61505 | RM and policy separated |
| task8 | 5 | 50160 | 4 | 61508 | RM and policy separated |
| task9 | 7 | 50161 | 6 | 61509 | RM and policy separated |

Observed process commands:

- RoboMeter service: `serve_robometer_progress_http.py --backend native --model-path .../Robometer-4B --forward-batch-size 8 --query-mode prefix_per_query --view-mode first --max-history-frames 8`
- OpenPI policy: `serve_reference_policy.py --policy openpi --config-name pi0_libero_baseline_10_bs32_150000 --checkpoint-path .../pi0_10000`

### 239 Scheduling Lesson

The important rule from 239 is to keep the reward model service and the OpenPI policy service on different GPUs. Earlier output directories/log labels show that putting policy on the RM GPU or sharing actor/policy/RM too aggressively caused bad runs, for example:

```text
task*.failed_actor_shared_policy_gpu_20260629_064738
task*.policy_on_rm_gpu_slow_20260629_065936
```

So for RoboMeter PBRS, treat `RM` and `OpenPI policy` as two separate heavyweight services even if their memory footprints look modest. The RM path can be compute-bound, and the policy server also needs stable latency for actor rollout.

### Recommended 239 Rule for RoboMeter PBRS

For each task pair:

1. Put RoboMeter RM on one GPU.
2. Put OpenPI policy on the other GPU.
3. Keep actor/env/eval with the policy-side GPU unless actor speed is poor.
4. Keep learner with the RM-side GPU only if learner speed remains normal; otherwise prefer separating learner from the RM in a later run.
5. If actor drops below `10 env/s`, first check whether policy and RM are sharing a GPU. If they are, split them before tuning reward batching.

The service placement that was known to be acceptable on 239 was:

```text
task4: RM 0, policy 1
task5: RM 3, policy 2
task8: RM 5, policy 4
task9: RM 7, policy 6
```

## General Deployment Checklist

Before launching:

1. Kill unrelated `lerobot/.venv/bin/python3` reserve/stress processes when they occupy planned GPUs.
2. Confirm ports do not conflict: env, eval env, policy, trainer, broadcast, data, and reward model ports must be unique per task.
3. Use absolute output roots when launching remotely. Relative output roots can accidentally resolve under `/root` if shell quoting/cwd is wrong.
4. Log the exact GPU mapping in the output directory name or a launch manifest.

After launching:

1. Wait for train env, eval env, policy, learner, and actor to report ready.
2. Check `actor/actor_timers.jsonl` and `actor/episode_logs.jsonl` within 3-5 minutes.
3. Check actor speed over a real window. Normal is usually `>10 env/s`; the 204 residual SAC runs reached `14-19 env/s`.
4. Do not overreact to learner silence before replay warmup. Wait for `replay warmup complete` and then `learner heartbeat`.
5. If one task is bad while the others are good, stop only that task and swap the two GPUs inside its pair.

## Current Best Defaults

For sparse residual SAC on 204:

```text
task4: learner 0, actor/env/eval/policy 1
task5: learner 3, actor/env/eval/policy 2
task8: learner 5, actor/env/eval/policy 4
task9: learner 7, actor/env/eval/policy 6
```

For RoboMeter PBRS service placement on 239:

```text
task4: RM 0, policy 1
task5: RM 3, policy 2
task8: RM 5, policy 4
task9: RM 7, policy 6
```

These are empirical defaults for the current H20 machines and current LIBERO/OpenPI/RoboMeter stacks. They should be treated as launch defaults, not permanent hardware truths: if actor speed is abnormal, swap within the pair and monitor again.
