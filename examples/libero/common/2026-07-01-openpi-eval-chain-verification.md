# OpenPI / LIBERO 评估链路一致性与波动来源验证记录

日期：2026-07-01  
仓库：`/vla/users/niejunnan/codebase/VLA-RL`  
相关目录：`examples/libero/residual_sac`、`examples/libero/common`

## 背景

这次验证的核心问题是：`residual_sac` 当前的训练链路和评估链路是否对齐；如果 base policy 的第 0 次 eval 成功率出现明显波动，这个波动到底是评估代码写错、预处理不一致、local/remote env 不一致，还是 OpenPI policy server 本身的非确定性和 request history 导致。

需要重点回答两个问题：

1. 当前 `resize / image preprocess` 是否和官方 OpenPI `main_10.py` 一致。
2. 当前 `run_residual_eval.py` 是否和训练链路一致，也就是是否都是：

```text
LIBERO obs
-> OpenPI 输入
-> websocket policy infer
-> 取前 chunk_horizon=5
-> residual compose
-> env.step_chunk
```

最终还需要验证：即使走原始官方 `openpi/examples/libero/main_10.py`，同一个 checkpoint、同一个 task、同一组 init states 是否也可能出现明显成功率偏移。

## 1. 先校对 residual_sac 的训练链路

首先检查训练入口：

```text
/vla/users/niejunnan/codebase/VLA-RL/examples/libero/residual_sac/scripts/train_residual_chunk.py
```

训练 actor 在每个决策点的逻辑是：

```text
obs
-> build_libero_state(obs)
-> extract_libero_images(obs)
-> build_libero_policy_input(prompt, state, images)
-> policy_client.infer(base_policy_input)
-> prepare_base_actions_chunk(..., chunk_horizon=5)
-> build_chunk_residual_obs(...)
-> agent.sample_action(...)
-> residual_action_spec.compose_chunk(base_action_chunk, residual_action)
-> env.step_chunk(action_chunk)
```

关键代码位置：

```text
examples/libero/residual_sac/scripts/train_residual_chunk.py:417-424
examples/libero/residual_sac/scripts/train_residual_chunk.py:446-463
```

这里确认了训练阶段不是直接 import OpenPI 做本地推理，而是通过 `policy_client.infer()` 走 OpenPI websocket client。base action 和 residual policy 的 observation 都来自同一份 OpenPI action chunk。

## 2. 再校对 run_residual_eval.py 的评估链路

然后检查评估入口：

```text
/vla/users/niejunnan/codebase/VLA-RL/examples/libero/residual_sac/scripts/run_residual_eval.py
```

串行 eval 的每个决策点逻辑是：

```text
obs
-> build_libero_state(obs)
-> extract_libero_images(obs)
-> build_libero_policy_input(prompt, state, images)
-> policy_client.infer(base_policy_input)
-> prepare_base_actions_chunk(..., chunk_horizon=5)
-> build_chunk_residual_obs(...)
-> agent.sample_action(...) 或 force_zero_residual 时 residual=0
-> residual_action_spec.compose_chunk(base_action_chunk, residual_action)
-> env.step_chunk(action_chunk)
```

关键代码位置：

```text
examples/libero/residual_sac/scripts/run_residual_eval.py:161-194
examples/libero/residual_sac/scripts/run_residual_eval.py:436-482
examples/libero/residual_sac/scripts/run_residual_eval.py:1004-1032
```

结论：当前实际使用的串行 eval 分支和训练链路是对齐的。

需要注意一个潜在风险：`run_residual_eval.py` 里也有并行 eval 分支。如果未来设置：

```text
eval.parallel_envs > 1
或 eval.policy_batch_size > 1
```

评估会走 `_build_decision_obs_many()`，并调用 `policy_client.infer_many()`。这和训练中的单条 `policy_client.infer()` 不再严格等价。当前实际 run 里没有启用这个分支。

## 3. 检查 OpenPI request payload 是否和官方一致

检查 OpenPI request builder：

```text
/vla/users/niejunnan/codebase/VLA-RL/residual_sac/policy/openpi/request_builder.py
```

当前默认 LIBERO layout 只发送这些字段：

```text
observation/image
observation/wrist_image
observation/state
prompt
```

这和官方 `main_10.py` 里的 OpenPI 输入字段一致。

官方代码位置：

```text
/vla/users/niejunnan/codebase/openpi/examples/libero/main_10.py:293-324
```

官方逻辑是：

```text
agentview_image / wrist_image
-> rotate 180 degrees
-> image_tools.resize_with_pad(..., 224, 224)
-> image_tools.convert_to_uint8(...)
-> observation/image, observation/wrist_image, observation/state, prompt
-> client.infer(element)["actions"]
-> 取前 replan_steps=5
-> env.step(action)
```

当前 `residual_sac` 的 request builder 没有再发送之前会造成差异的扩展字段，例如：

```text
images/image_mask/state/image/*
```

因此当前 OpenPI payload 已经和官方路径对齐。

## 4. 验证 resize / preprocess 是否 bit-level 一致

为了避免只靠代码阅读，专门做了 resize 对齐测试。

输出目录：

```text
/vla/users/niejunnan/eval_results/openpi_resize_check_20260630_221225
```

测试包括两类输入：

1. 随机图片输入。
2. 真实 LIBERO observation，覆盖 task4 / task5 / task8 / task9，每个 task 取 init 0 / 1 / 2，并分别检查 agentview 和 wrist view。

对比对象：

```text
当前 residual_sac 的 _resize_with_pad / extract_libero_images
vs
官方 openpi_client.image_tools.resize_with_pad + convert_to_uint8
```

结果：

| case | 数量 | mismatch | max_abs |
|---|---:|---:|---:|
| random images | 35 | 0 | 0 |
| real LIBERO obs | 24 | 0 | 0 |

结论：resize / image preprocess 链路和官方 bit-level 一致。

## 5. 检查 train config 和 async eval config 是否分叉

因为 async eval worker 是从 learner run dir 的 `.hydra/config.yaml` 派生 eval 配置，所以又检查了 actor 和 learner 的 Hydra 配置是否一致。

抽查对象包括：

```text
examples/libero/residual_sac/outputs/libero_spatial_sparse_300k/task05
examples/libero/residual_sac/outputs/reward_model/libero_spatial_robodopamine_pbrs_300k/task05
examples/libero/residual_sac/outputs/reward_model/libero_spatial_robometer_pbrs_300k/task05
examples/libero/residual_sac/outputs/reward_model/libero_spatial_robodopamine_pbrs_300k_s17/task05
examples/libero/residual_sac/outputs/reward_model/libero_spatial_robometer_pbrs_300k_s17/task05
```

结果：actor 和 learner 的 `.hydra/config.yaml` 除了这一项不同：

```text
runtime.role: actor / learner
```

其余配置一致。

结论：async eval 从 learner config 派生不会造成训练和评估配置分叉。

## 6. 确认第 0 次 async eval 是否真的是 base policy

检查训练代码里初始 eval 入队逻辑：

```text
examples/libero/residual_sac/scripts/train_residual_chunk.py:1220-1225
```

初始 eval 会调用：

```text
_queue_async_eval(target_episode=0, target_env_step=0, force_zero_residual=True)
```

然后在 eval 侧，如果 `force_zero_residual=True`，会直接使用零 residual：

```text
residual_actions = zeros(...)
final_action = base_action + 0
```

实际 run 的 `async_eval_results.jsonl` 也验证了这一点。

部分结果如下：

| run | task | first eval success | checkpoint_step | force_zero_residual | residual_l1 | policy_batch_requests |
|---|---:|---:|---:|---|---:|---:|
| Sparse | 4 | 0.56 | 0 | true | 0.0 | 0 |
| Sparse | 5 | 0.70 | 0 | true | 0.0 | 0 |
| Sparse | 9 | 0.88 | 0 | true | 0.0 | 0 |
| Robo-Dopamine PBRS s17 | 4 | 0.70 | 0 | true | 0.0 | 0 |
| Robo-Dopamine PBRS s17 | 5 | 0.62 | 0 | true | 0.0 | 0 |
| Robo-Dopamine PBRS s17 | 8 | 0.84 | 0 | true | 0.0 | 0 |
| Robo-Dopamine PBRS s17 | 9 | 0.84 | 0 | true | 0.0 | 0 |
| RoboMeter PBRS s17 | 4 | 0.68 | 0 | true | 0.0 | 0 |
| RoboMeter PBRS s17 | 5 | 0.58 | 0 | true | 0.0 | 0 |
| RoboMeter PBRS s17 | 8 | 0.78 | 0 | true | 0.0 | 0 |
| RoboMeter PBRS s17 | 9 | 0.86 | 0 | true | 0.0 | 0 |

这里 `policy_batch_requests=0` 很重要，说明实际 eval 没有走 batch infer / infer_many，而是单条 `infer()`。

结论：第 0 次 async eval 确实是在评估 base policy，不是加载了 residual checkpoint。

## 7. 检查 local env 和 remote env 是否一致

为了排除 remote env wrapper 导致状态演化不同，做了固定 action 一致性测试。

输出目录：

```text
/vla/users/niejunnan/eval_results/residual_sac_env_consistency_20260630_220105
```

设置：

```text
tasks: [4, 5, 8, 9]
init_indices: [0, 1, 2]
steps_per_init: 12
```

方法：给 local env 和 remote env 完全相同的 action sequence，然后逐步比较 obs / reward / done / truncated。

结果：

```text
cases = 156
array_equal_all = true
reward_mismatch_count = 0
done_mismatch_count = 0
truncated_mismatch_count = 0
max_abs_by_key 全部为 0.0
```

结论：在固定 action 条件下，local env 和 remote env 是 bit-level 一致的。当前波动不是 remote env wrapper 的状态演化错误。

## 8. 验证 OpenPI 同一个 observation 重复 infer 是否确定

接下来验证 OpenPI policy server 本身是否 deterministic。

输出目录：

```text
/vla/users/niejunnan/eval_results/openpi_same_obs_stochastic_20260630_221326
```

设置：

```text
task_id = 5
init_episode_idx = 0
同一个 observation
连续请求同一个 OpenPI websocket server 48 次 infer()
```

结果：

```text
num_repeated_infers = 48
unique_action_hashes = 48
all_equal_to_first = false
max_abs_vs_first_max = 0.580839216709137
pairwise_max_abs = 0.6995512843132019
per_element_std_mean = 0.06796972453594208
per_element_std_max = 0.15926003456115723
```

结论：同一个 obs、同一个 server、同一个 checkpoint，连续 `infer()` 也会返回不同 action chunk。OpenPI policy server 不是 deterministic 的。

这和 OpenPI JAX policy 内部使用 RNG sampling 的行为一致。只要每次 infer 消耗 / split RNG，server 的请求历史不同，后续 action 就可能不同。

## 9. 验证 request history 对 task5 成功率的影响

为了进一步验证“请求历史会影响结果”，做了 task5 的 request-history controlled experiment。

输出目录：

```text
/vla/users/niejunnan/eval_results/openpi_request_history_task5_20260630_222711
```

实验设计：

```text
同一个 task: libero_spatial task5
同一个 checkpoint: pi0_10000
同一组 init states: 0..49
同一个 eval helper: examples/libero/residual_sac/scripts/run_openpi_base_eval.py
同样 replan_steps=5, num_steps_wait=10, resize_size=224
每组启动一个新的 OpenPI server
区别只在 eval 前先消耗多少次 dummy infer 请求
```

最终结果：

| burn-in infer 次数 | success | successes / episodes | policy calls | env steps |
|---:|---:|---:|---:|---:|
| 0 | 0.34 | 17 / 50 | 1808 | 9003 |
| 100 | 0.36 | 18 / 50 | 1794 | 8927 |
| 500 | 0.48 | 24 / 50 | 1660 | 8259 |
| 1000 | 0.28 | 14 / 50 | 1871 | 9318 |

对应成功的 init_state 集合也不一样。例如：

```text
burn0 success_init_indices:
[1, 3, 7, 9, 15, 16, 20, 23, 25, 29, 30, 31, 33, 34, 39, 41, 44]

burn500 success_init_indices:
[0, 4, 5, 7, 11, 12, 13, 14, 17, 18, 23, 25, 26, 28, 30, 34, 36, 39, 43, 44, 46, 47, 48, 49]

burn1000 success_init_indices:
[0, 3, 4, 5, 6, 13, 17, 19, 20, 25, 27, 35, 37, 43]
```

结论：只改变 OpenPI server 的请求历史，就可以把 task5 的 50-episode success 从 0.28 推到 0.48，跨度 20 个百分点。这说明 request history 对结果有实际影响。

## 10. 用原始官方 main_10.py 复核

最后用原始官方路径复核。

脚本：

```text
/vla/users/niejunnan/codebase/openpi/examples/libero/main_10.py
```

命令形式：

```bash
/vla/users/niejunnan/envs/serl_torch/bin/python examples/libero/main_10.py \
  --task-suite-name libero_spatial \
  --host 127.0.0.1 \
  --port <openpi_port> \
  --num-trials-per-task 50 \
  --resize-size 224 \
  --replan-steps 5 \
  --num-steps-wait 10 \
  --seed 7 \
  --output-root <output_dir>
```

输出目录：

```text
/vla/users/niejunnan/eval_results/openpi_main10_spatial_repeats_20260630_221509
```

其中 repeat0 在 task2 附近出现 MuJoCo render context 相关异常并卡住，所以不作为完整有效 repeat。repeat1 和 repeat2 正常跑完了目标 task4 / task5 / task8 / task9。

结果：

| task | 旧文档 official | main_10 repeat1 | main_10 repeat2 |
|---|---:|---:|---:|
| 4 | 0.64 | 0.62 | 0.62 |
| 5 | 0.62 | 0.36 | 0.36 |
| 8 | 0.76 | 0.76 | 0.76 |
| 9 | 0.76 | 0.96 | 0.96 |

这个结果非常关键：

- task4 基本和旧文档对齐。
- task8 完全和旧文档对齐。
- task5 明显低于旧文档，从 0.62 到 0.36。
- task9 明显高于旧文档，从 0.76 到 0.96。

结论：即使走原始官方 `main_10.py`，也能出现明显偏移。尤其 task5 的低成功率不是 `residual_sac` eval helper 特有的问题，官方路径本身这次也跑出了 0.36。

## 11. 当前结论

综合代码审查和实验验证，当前结论是：

1. `resize / preprocess` 链路和官方一致。
2. 当前 `run_residual_eval.py` 的串行评估链路和训练链路一致。
3. 当前实际 run 的 async eval 没有走 `infer_many`，所以不存在“训练单条 infer，eval batch infer”导致的不一致。
4. 第 0 次 async eval 确实是 base policy：`force_zero_residual=True`、`checkpoint_step=0`、`residual_l1=0.0`。
5. local env 和 remote env 在固定 action 条件下 bit-level 一致。
6. OpenPI policy server 对同一个 obs 连续 infer 不是 deterministic 的。
7. OpenPI request history 会实质影响成功率，task5 上 50 episodes success 可以从 0.28 到 0.48。
8. 原始官方 `main_10.py` 也能跑出和旧文档明显不同的结果：task5=0.36、task9=0.96。

因此，当前更合理的判断是：

```text
这次 base policy eval 的波动，主要不是 residual_sac eval 链路写错，
而是 OpenPI JAX policy sampling / RNG request history / 单次 50 episodes 统计波动共同造成。
```

## 12. 对后续实验的建议

后续如果要让 base policy eval 更可追溯，建议：

1. 每次 eval 都记录 OpenPI server 启动时间、端口、policy config、checkpoint path。
2. 每次 eval 都记录是否复用 server，以及 eval 前 server 已经处理过多少请求。
3. base policy sanity check 不要只看单次 50 episodes，最好用多 repeat 汇总 mean/std。
4. 如果要完全比较不同训练 run 的第 0 次 eval，最好每组都使用独立新 OpenPI server，并固定是否 burn-in。
5. 如果未来打开并行 eval，需要重新验证 `infer_many` 和单条 `infer` 是否语义一致；否则不要把并行 eval 结果和训练链路严格等价。
6. 对 residual SAC 训练结果，更应该看完整训练曲线和最终多次 eval，而不是单独依赖第 0 次 base policy eval。

## 13. 实验残留清理

完成验证后，清理了本次实验残留的 `main_10.py` / OpenPI server 进程。清理后 225 上 GPU0 / GPU1 / GPU2 / GPU4 已释放。GPU3 / GPU5 / GPU6 / GPU7 当时还有 `lerobot` 占卡脚本，不属于这次验证残留。
