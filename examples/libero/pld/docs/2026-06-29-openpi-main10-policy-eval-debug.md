# 2026-06-29 OpenPI main_10 Policy Eval 对齐记录

#### 背景

这次目标是确认 `pi0_10000` base policy 本身是否正常，以及 VLA-RL 当前 PLD / residual SAC eval 路径为什么和旧文档里的 OpenPI 官方评估结果不一致。

官方基准路径是：

```text
/vla/users/niejunnan/codebase/openpi/examples/libero/main_10.py
```

这条路径直接使用 OpenPI websocket policy server，并在 LIBERO 中按 `episode_idx % len(initial_states)` 轮询 50 个 init state。

#### 官方 main_10.py 复跑结果

运行位置：225 服务器。

policy server：

```text
/vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi_10000_policy.sh
port=64100
policy.config=pi0_libero_baseline_10_bs32_150000
policy.dir=/vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

client 输出目录：

```text
/vla/users/niejunnan/codebase/openpi/outputs/official_main10_pi0_10000_spatial_20260629_010306
```

结果：

| task | success | rate |
|---|---:|---:|
| task0 | 42 / 50 | 0.84 |
| task1 | 44 / 50 | 0.88 |
| task2 | 42 / 50 | 0.84 |
| task3 | 45 / 50 | 0.90 |
| task4 | 27 / 50 | 0.54 |
| task5 | 31 / 50 | 0.62 |
| task6 | 48 / 50 | 0.96 |
| task7 | 43 / 50 | 0.86 |
| task8 | 38 / 50 | 0.76 |
| task9 | 40 / 50 | 0.80 |
| total | 400 / 500 | 0.80 |

结论：`pi0_10000` policy 本身是正常的。尤其 task5 官方路径是 `31/50 = 0.62`，和旧文档一致。

#### VLA-RL eval 路径对比

VLA-RL 旧路径使用 `examples/libero/pld/scripts/eval_residual_sac.py --force-zero-residual`，也就是 residual 强制为 0，只测 base policy。

旧路径输出：

```text
/vla/users/niejunnan/codebase/VLA-RL/examples/libero/pld/outputs/base_policy_eval/pi0_10000_vlarl_eval_path_seq2_20260629_003623
```

| task | success rate |
|---|---:|
| task4 | 0.64 |
| task5 | 0.32 |
| task8 | 0.80 |
| task9 | 0.90 |

其中 task5 明显异常，因为官方 `main_10.py` 是 `0.62`。

#### 已确认并修复的问题 1：OpenPI payload 字段不一致

VLA-RL 的 `Observation.raw["openpi_observation"]` 里包含两类字段：

```text
官方 main_10.py 字段：
observation/image
observation/wrist_image
observation/state
prompt

VLA-RL 扩展字段：
state
images
image_mask
image/image_rgb_0
image/image_rgb_1
image/image_rgb_2
```

官方 `main_10.py` 只向 OpenPI server 发送前四个字段。修复前 VLA-RL 会把扩展字段一起发给 OpenPI。实测同一个 task5/init0 observation、同一个官方 websocket policy server，仅 payload 字段不同就会导致动作明显不同：

```text
max_abs(full_payload_action - official_payload_action) = 0.5113
```

因此这是一个确定的 eval / policy inference bug。

修复方式：当 raw observation 中存在官方 LIBERO 四字段时，OpenPI backend 只发送这四个字段。

涉及代码：

```text
/vla/users/niejunnan/codebase/VLA-RL/vla_rl/policies/openpi/backend.py
/vla/users/niejunnan/codebase/VLA-RL/examples/libero/pld/scripts/eval_residual_sac_openpi_official.py
```

修复后新增测试覆盖：

```text
/vla/users/niejunnan/codebase/VLA-RL/tests/test_openpi_policy_backend.py
```

测试命令：

```text
/vla/users/niejunnan/envs/serl_torch/bin/python -m pytest \
  tests/test_api_boundaries.py \
  tests/test_openpi_policy_backend.py \
  tests/test_rlt_reference_policy.py \
  tests/test_libero_observation.py \
  tests/test_agibot_real_examples.py -q
```

结果：

```text
21 passed
```

#### 已确认并修复的问题 2：LIBERO reset 时重复 seed

官方 `main_10.py` 的 env seed 逻辑是：

```text
创建 OffScreenRenderEnv 后调用一次 env.seed(seed)
每个 episode 只执行 env.reset() + env.set_init_state(initial_states[idx])
不在每个 episode reset 时重新 env.seed(seed)
```

`serl_torch` 的 `LiberoTaskEnv.reset()` 修复前每个 episode 都会重新调用：

```text
self.env.seed(applied_seed)
```

这会导致 init0 之后的 simulator RNG 轨迹和官方 `main_10.py` 不一致。直接验证结果：

- 修复前，在 Python3.8 官方 runtime 下，`serl_torch` wrapper 的 init0 payload 和官方一致，但 init1 之后图像开始不同。
- 修复后，在 Python3.8 官方 runtime 下，`serl_torch` wrapper 的 task5 前 5 个 init payload 和官方 `main_10.py` 完全逐像素一致。

修复位置：

```text
/vla/users/niejunnan/codebase/serl_torch/examples/libero/env/task_env.py
```

修复方式：保留构造函数里的 `self.env.seed(self.env_seed)`，但 reset 时不再重复 seed。

#### 修复后的 task5 验证

payload 修复后用官方 websocket server，再通过 VLA-RL 的 official-eval 脚本跑 task5：

```text
/vla/users/niejunnan/codebase/VLA-RL/examples/libero/pld/outputs/base_policy_eval/openpi_official_eval_script_task5_fixed_20260629_022046
```

结果：

| 路径 | task5 |
|---|---:|
| VLA-RL 旧 force-zero-residual | 16 / 50 = 0.32 |
| VLA-RL 修复 payload 后 | 24 / 50 = 0.48 |
| VLA-RL 修复 payload + reset seed 后，Python3.10 runtime，独占 server | 22 / 50 = 0.44 |
| 官方 main_10.py，Python3.8 runtime | 31 / 50 = 0.62 |

修复 payload 后 task5 从 `0.32` 提升到 `0.48`，说明 payload bug 确实是主要问题之一。随后又确认并修复了 `serl_torch` reset 重复 seed 的问题。

独占 server 下重新跑 VLA-RL Python3.10 runtime，task5 最终是 `22/50 = 0.44`。这个结果仍低于官方 Python3.8 `main_10.py` 的 `31/50 = 0.62`，说明当前 VLA-RL/serl_torch Python3.10 runtime 仍不能作为严格复现官方文档的 base-policy eval 路径。

#### 剩余差异分析

在只修复 payload、尚未修复 reset seed 时，进一步比较同一个 task5/init state，在 warmup 后构造的 policy payload：

| 对比项 | 结果 |
|---|---:|
| state max diff | 约 `1e-7` |
| image mean abs diff | 约 `0.7 - 2.9` |
| wrist image max diff | 约 `171 - 193` |

state、prompt、init-state 轮询逻辑都对齐；图像不是逐像素一致。

确认过：

- 改 VLA-RL 的 `libero_root` 到 `/vla/users/niejunnan/codebase/LIBERO` 后，图像仍和 VLA-RL 原结果一致。
- `openpi_client.image_tools.resize_with_pad` 两边源码一致。
- `MUJOCO_EGL_DEVICE_ID=0/1/2/7` 对首帧 payload 基本没有解释力。
- reset seed 修复后，Python3.8 官方 runtime 下的 `serl_torch` wrapper payload 可以和官方 `main_10.py` 完全对齐。

因此 reset seed 是第二个确定 bug。修复它后，如果仍有细小差异，才需要继续考虑 Python 3.8 官方 `libero` runtime / robosuite / mujoco 与 VLA-RL 当前 Python 3.10 runtime 的差异。

官方 `libero` 环境不能直接跑当前 VLA-RL eval 脚本，因为 VLA-RL 使用了 Python 3.10 风格能力，例如 `dataclass(slots=True)`，在 Python 3.8 下会报错。

#### 新增官方对齐脚本

为了避免以后把 VLA-RL runtime 差异误判成 policy checkpoint 问题，新增了一个不依赖 VLA-RL 包的 official-style evaluator：

```text
/vla/users/niejunnan/codebase/VLA-RL/examples/libero/pld/scripts/eval_openpi_main10_official.py
```

这个脚本复刻 OpenPI `main_10.py` 的关键路径，同时支持 `--task-ids` 只跑指定任务。它应该用官方 `libero` Python3.8 环境运行：

```bash
cd /vla/users/niejunnan/codebase/VLA-RL
/vla/users/niejunnan/envs/libero/bin/python   examples/libero/pld/scripts/eval_openpi_main10_official.py   --host 127.0.0.1   --port 64210   --task-suite-name libero_spatial   --task-ids 5   --num-trials-per-task 50   --output-dir examples/libero/pld/outputs/base_policy_eval/openpi_main10_official_task5
```

smoke test：task5 前 2 个 init state 跑通，结果 `2/2 = 1.0`，与官方 `main_10.py` 开头一致。


#### 正式 residual SAC 入口的修复后验证

本轮按照正式 residual SAC 入口验证，而不是使用旁路 `_fixed.py` 类脚本：

```text
训练入口：/vla/users/niejunnan/codebase/VLA-RL/examples/libero/pld/scripts/train_residual_sac.py
评估入口：/vla/users/niejunnan/codebase/VLA-RL/examples/libero/pld/scripts/eval_residual_sac.py
policy server：/vla/users/niejunnan/codebase/VLA-RL/scripts/serve_reference_policy.py
```

验证时启动新的 OpenPI reference policy server，走 `ReferencePolicyClient.sample_actions -> serve_reference_policy.py sample_actions -> OpenPIBackend.sample_actions -> official_openpi_libero_payload -> policy.infer`。这条路径会把 `Observation.raw["openpi_observation"]` 中的扩展字段丢弃，只保留官方四字段：

```text
observation/image
observation/wrist_image
observation/state
prompt
```

新增回归测试覆盖了两条路径：

- `OpenPIBackend.sample_actions()` 收到带扩展字段的 LIBERO raw payload 时，只把官方四字段传给 OpenPI。
- `OpenPIBackend.extract_features()` 收到同样 payload 时，也只把官方四字段传给 OpenPI。
- 半截 LIBERO official payload 会直接报错，不再静默回退到包含扩展字段的 legacy payload。
- 显式 `task` override 会覆盖 official payload 里的 `prompt`。
- AgiBot 这类非 LIBERO OpenPI schema 会原样透传，不受 LIBERO 四字段清洗影响。

测试命令：

```bash
cd /vla/users/niejunnan/codebase/VLA-RL
/vla/users/niejunnan/envs/serl_torch/bin/python -m pytest \
  tests/test_api_boundaries.py \
  tests/test_openpi_policy_backend.py \
  tests/test_rlt_reference_policy.py \
  tests/test_libero_observation.py \
  tests/test_agibot_real_examples.py -q
```

结果：

```text
21 passed
```

编译检查也通过：

```bash
/vla/users/niejunnan/envs/serl_torch/bin/python -m py_compile   vla_rl/policies/openpi/backend.py   vla_rl/policies/reference.py   scripts/serve_reference_policy.py   examples/libero/pld/scripts/train_residual_sac.py   examples/libero/pld/scripts/eval_residual_sac.py   examples/libero/common/eval_queue.py   vla_rl/envs/libero/backend.py
```

同时 `serl_torch` 的 LIBERO wrapper 编译检查通过：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
/vla/users/niejunnan/envs/libero/bin/python -m py_compile examples/libero/env/task_env.py
```

##### force-zero residual sanity：本地 Python3.10 backend

输出目录：

```text
/vla/users/niejunnan/codebase/VLA-RL/examples/libero/pld/outputs/base_policy_eval/openpi_payload_fix_force_zero_20260629_055005
```

配置：

```text
policy server: 新启动的 VLA-RL serve_reference_policy.py，端口 64600
policy root: /vla/users/niejunnan/codebase/openpi-modified
policy config: pi0_libero_baseline_10_bs32_150000
checkpoint: /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
eval script: examples/libero/pld/scripts/eval_residual_sac.py
force_zero_residual: true
env backend: LiberoLocalEnvBackend，运行在 /vla/users/niejunnan/envs/serl_torch Python3.10 runtime
```

结果：

| task | success rate | avg length | residual L1/L2 |
|---|---:|---:|---:|
| task4 | 0.64 | 165.48 | 0.0 / 0.0 |
| task5 | 0.30 | 185.86 | 0.0 / 0.0 |
| task8 | 0.72 | 134.04 | 0.0 / 0.0 |
| task9 | 0.90 | 132.60 | 0.0 / 0.0 |

task4、task8、task9 和官方 `main_10.py` 结果大致接近，但 task5 仍明显偏低：本地 backend 是 `0.30`，官方 Python3.8 `main_10.py` 是 `0.62`。

##### force-zero residual sanity：Python3.8 remote env backend

为了确认 task5 低是否来自本地 Python3.10 LIBERO runtime，又用正式 `eval_residual_sac.py` 跑了一组 remote-env 对照。不同点只有 env backend：VLA-RL client 仍在 Python3.10 中运行，但 LIBERO env 通过 RPC 调用 `/vla/users/niejunnan/envs/libero` Python3.8 环境中的 `serl_torch/examples/libero/scripts/serve_env.py`。

输出目录：

```text
/vla/users/niejunnan/codebase/VLA-RL/examples/libero/pld/outputs/base_policy_eval/openpi_payload_fix_force_zero_remote_env_task5_20260629_061700
```

配置：

```text
eval script: examples/libero/pld/scripts/eval_residual_sac.py
force_zero_residual: true
policy url: http://127.0.0.1:64600
env backend: vla_rl.envs.libero.LiberoRemoteEnvBackend
env url: http://127.0.0.1:64610
env server python: /vla/users/niejunnan/envs/libero/bin/python
```

结果：

| path | task5 success rate | avg length | residual L1/L2 |
|---|---:|---:|---:|
| VLA-RL local Python3.10 backend | 0.30 | 185.86 | 0.0 / 0.0 |
| VLA-RL remote Python3.8 env backend | 0.66 | 146.06 | 0.0 / 0.0 |
| official OpenPI main_10.py | 0.62 | - | - |

remote env backend 的 task5 是 `0.66`，已经和官方 `main_10.py` 的 `0.62` 对齐。这说明：

1. OpenPI payload 四字段修复在正式 residual SAC policy path 中生效。
2. `train_residual_sac.py` / `eval_residual_sac.py` 的 action-only base policy path 可以通过 `sample_actions` 走官方 `policy.infer()` 语义。
3. task5 的剩余异常不是 payload bug，而是本地 Python3.10 LIBERO runtime / rendering stack 与官方 Python3.8 runtime 的差异。
4. 后续正式重跑 12 组 residual SAC，如果目标是和官方 `main_10.py` base policy 行为严格对齐，建议把 PLD 训练和 eval 切到 `LiberoRemoteEnvBackend`，让 env server 在 `/vla/users/niejunnan/envs/libero` 下运行。

#### 当前结论

1. `pi0_10000` base policy 本身没有问题，官方 `main_10.py` 在 LIBERO-Spatial 上复跑总成功率是 `400/500 = 0.80`。
2. VLA-RL 旧 eval 路径确实有 bug：OpenPI payload 发送了官方路径没有的扩展字段，导致 action 改变。
3. `serl_torch` 的 LIBERO wrapper 也有对齐问题：每个 reset 重复 seed，和官方 `main_10.py` 的 seed 语义不同。
4. 两个确定 bug 都已经修复：OpenPI payload 只保留官方四字段；LIBERO wrapper 只在构造 env 时 seed，不在每个 episode reset 时重复 seed。
5. 修复后的 VLA-RL Python3.10 runtime 仍不能严格复现官方 Python3.8 `main_10.py`，task5 是 `0.44` vs 官方 `0.62`。这个剩余差异来自 runtime / rendering stack，而不是 checkpoint 本身。
6. 如果目标是检查 base policy 本身是否达到旧文档水平，使用新增的 `eval_openpi_main10_official.py`，并在 `/vla/users/niejunnan/envs/libero` 环境下运行。
7. 如果目标是评估 VLA-RL 训练得到的 residual policy，则结果应明确标注为 VLA-RL Python3.10 runtime 下的结果，不应和官方 Python3.8 `main_10.py` 直接混为同一条评估链路。
8. 如果后续希望 VLA-RL 训练/eval 也使用官方 Python3.8 LIBERO 渲染栈，建议走已有的 `LiberoRemoteEnvBackend`：用 `/vla/users/niejunnan/envs/libero` 启动 `serl_torch/examples/libero/scripts/serve_env.py`，再让 VLA-RL 通过 RPC 连接该 env server。
9. 已做 remote-env smoke test：VLA-RL Python3.10 client + Python3.8 `serve_env.py` + official OpenPI server 可以跑通 2-episode force-zero-residual eval。这个验证证明 RPC 链路可用；小样本 success 数值不作为统计结论。
