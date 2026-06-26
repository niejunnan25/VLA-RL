# RLT 与 PI 论文的差距说明（暂不处理）

本文档记录 `examples/libero/rlt` 当前实现与 Physical Intelligence **RL Token (RLT)** 论文之间的已知差距，以及若未来希望更贴近论文设定时的改进优先级。

**状态：仅文档记录，暂不实现。**

论文参考：

- [RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://arxiv.org/html/2604.23073v1)
- [PI 项目页](https://www.pi.website/research/rlt)

相关代码与说明：

- Stage 1：`examples/libero/rlt/scripts/train_stage1.py`
- Stage 2：`examples/libero/rlt/scripts/train_stage2.py`
- 算法：`vla_rl/algorithms/rlt/`
- 算法语义摘要：`docs/algorithms/rlt.md`

---

## 当前结论（摘要）

当前 LIBERO RLT 链路在 **Stage 1 RL token 训练 + Stage 2 reference-conditioned actor-critic** 上与论文 **算法骨架一致**，可以正常训练。

但它是面向 **LIBERO 仿真的 RLT v0 变体**，**不能** 1:1 复现论文在 4 个真实机器人高精度任务上的实验设定与数值结果。下列三项是「若向论文进一步对齐」时 **收益最高、改动相对可控** 的优先项。

---

## 优先级 1：Proprio 未进入 Actor / Critic

### 论文设定

论文 Appendix B 写明：actor 与 critic 的输入状态 `x` 包含 **RL token** 以及 **proprio（本体感知）**，例如关节位置/速度或末端位姿等。

### 当前实现

- Actor 循环会通过 `predict_actions_and_prefix()` 取得 `proprio`，并写入 `rlt_obs`（见 `encode_rlt_obs()`）。
- 但 `RLTAgent` 在构造网络时忽略 `proprio_dim`，actor/critic 实际只使用 **`z_rl`**（以及 actor 侧的 `reference_action`）。

相关代码：

- `vla_rl/algorithms/rlt/features.py` — `encode_rlt_obs()` 输出 `proprio`
- `vla_rl/algorithms/rlt/agent.py` — `del proprio_dim`；`_convert_batch()` 只 stack `z_rl`
- `vla_rl/algorithms/rlt/modeling.py` — `RLTActor` / `RLTCritic` 输入为 `[state, action_chunk]`

`docs/algorithms/rlt.md` 亦注明：v0 中 actor/critic state **currently uses `z_rl` only**。

### 若未来要对齐

- 将 `proprio` 拼入 actor/critic 的 state 向量，例如 `[z_rl, proprio]`。
- 同步更新 `RLTActor` / `RLTCritic` 的 `input_dim`、checkpoint `config` 字段与 Stage 2 yaml 中的 `proprio_dim`。
- 需评估旧 checkpoint 是否兼容（维度变化会导致无法直接 load）。

### 预期影响

- 对需要精确定位的 manipulation，proprio 通常很重要；补上后更接近论文 state 定义，可能改善样本效率与渐近性能。
- 这是与论文 **模型输入** 最直接的一处差距。

---

## 优先级 2：Stage 1 缺少可选的 VLA Co-Fine-Tuning（α > 0）

### 论文设定

论文 Algorithm 1 / Section IV-A：Stage 1 在 task demo 上联合优化

```text
L = L_ro(ϕ) + α · L_vla(θ_vla)
```

即在训练 RL token（encoder-decoder 重建 VLA prefix embedding）的同时，**可选地** 在少量任务数据上微调 VLA 权重。

### 当前实现

- Stage 1（`train_stage1.py`）**仅** 训练 `RLTokenEncoder` 与 `RLTokenDecoder`。
- OpenPI / VLA **全程冻结**；等价于论文中 **α = 0** 的路径。
- Stage 2 同样只训练 RLT actor/critic，reference policy 保持 frozen。

### 若未来要对齐

- 在 Stage 1 增加可选分支：在 LeRobot / LIBERO demo 上对 VLA 做 task-specific SFT（或论文使用的等价目标），与 RL token 重建 **联合或分阶段** 训练。
- 需明确：哪些 VLA 参数可训、学习率、与 OpenPI checkpoint 的衔接方式、Stage 2 是否加载 co-finetuned VLA。

### 预期影响

- 当 base OpenPI 在目标任务上偏弱时，只训 RL token、不动 VLA 可能限制 Stage 2 上限。
- 论文真实机器人实验通常包含这一步 task adaptation；LIBERO 上是否必需需实验验证。

---

## 优先级 3：UTD=5 时对同一 Batch 重复梯度更新

### 背景：UTD（Update-To-Data ratio）

每从 replay **采样 1 个 batch**，learner 执行 **多次** 梯度更新。当前 Stage 2 配置常见 `utd_ratio: 5`，即 1 batch → 5 次 update。

### 论文 / 常见实践

论文 Section V 提到高 UTD（如 G=5）对低数据在线 RL 很重要。SERL/RLPD 类实现中，高 UTD 通常配合 **不同 minibatch** 或 **重新采样**，避免对完全相同的数据重复刷梯度。

### 当前实现

`RLTAgent.update()` 中：

```python
for _ in range(max(1, self.utd_ratio)):
    critic_loss, ... = self._critic_step(fb)  # 同一个 fb
    ...
    if self._critic_step_count % self.policy_update_freq == 0:
        info.update(self._actor_step(fb))     # 同一个 fb
```

即 `utd_ratio=5` 时，**5 次 critic/actor 更新共用同一份 batch 张量**，未做 minibatch 切分或 reshuffle。

对比：PLD 侧已采用按 `utd_ratio` 切分不同 minibatch 的做法（见 `vla_rl/algorithms/pld/agent.py`）。

### 若未来要对齐

- 参考 PLD / RLPD：将 batch 按 `utd_ratio` 切成互不重叠的 minibatch，或每次 critic step 重新 `replay.sample()`。
- 可选：mixed batch 在切分前做 shuffle，避免 online/offline 块在同一次 update 内同质。

### 预期影响

- 通常 **不会** 导致训练逻辑完全错误，但可能降低有效样本效率、增加对单 batch 过拟合风险。
- 与论文「UTD=5」的实际效果可能不完全一致；属于 **优化细节**，优先级低于 proprio 与 Stage 1 co-FT。

---

## 其他已知差距（本文档暂不列为上述三项）

以下差距在审核中已记录，但改动面更大或属于 **实验设定差异**，当前 **不计划** 作为近期实现目标：

| 差距 | 说明 |
| --- | --- |
| Critical-phase handover | 论文在关键阶段才切换 RL；LIBERO 例子里 warmup 后整段 episode 使用 RL actor |
| Human intervention | 论文支持 teleop 覆盖并写入 replay；当前 LIBERO 链路未实现 |
| `chunk_size=5` vs 论文 `C=10` | 仿真适配；论文 Appendix B 默认 RL chunk 长度为 10 |
| Remote progress reward | LIBERO 扩展（PBRS），超出论文 sparse human ±1 reward |
| 真实机器人 4 任务 | 论文实验环境与 LIBERO 不同，数值不可直接对比 |

---

## 已实现且与论文一致的部分（供对照）

以下在设计审核中认为 **正确**，无需因上述三项而改动：

- Stage 1 encoder-decoder RL token 重建目标（Eq. 1–2 语义）
- Stage 2 actor 条件于 VLA reference action chunk + BC 正则（Eq. 5）
- Reference action dropout（`ref_dropout: 0.5`）
- TD3 风格双 Q critic + target smoothing
- `policy_update_freq: 2`（约 2 次 critic update 配 1 次 actor update）
- `critic_terminal = chunk_success`（成功才终止 bootstrap；超时截断仍 bootstrap，与 LIBERO env 语义一致）
- Window replay（`subsample_stride=2`）与跨 chunk action 拼接

---

## 变更记录

| 日期 | 说明 |
| --- | --- |
| 2026-06-26 | 初版：记录三项论文保真度差距，标记为暂不实现 |
