# RoboMeter 共享运行时

## 目标

225、239、204、234 共享 `/vla`，但它们的系统 Python、UV 缓存和 CUDA/cuDNN
安装并不一致。RoboMeter 统一使用以下共享运行时：

```text
/vla/users/niejunnan/runtime/robometer/bin/uv
/vla/users/niejunnan/.cache/uv
/vla/users/niejunnan/runtime/robometer/uv-python
/vla/users/niejunnan/workspace/robometer/.venv-shared
```

`.venv-shared` 绑定共享盘上的 UV-managed CPython 3.10.20，不绑定任何机器的
`/usr/bin/python3.10`。启动时会把环境内 Torch 和 NVIDIA 动态库放在
`LD_LIBRARY_PATH` 最前面，避免 239 的系统 cuDNN 9.1 覆盖 Torch 需要的 cuDNN
9.10。

## 创建或更新

只需在任意一台已经安装 UV 的共享盘机器上执行一次，通常选择 225：

```bash
cd /vla/users/niejunnan/codebase/VLA-RL
bash examples/libero/rlpd/tools/prepare_robometer_shared_runtime.sh
```

首次创建不需要额外参数。已有环境需要更新时，先确认 204、225、234、239 上都
没有使用 `.venv-shared` 的进程，再执行：

```bash
ROBOMETER_CONFIRM_CLUSTER_DRAINED=1 \
  bash examples/libero/rlpd/tools/prepare_robometer_shared_runtime.sh
```

该命令使用 `uv.lock` 和 `--frozen --extra robometer` 同步依赖。脚本的 `pgrep`
只能检查执行命令的本机，因此更新已有共享环境必须显式确认整个集群已经排空。
更新期间共享 `.prepare.lock` 会阻止新的 RoboMeter 服务和手工激活，避免同步过程与
新任务启动交叉。

## 激活环境

不要只执行 `.venv-shared/bin/activate`，它不会修正宿主机继承的
`LD_LIBRARY_PATH`。请在 Bash 中使用：

```bash
cd /vla/users/niejunnan/codebase/VLA-RL
source examples/libero/rlpd/tools/activate_robometer_shared_runtime.sh
```

激活脚本会设置 Python、共享 UV 位置和环境内 CUDA/cuDNN 动态库优先级，并默认
执行一次完整运行时检查。退出时使用普通的 `deactivate`，原来的
`LD_LIBRARY_PATH` 和 UV 环境变量会恢复。

## 主机检查

在每台机器执行：

```bash
cd /vla/users/niejunnan/codebase/VLA-RL
bash examples/libero/rlpd/tools/check_robometer_shared_runtime.sh
```

检查内容包括共享 Python 的真实路径、Torch/CUDA/cuDNN 版本，以及实际加载的
cuDNN 是否全部来自 `.venv-shared`。

## 训练启动

训练命令不需要新增参数。`launch_residual_sac.sh` 调用
`serve_robometer_progress_stack.sh` 时会默认选择 `.venv-shared`。例如：

```bash
bash examples/libero/residual_sac/tools/launch_residual_sac.sh \
  --mode chunk \
  --config-file /vla/users/niejunnan/codebase/VLA-RL/examples/libero/residual_sac/configs/reward_model/example_robometer.yaml \
  --actor-gpu 1 \
  --env-gpu 1 \
  --eval-env-gpu 1 \
  --policy-gpu 1 \
  --backfill-gpu 1 \
  --learner-gpu 0 \
  --reward-model-gpu 0 \
  --reward-model true \
  --policy-server managed \
  --with-eval-env
```

共享运行时验证通过后，调度器不再需要
`--disallow-reward-kind-on-host robometer:239`。端口、reward 公式、scale、seed 和
训练步数仍全部由 YAML 管理。

## 显式覆盖

必要时可以通过 `ROBOMETER_SHARED_RUNTIME_ROOT`、`ROBOMETER_VENV` 或
`ROBOMETER_PYTHON_CMD` 指定其他环境；`ROBOMETER_PYTHON_CMD` 必须是单个 Python
可执行文件路径。使用共享 UV 安装目录之外的环境还需要显式设置
`ROBOMETER_ALLOW_EXTERNAL_VENV=1`。默认路径缺失或运行时校验失败时，服务会直接
报错，不会静默回退到宿主机的 `uv run python`。
