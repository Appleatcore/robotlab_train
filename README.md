# Go2W CusRL 简要修改说明

### 版本：cusrl==1.1.8.post0

## 这次主要改了什么

这次整理了四类改动：

- 新增了 `cusrl_rsl_aligned` 训练配置  
  目的：让 Go2W 的 CusRL 配方更接近已经验证有效的 RSL 风格配置。

- 调整了一组 reward  
  目的：减少策略过于保守的问题，让机器人更敢发力、抬腿和过楼梯。

- 新增并接入了一个 hip 回中 reward  
  目的：约束 hip 不要在上楼时乱摆，让腿部姿态更稳定。

- 接入了 CusRL 的 `ObservationNormalization`
  目的：让 Go2W 的 CusRL 配置使用 running mean/std 归一化整条 policy/critic observation，其中 rough 环境的 `height_scan` 也会作为 observation 的一部分被逐维归一化。

## 当前效果

目前最明显的变化是：

- 机器人上楼时腿不会像之前那样明显乱摆
- hip 姿态更收敛
- 整体上楼动作更稳定一些

## 这个 hip reward 的定位

这个新增 reward 更适合当作一个“辅助正则项”：

- 它有助于稳定姿态
- 但它不是专门解决楼梯问题的唯一主项

后续如果继续优化楼梯表现，仍然要结合：

- 脚打滑相关 reward
- 踢台阶相关 reward
- 轮子滚动相关 reward

## 如何训练

可以直接用下面的命令训练：

```bash
python /home/applepie/project_for_test/go2w_demo/robot_lab/scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_rsl_aligned_cfg_entry_point \
  --seed 42 \
  --run_name hip_default_pos \
  --headless
```

如果想让实验名更明确，也可以写成：

```bash
--run_name hip_default_pos_m005
```

## CusRL 观测归一化

这次在 Go2W 的 CusRL agent 配置中统一加入了 observation preprocessing hooks：

```python
cusrl.hook.ObservationNanToNum()
cusrl.hook.ObservationNormalization(renormalize=True)
```

改动文件：

```text
source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/agents/cusrl_ppo_cfg.py
```

已覆盖的训练入口：

- `cusrl_cfg_entry_point`
- `cusrl_rsl_aligned_cfg_entry_point`
- `cusrl_heightmap_cfg_entry_point`

它参考 CusRL 的 `cusrl/hook/mdp/observation.py` 工作方式：先用 `ObservationNanToNum` 把 observation/state 中的 NaN/Inf 清成 0，再由 `ObservationNormalization` 在训练时维护 observation/state 的 running mean/std，在 `pre_act` 和 `post_step` 阶段把 `observation`、`state`、`next_observation`、`next_state` 替换成归一化后的张量，同时保留 `original_*` 原始值。

这里显式使用 `renormalize=True`，目的是在 objective 阶段用最新 running mean/std 重新归一化保存下来的 `original_*` 张量，避免采样阶段和更新阶段统计量变化造成 batch 内 observation 使用旧归一化值。

对 Go2W rough 环境来说，`height_scan` 在 `velocity_env_cfg.py` 的 `ObservationsCfg.PolicyCfg` / `CriticCfg` 中是拼接 observation 的最后一项，尺寸来自 `height_scanner` 的 11 x 17 网格，也就是 187 维。因此启用 `ObservationNormalization` 后：

- 本体状态、关节状态、上一时刻动作等普通观测会被逐维归一化
- `height_scan` 的 187 个采样点也会被当作普通 observation 维度逐维归一化
- `cusrl_heightmap_cfg_entry_point` 中的 `HeightMapEncoderMlp` 接收到的是已经归一化后的完整 observation，再从末尾切出 187 维 `height_scan` 做 CNN 编码
- 导出 ONNX/JIT 时，CusRL 会通过 hook 的 `pre_export` 把 `observation_rms` 写进 actor graph，所以 Stepit 侧应输入与训练时同语义、同单位、同展开顺序的原始 observation，而不是再次手动套一套训练统计量归一化

这不是高程图专用的 min-max 或二维图像归一化，而是 CusRL 通用的逐维 running mean/std。flat 环境中 `height_scan` 已经被置为 `None`，所以 flat 只会归一化剩余 observation 维度。

## RSL-RL 环境排查记录

今天在服务器 `env_isaaclab` 环境中运行 RSL-RL 高程图编码训练时，默认安装的版本是：

```text
rsl-rl-lib == 5.0.1
```

启动命令为：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent rsl_rl_heightmap_cfg_entry_point \
  --num_envs 4096 \
  --max_iterations 20000 \
  --headless
```

报错信息为：

```text
ModuleNotFoundError: No module named 'rsl_rl.networks'
```

排查后确认：当前自定义的 `HeightMapActorCritic` 里使用了旧版/另一版 RSL-RL 的导入路径：

```python
from rsl_rl.networks import EmpiricalNormalization, MLP
```

但服务器默认的 `rsl-rl-lib == 5.0.1` 中没有 `rsl_rl.networks` 模块。执行下面命令降级后，训练可以正常启动：

```bash
pip install rsl-rl-lib==3.0.1
```

因此当前高程图编码策略训练需要使用 `rsl-rl-lib == 3.0.1`，或者后续将自定义网络迁移到 RSL-RL 5.x 的新接口。

## 结论

这次新增的 hip 约束是有效的，值得保留。  
后面更合理的方向是：以它为基础，再继续和楼梯相关 reward 做组合调参。

## 代码与结果同步

当前仓库里有两个同步脚本，默认都使用：

- 服务器：`ycl@10.12.120.237`
- SSH 端口：`11222`
- 服务器仓库目录：`/srv/shared/home/ycl/workspace/robotlab_train/`

### 1. 上传本地代码到服务器

执行：

```bash
bash scripts/tools/rsync_robotlab_to_server.sh
```

这个脚本会把当前仓库同步到服务器，但会排除训练结果和缓存目录，例如：

- `logs`
- `runs`
- `recordings`
- `output`
- `outputs`
- `videos`
- `wandb`
- `.neptune`

因此适合在本地改完代码后，把可运行代码快速推到服务器。

### 2. 从服务器回传结果到本地

这个脚本必须在本地机器执行，不要在服务器 `venus` 上执行。

执行：

```bash
bash scripts/tools/rsync_robotlab_results_from_server.sh
```

这个脚本不会覆盖本地代码，只会把服务器上的结果目录同步回当前仓库，包含：

- `logs`
- `runs`
- `recordings`
- `output`
- `outputs`
- `videos`
- `wandb`
- `.neptune`

其中会默认排除 `logs/` 目录下的 `*.out` 和 `*.err` 作业日志，避免把服务器侧的终端输出日志也拉回本地。

适合在服务器训练完成后，把日志、权重、可视化结果和实验记录拉回本地保存。

如果你是在服务器 shell 里运行这个脚本，即使 `rsync` 输出看起来正常，也只是“服务器从服务器同步到服务器本机”，并没有把文件带回本地。

### 3. 在本地查看服务器上的 TensorBoard

如果训练结果还在服务器上，不想先回传到本地，可以直接通过 SSH 端口转发在本地浏览器查看 TensorBoard。

1. 在服务器 `venus` 上启动 TensorBoard：

```bash
tensorboard --logdir ./logs/cusrl/unitree_go2w_rough_cusrl_heightmap_encoder --host 127.0.0.1 --port 6010
```

2. 在本地机器建立 SSH 端口转发：

```bash
ssh -L 6010:127.0.0.1:6010 ycl@10.12.120.237 -p 11222
```

3. 在本地浏览器打开：

```text
http://127.0.0.1:6010/
```
