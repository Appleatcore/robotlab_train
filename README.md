# Go2W CusRL 简要修改说明

## 这次主要改了什么

这次整理了三类改动：

- 新增了 `cusrl_rsl_aligned` 训练配置  
  目的：让 Go2W 的 CusRL 配方更接近已经验证有效的 RSL 风格配置。

- 调整了一组 reward  
  目的：减少策略过于保守的问题，让机器人更敢发力、抬腿和过楼梯。

- 新增并接入了一个 hip 回中 reward  
  目的：约束 hip 不要在上楼时乱摆，让腿部姿态更稳定。

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

## 2026-05-17 RSL-RL 环境排查记录

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
