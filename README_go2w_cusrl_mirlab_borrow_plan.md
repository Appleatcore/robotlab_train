# Go2W CusRL MIRLab Borrow Plan

## 目的

这份文档记录一套“借鉴 MIRLab，但不直接照抄”的控制变量实验计划。

目标是：

- 提升连续上楼梯时的稳定性和丝滑程度
- 尽量保留 RobotLab 当前“平地主要靠轮子滚动”的行为
- 先改训练分布和终止逻辑，再小幅借 MIRLab 的楼梯相关 reward
- 不直接修改现有代码，只先列出明确的改动方案

## 固定不变的部分

- 训练脚本：
  `/home/applepie/project_for_test/go2w_demo/robot_lab/scripts/reinforcement_learning/cusrl/train.py`
- 任务：
  `RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0`
- Agent：
  `cusrl_rsl_aligned_cfg_entry_point`
- 主配置文件：
  `/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py`
- 可选补充文件：
  `/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- 可选 trainer 文件：
  `/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/agents/cusrl_ppo_cfg.py`

建议每轮实验都固定：

- `--seed`
- `--num_envs`
- GPU 环境
- 每轮只改本文档指定的项

## 为什么先这样借 MIRLab

对比发现，MIRLab 里楼梯更丝滑，不只是因为 `feet_slide / feet_stumble` 更大，更关键的是：

- reset 更温和
- `base + hip` 非法接触会终止
- 对默认关节姿态的惩罚更弱
- 对机身姿态约束更明确
- 可选地还有 finer velocity tracking 和 trainer hooks

所以这份计划的顺序是：

1. 先借 `reset`
2. 再单独测试 `illegal_contact`
3. 再小幅借 `stair-contact rewards`
4. 再补 finer tracking
5. 再决定要不要做 `height_scan` 的 sim-only 分支
6. 最后再考虑 trainer hooks

## Round 1: Reset Only

目标：

- 减少 recovery 训练成分
- 先单独验证更温和的 reset 是否能提升早期课程稳定性
- 暂时不引入 `illegal_contact`，避免把“reset 改动”和“终止逻辑改动”混在一起

修改文件：

- `/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py`

具体改动：

1. 修改 `randomize_reset_base.params["pose_range"]`

位置：

- `rough_env_cfg.py:109`

当前：

```python
"x": (-0.5, 0.5),
"y": (-0.5, 0.5),
"z": (0.0, 0.2),
"roll": (-3.14, 3.14),
"pitch": (-3.14, 3.14),
"yaw": (-3.14, 3.14),
```

建议：

```python
"x": (-0.5, 0.5),
"y": (-0.5, 0.5),
"z": (0.0, 0.05),
"roll": (0.0, 0.0),
"pitch": (0.0, 0.0),
"yaw": (-3.14, 3.14),
```

2. 修改 `randomize_reset_base.params["velocity_range"]`

位置：

- `rough_env_cfg.py:118`

当前：

```python
"x": (-0.5, 0.5),
"y": (-0.5, 0.5),
"z": (-0.5, 0.5),
"roll": (-0.5, 0.5),
"pitch": (-0.5, 0.5),
"yaw": (-0.5, 0.5),
```

建议：

```python
"x": (-0.5, 0.5),
"y": (-0.5, 0.5),
"z": (-0.2, 0.2),
"roll": (-0.2, 0.2),
"pitch": (-0.2, 0.2),
"yaw": (-0.5, 0.5),
```

训练命令：

```bash
python /home/applepie/project_for_test/go2w_demo/robot_lab/scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_rsl_aligned_cfg_entry_point \
  --seed 42 \
  --run_name mirlab_round1_reset_only \
  --headless
```

## Round 1.5: Conservative `illegal_contact`

前提：

- 先完成 `Round 1` 的 reset-only 训练
- 再单独测试 `illegal_contact` 对课程推进和楼梯稳定性的影响
- 这一步不要直接照抄 MIRLab 的 `base + hip + threshold=1.0`
- 先做一个更保守的版本，避免训练一开始就被频繁提前终止

修改文件：

- `/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py`

具体改动：

1. 保持 `self.rewards.is_terminated.weight = 0`

位置：

- `rough_env_cfg.py:139`

当前：

```python
self.rewards.is_terminated.weight = 0
```

说明：

- 这一项保持不变，不额外给终止惩罚

2. 恢复 `illegal_contact`，但先只监控 `base`

位置：

- `rough_env_cfg.py:227`

当前：

```python
# self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
self.terminations.illegal_contact = None
```

建议：

```python
self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
# 同时把 threshold 从默认 1.0 提高到更宽松的值，例如 20.0 或 50.0
self.terminations.illegal_contact.params["threshold"] = 20.0
# 删除或注释掉下面这一行
# self.terminations.illegal_contact = None
```

原因：

- 直接使用 `base + hip` 且 `threshold=1.0` 太激进
- 在楼梯 early stage，base 或 hip 很容易轻微擦碰
- 这样会让 episode 太早结束，XY 位移不够，课程会频繁 `move_down`
- 所以第一轮 `illegal_contact` 建议先只看 `base`
- `hip` 是否要加回去，放到后续再单独测试
- `threshold` 也不要直接用 `1.0`，先从 `20.0` 起步更稳

训练命令：

```bash
python /home/applepie/project_for_test/go2w_demo/robot_lab/scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_rsl_aligned_cfg_entry_point \
  --seed 42 \
  --run_name mirlab_round1p5_illegal_contact_base_only \
  --headless
```

## Round 2: 小幅借 MIRLab 的楼梯接触奖励

前提：

- 只有 Round 1 之后楼梯姿态更稳，但仍有明显脚滑、踢阶时再做

修改文件：

- `/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py`

具体改动：

1. 增强 `wheel_vel_penalty`

位置：

- `rough_env_cfg.py:172`

当前：

```python
self.rewards.wheel_vel_penalty.weight = -5e-4
```

建议：

```python
self.rewards.wheel_vel_penalty.weight = -0.0015
```

2. 增强 `feet_stumble`

位置：

- `rough_env_cfg.py:202`

当前：

```python
self.rewards.feet_stumble.weight = -0.02
```

建议：

```python
self.rewards.feet_stumble.weight = -0.1
```

3. 增强 `feet_slide`

位置：

- `rough_env_cfg.py:204`

当前：

```python
self.rewards.feet_slide.weight = -0.02
```

建议：

```python
self.rewards.feet_slide.weight = -0.05
```

说明：

- 这里不建议直接照抄 MIRLab 的：
  - `wheel_vel_penalty = -0.01`
  - `feet_stumble = -1.0`
  - `feet_slide = -0.2`
- 那样更容易把平地轮子滚动一起压掉

训练命令：

```bash
python /home/applepie/project_for_test/go2w_demo/robot_lab/scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_rsl_aligned_cfg_entry_point \
  --seed 42 \
  --run_name mirlab_round2_stair_contact \
  --headless
```

## Round 3: 增加 finer velocity tracking

目标：

- 借 MIRLab 的双层速度跟踪思路
- 让连续上楼时的 body velocity 更规整

修改文件：

- `/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py`

具体改动：

1. 新增 `track_lin_vel_xy_exp_fine`

位置：

- 在 `velocity_env_cfg.py:526` 的 `track_lin_vel_xy_exp` 后
- 在 `velocity_env_cfg.py:529` 的 `track_ang_vel_z_exp` 前

建议插入：

```python
track_lin_vel_xy_exp_fine = RewTerm(
    func=mdp.track_lin_vel_xy_exp,
    weight=0.0,
    params={"command_name": "base_velocity", "std": 0.2},
)
```

2. 调整 tracking 权重

位置：

- `rough_env_cfg.py:191`

当前：

```python
self.rewards.track_lin_vel_xy_exp.weight = 3.0
self.rewards.track_ang_vel_z_exp.weight = 1.5
```

建议：

```python
self.rewards.track_lin_vel_xy_exp.weight = 1.5
self.rewards.track_lin_vel_xy_exp_fine.weight = 1.0
self.rewards.track_ang_vel_z_exp.weight = 1.0
```

训练命令：

```bash
python /home/applepie/project_for_test/go2w_demo/robot_lab/scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_rsl_aligned_cfg_entry_point \
  --seed 42 \
  --run_name mirlab_round3_tracking_split \
  --headless
```

## Round 4: height_scan 的 sim-only 分支

这一轮不建议直接并入准备部署到 StepIt 的主线。

原因：

- MIRLab 的 actor 直接吃 `height_scan`
- RobotLab 当前 Go2W 明确关掉了 policy `height_scan`
- 一旦打开，obs 契约会变，不再是当前可直接部署的主线

修改文件：

- `/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py`

具体改动：

位置：

- `rough_env_cfg.py:95`

当前：

```python
self.observations.policy.height_scan = None
```

建议：

```python
# 删除或注释这一行，让 policy 保留默认 height_scan
```

训练命令：

```bash
python /home/applepie/project_for_test/go2w_demo/robot_lab/scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_rsl_aligned_cfg_entry_point \
  --seed 42 \
  --run_name mirlab_round4_heightscan_sim_only \
  --headless
```

## Round 5: 借 MIRLab 的 CusRL hooks

这一轮改 trainer，不改环境。

修改文件：

- `/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/agents/cusrl_ppo_cfg.py`

具体改动：

位置：

- `cusrl_ppo_cfg.py:73`

当前：

```python
hooks=[
    cusrl.hook.ValueComputation(),
    cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.99, lamda=0.95),
    cusrl.hook.AdvantageNormalization(),
    cusrl.hook.ValueLoss(),
    cusrl.hook.OnPolicyPreparation(),
    cusrl.hook.PpoSurrogateLoss(),
    cusrl.hook.EntropyLoss(weight=0.01),
    cusrl.hook.GradientClipping(max_grad_norm=1.0),
    cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
    cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence=0.01),
]
```

建议：

```python
hooks=[
    cusrl.hook.ObservationNormalization(),
    cusrl.hook.ValueComputation(),
    cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.99, lamda=0.95),
    cusrl.hook.AdvantageNormalization(),
    cusrl.hook.ValueLoss(),
    cusrl.hook.SymmetricDataAugmentation(),
    cusrl.hook.OnPolicyPreparation(),
    cusrl.hook.PpoSurrogateLoss(),
    cusrl.hook.EntropyLoss(weight=0.01),
    cusrl.hook.GradientClipping(max_grad_norm=1.0),
    cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
    cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence=0.01),
]
```

训练命令：

```bash
python /home/applepie/project_for_test/go2w_demo/robot_lab/scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_rsl_aligned_cfg_entry_point \
  --seed 42 \
  --run_name mirlab_round5_agent_hooks \
  --headless
```

## 建议执行顺序

严格按下面顺序做：

1. Round 1
2. Round 1.5
3. 如果楼梯明显更稳，再做 Round 2
4. 如果楼梯稳了但速度和连续性一般，再做 Round 3
5. 如果想验证 MIRLab 的 `height_scan` 贡献，再做 Round 4
6. 最后再做 Round 5

## TensorBoard 重点看什么

每轮重点看：

- `Environment/Curriculum/terrain_levels`
- `Environment/Episode_Termination/terrain_out_of_bounds`
- `Environment/Episode_Reward/feet_slide`
- `Environment/Episode_Reward/feet_stumble`
- `Environment/Metrics/base_velocity/error_vel_xy`
- `Metric/episode_length`

判断原则：

- 楼梯更丝滑：
  - `terrain_levels` 更高
  - `terrain_out_of_bounds` 更低
  - `feet_slide` 更接近 `0`
  - `feet_stumble` 更接近 `0`
  - `episode_length` 更接近上限
- 如果这些改善了，但平地轮子明显不滚，就说明轮式行为被压太多

## 备注

- 这份文档只给修改方案，不代表这些修改已经执行
- 行号基于当前工作区快照；如果后续文件变动，行号可能会漂移
- 如果要真正开做，建议每一轮都单独 commit 或单独保存 patch，避免混淆
