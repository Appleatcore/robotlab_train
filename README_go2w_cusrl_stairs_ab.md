# Go2W CusRL Stairs Reward A/B Plan

## 目的

这份文档只记录一套控制变量实验流程：

- 第一轮只做 `A1`
- 如果明显减少打滑，再做 `A1 + A2`
- 然后再试 `A1 + A2 + A3`
- 最后再加 `A4/A5`

这次不新增 task，不改 agent，不改已有 Python 代码结构。每一轮都只手动编辑同一个文件，再用同一条训练脚本启动。

## 固定不变的部分

- 训练脚本：
  `/home/applepie/project_for_test/go2w_demo/robot_lab/scripts/reinforcement_learning/cusrl/train.py`
- 任务：
  `RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0`
- Agent：
  `cusrl_rsl_aligned_cfg_entry_point`
- 主要奖励配置文件：
  `/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py`

建议训练时也固定下面这些条件，便于对比：

- 固定 `--seed`
- 固定 `--num_envs`
- 固定 GPU 和其他启动参数
- 每一轮只改本文档指定的 reward 参数

## 参数含义

### `A1`

- 修改项：`feet_slide.weight`
- 建议值：`-0.02`
- 含义：惩罚脚接触地面或台阶时的横向滑动
- 目的：先直接打击“踩上去但脚往后滑”的问题

### `A2`

- 修改项：`feet_stumble.weight`
- 建议值：`-0.2`
- 含义：惩罚脚撞到楼梯立面的情况
- 目的：减少踢台阶前沿

### `A3`

- 修改项：`wheel_vel_penalty.weight`
- 建议值：`-5e-4`
- 含义：抑制不合适的轮速，尤其是抬脚或不该滚的时候空转
- 目的：平地仍可滚轮，上楼时减少“轮子空转带来的滑动”

### `A4`

- 修改项：`feet_height_body.weight`
- 建议值：`-0.05`
- 含义：约束脚在机体系下接近目标抬腿高度
- 目的：让脚更容易抬过台阶

### `A5`

- 修改项：`feet_height_body.params["target_height"]`
- 建议值：`-0.14`
- 含义：把目标脚高从 `-0.2` 提到 `-0.14`，鼓励更高抬腿
- 目的：进一步提高过阶裕量

注意：

- `feet_height_body` 当前实现是“离目标高度的平方误差”，不是正向奖励
- 所以 `feet_height_body.weight` 要保持负值，不能设成正值

## 实验轮次

### Round 1: 只做 A1

编辑文件：

`/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py`

只改这一项：

```python
self.rewards.feet_slide.weight = -0.02
```

其他相关项保持：

```python
self.rewards.feet_stumble.weight = 0
self.rewards.wheel_vel_penalty.weight = 0
self.rewards.feet_height_body.weight = 0
self.rewards.feet_height_body.params["target_height"] = -0.2
```

训练命令：

```bash
python /home/applepie/project_for_test/go2w_demo/robot_lab/scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_rsl_aligned_cfg_entry_point \
  --seed 42 \
  --run_name a1_feet_slide_m002 \
  --headless \
  --max_iterations 7000
```

### Round 2: A1 + A2

继续编辑同一个文件：

`/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py`

保留 `A1`，新增：

```python
self.rewards.feet_slide.weight = -0.02
self.rewards.feet_stumble.weight = -0.2
```

其他相关项保持：

```python
self.rewards.wheel_vel_penalty.weight = 0
self.rewards.feet_height_body.weight = 0
self.rewards.feet_height_body.params["target_height"] = -0.2
```

训练命令：

```bash
python /home/applepie/project_for_test/go2w_demo/robot_lab/scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_rsl_aligned_cfg_entry_point \
  --seed 42 \
  --run_name a1_a2_slide_stumble \
  --headless \
  --max_iterations 7000
```

### Round 3: A1 + A2 + A3

继续编辑同一个文件：

`/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py`

保留 `A1 + A2`，新增：

```python
self.rewards.feet_slide.weight = -0.02
self.rewards.feet_stumble.weight = -0.2
self.rewards.wheel_vel_penalty.weight = -5e-4
```

其他相关项保持：

```python
self.rewards.feet_height_body.weight = 0
self.rewards.feet_height_body.params["target_height"] = -0.2
```

训练命令：

```bash
python /home/applepie/project_for_test/go2w_demo/robot_lab/scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_rsl_aligned_cfg_entry_point \
  --seed 42 \
  --run_name a1_a2_a3_slide_stumble_wheel \
  --headless \
  --max_iterations 7000
```

### Round 4: A1 + A2 + A3 + A4/A5

继续编辑同一个文件：

`/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py`

保留 `A1 + A2 + A3`，新增：

```python
self.rewards.feet_slide.weight = -0.02
self.rewards.feet_stumble.weight = -0.2
self.rewards.wheel_vel_penalty.weight = -5e-4
self.rewards.feet_height_body.weight = -0.05
self.rewards.feet_height_body.params["target_height"] = -0.14
```

训练命令：

```bash
python /home/applepie/project_for_test/go2w_demo/robot_lab/scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_rsl_aligned_cfg_entry_point \
  --seed 42 \
  --run_name a1_a2_a3_a4a5_slide_stumble_wheel_height \
  --headless \
  --max_iterations 7000
```

## 结果记录建议

每一轮至少记录这几项：

- 是否更容易踩稳第一阶
- 是否明显减少脚打滑
- 是否更容易连续上两阶以上
- 平地是否仍主要靠轮子滚动，而不是变成明显腿式步态
- 导出的 `actor.onnx` 在 IsaacLab play 里的主观表现

## 日志目录

因为这几轮都使用同一个 agent：

`cusrl_rsl_aligned_cfg_entry_point`

所以日志根目录仍会落在：

`/home/applepie/project_for_test/go2w_demo/robot_lab/logs/cusrl/unitree_go2w_rough_cusrl_rsl_aligned`

不同实验主要靠 `--run_name` 区分。

