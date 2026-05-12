# Go2W 过阶奖励组合

## 目的

这套改动针对的是在 `play.py --keyboard` 中观察到的失败模式：

- 机器人看到较大的台阶后会减速甚至停住
- 它不愿意主动抬腿
- 抬腿高度不足，容易碰到台阶前沿

当前的判断前提是：

- policy 已经能拿到 `height_scan`
- 当前策略的主要瓶颈更像是奖励过于保守、腿部动作幅度偏小，而不是完全缺少地形感知

## 改动文件

- `/home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py`

## 具体改动

### v2 继续改动：更激进的过阶探索

这组改动是在 v1 基础上继续加强“有速度命令时不要停在台阶前”的训练压力。

- `joint_pos.scale`
- hip: `0.15 -> 0.18`
- 其他腿部关节: `0.3 -> 0.36`
- `track_lin_vel_xy_exp.weight: 3.0 -> 4.0`
- `feet_air_time.weight: 0 -> 0.15`
- `feet_air_time.threshold: 0.5 -> 0.25`
- `feet_height_body.weight: -0.05 -> -0.08`
- `feet_height_body.target_height: -0.14 -> -0.10`
- `joint_torques_l2.weight: -2.5e-5 -> -1.5e-5`
- `joint_acc_l2.weight: -2.5e-7 -> -1.5e-7`
- `joint_power.weight: -1e-5 -> -5e-6`
- `stand_still.weight: -1.0 -> -0.5`
- `joint_pos_penalty.weight: -0.4 -> -0.2`
- `default_hip_joint_pos.weight: -0.02 -> -0.01`
- `joint_mirror.weight: -0.01 -> -0.005`
- `action_rate_l2.weight: -0.003 -> -0.0015`
- `contact_forces.weight: -7e-5 -> -4e-5`

目的：

- 增大腿部动作权限，让策略有空间抬腿越过更高台阶前沿
- 提高速度跟踪奖励，降低“看到高台阶后停住不尝试”的收益
- 用较小的 `feet_air_time` 正奖励鼓励移动时产生摆腿，而不是只靠轮子顶台阶
- 把机体系足端高度目标从 `-0.14` 提到 `-0.10`，鼓励摆动脚离机身更近，也就是离地更高
- 继续降低能耗、动作变化、镜像约束和接触力惩罚，减少策略为了保守而不抬腿的倾向

风险：

- v2 会比 v1 更容易产生大幅腿部动作；如果 IsaacLab 中能上阶但 sim2sim 跟踪误差继续变大，需要同步检查 MuJoCo 侧 PD/力矩限制，而不是只继续加 reward。

### 1. 小幅增大腿部动作范围

- `joint_pos.scale`
- hip: `0.125 -> 0.15`
- 其他腿部关节: `0.25 -> 0.3`

目的：

- 在面对较高台阶时，给策略更多抬腿和落腿空间

### 2. 放松保守的姿态/平滑性惩罚

- `joint_pos_limits.weight: -5.0 -> -2.0`
- `joint_power.weight: -2e-5 -> -1e-5`
- `stand_still.weight: -2.0 -> -1.0`
- `joint_pos_penalty.weight: -1.0 -> -0.4`
- `default_hip_joint_pos.weight: -0.05 -> -0.02`
- `joint_mirror.weight: -0.05 -> -0.01`
- `action_rate_l2.weight: -0.01 -> -0.003`
- `contact_forces.weight: -1.5e-4 -> -7e-5`

目的：

- 降低策略为了维持低风险滚动姿态而“停住不动”的倾向
- 降低在过阶时使用更强烈、稍不对称腿部动作的代价

### 3. 打开台阶相关奖励 shaping

- `wheel_vel_penalty.weight: 0 -> -5e-4`
- `feet_stumble.weight: 0 -> -0.2`
- `feet_slide.weight: 0 -> -0.02`
- `feet_height_body.weight: 0 -> -0.05`
- `feet_height_body.target_height: -0.2 -> -0.14`

目的：

- 抑制本该抬脚时的轮子空转
- 减少脚或轮子踢到台阶立面的情况
- 减少接触后的脚滑
- 鼓励机体系下更高的摆腿轨迹

注意：

- `feet_height_body` 当前实现是“目标脚高误差的平方惩罚”，因此它的权重必须保持为负值

## 预期效果

相较于原配置，策略应该更愿意：

- 在接近台阶前先主动抬腿
- 避免踢到台阶前沿或立面
- 避免为了维持低惩罚姿态而直接停在大台阶前

## 建议训练命令

```bash
python /home/applepie/project_for_test/go2w_demo/robot_lab/scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_rsl_aligned_cfg_entry_point \
  --seed 42 \
  --run_name step_reward_bundle_v2 \
  --headless
```

## Play 时建议观察

- 是否更容易迈过第一个大台阶
- 落脚后是否减少明显打滑
- 是否减少脚尖或轮子碰撞台阶立面
- 平地运动是否仍然主要保持轮式驱动
