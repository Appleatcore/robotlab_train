# GO2W 高程图编码策略网络改动说明

本文档记录在 RobotLab 工程中新增的 GO2W RSL-RL 高程图编码策略网络。目标是验证：将 `height_scan` 从原始 `187` 维先压缩为 `64` 维 latent，再与腿部关节、本体姿态、速度命令和上一帧动作等观测一起输入运动控制策略，是否能改善 GO2W 上楼梯效果。

## 设计目标

- 保留当前已经验证有效的 GO2W 资产、动作空间、奖励、地形和随机化配置。
- 不修改现有 RSL-RL/CusRL 配置，新增一个独立 agent entry point 做 A/B 实验。
- 使用当前 RobotLab 的一维高度图观测：`17 x 11 x 1 = 187`。
- 将高程图通过 CNN 压缩为 `64` 维，再与非高程图观测拼接后送入 actor/critic MLP。

## 网络结构

策略观测仍保持拼接形式：

```text
policy_obs = [proprio_obs, height_scan]
```

其中 `height_scan` 必须位于观测末尾。新增网络会自动从末尾切出 `187` 维：

```text
height_scan: [B, 187]
reshape -> [B, 1, 11, 17]
```

高程图编码器：

```text
Conv2d(1 -> 16, kernel=3, stride=1, padding=1)
ELU
Conv2d(16 -> 32, kernel=3, stride=2, padding=1)
ELU
Conv2d(32 -> 64, kernel=3, stride=2, padding=1)
ELU
Flatten
Linear(flatten_dim -> 64)
ELU
```

actor 输入：

```text
actor_input = concat(proprio_obs, height_latent_64)
actor_mlp   = [512, 256, 128] -> actions
```

critic 使用同样结构：

```text
critic_input = concat(critic_proprio_obs, height_latent_64)
critic_mlp   = [512, 256, 128] -> value
```

actor 和 critic 各自拥有一个高程图编码器，避免价值函数和策略梯度共享同一套 perception 参数造成额外耦合。

## 新增文件

- `source/robot_lab/robot_lab/heightmap_relate/__init__.py`
  - 导出 `HeightMapActorCritic`。
  - 提供 `register_heightmap_actor_critic()`，将自定义策略类注册到 RSL-RL runner 的全局命名空间。

- `source/robot_lab/robot_lab/heightmap_relate/heightmap_actor_critic.py`
  - 新增 `HeightMapEncoderMlp`。
  - 新增 `HeightMapActorCritic`。
  - 兼容 RSL-RL 3.x 的 `act()`、`act_inference()`、`evaluate()`、`get_actions_log_prob()` 和 `update_normalization()` 接口。
  - 默认配置：
    - `heightmap_shape=(11, 17)`
    - `heightmap_channels=1`
    - `heightmap_latent_dim=64`

## 修改文件

- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/agents/rsl_rl_ppo_cfg.py`
  - 新增 `RslRlPpoHeightMapActorCriticCfg`。
  - 新增 `UnitreeGo2WHeightMapEncoderPPORunnerCfg`。
  - 显式设置 `obs_groups = {"policy": ["policy"], "critic": ["critic"]}`，避免 RSL-RL 依靠同名观测组做隐式推断。
  - 新实验日志目录为：

```text
logs/rsl_rl/unitree_go2w_rough_heightmap_encoder
```

- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/__init__.py`
  - 给 rough GO2W 任务新增 agent entry point：

```text
rsl_rl_heightmap_cfg_entry_point
```

- `scripts/reinforcement_learning/rsl_rl/train.py`
  - 调用 `register_heightmap_actor_critic()`。
  - 作用是让 RSL-RL 的 `OnPolicyRunner` 能通过 `class_name="HeightMapActorCritic"` 找到新增网络类。
  - 对 RSL-RL 3.x runner dict 移除 `algorithm.share_cnn_encoders`，避免旧版 `PPO.__init__()` 收到不支持的参数。

- `scripts/reinforcement_learning/rsl_rl/play.py`
  - 同样注册 `HeightMapActorCritic`，保证回放和加载 checkpoint 时能恢复自定义策略网络。
  - 同样对 RSL-RL 3.x 移除 `algorithm.share_cnn_encoders`。

## 训练命令

使用新增的高程图编码策略：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent rsl_rl_heightmap_cfg_entry_point \
  --num_envs 4096 \
  --max_iterations 20000 \
  --headless
```

默认 RSL-RL 原配置仍然使用：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 4096 \
  --max_iterations 20000 \
  --headless
```

注意：当前 `unitree_go2w/rough_env_cfg.py` 保留了 `policy.height_scan`，因此默认 RSL-RL 原配置更接近 `raw_height`，不是严格盲走。如果要做严格 blind 对照，需要另建一个关闭 `policy.height_scan` 的环境或 agent 配置。

## 回放命令

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent rsl_rl_heightmap_cfg_entry_point \
  --checkpoint logs/rsl_rl/unitree_go2w_rough_heightmap_encoder/<run_dir>/<checkpoint>.pt \
  --num_envs 1
```

## 验证建议

建议至少比较以下三组：

```text
blind:        关闭 policy height_scan，普通 MLP/LSTM
raw_height:   保留 height_scan，直接拼接进普通 MLP
encoded_height: 保留 height_scan，先压缩为 64 维再进 MLP
```

本次改动实现的是 `encoded_height`。如果只比较 `blind` 和 `encoded_height`，提升可能来自“使用了高程图”，不一定来自“高程图编码”。如果要确认压缩编码本身的贡献，应对比 `raw_height` 和 `encoded_height`。

## 注意事项

- 此实现假设 `height_scan` 是观测最后一项。
- 此实现假设 RobotLab 当前 GO2W rough 的高度扫描尺寸为 `17 x 11`，即 `187` 维。
- 当前实现面向 RSL-RL 3.x 的 legacy `policy` 配置路径。RobotLab/IsaacLab 的配置类可能包含新版本字段，例如 `share_cnn_encoders`；训练和回放脚本会在创建旧版 RSL-RL runner 前移除该字段。
- 本次没有修改 GO2W 资产、奖励、地形、动作空间和 CusRL 配置。
