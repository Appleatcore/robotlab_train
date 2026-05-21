# Go2W CusRL MLP + 高程图编码网络代码变更说明

本文记录本次已经落地的 CusRL 代码改动。目标是把 Go2W rough 环境中的 `height_scan` 先编码，再送入 CusRL 的 actor/critic MLP。

## 本次新增/修改的文件

1. 新增 `source/robot_lab/robot_lab/cusrl_heightmap_relate/__init__.py`
2. 新增 `source/robot_lab/robot_lab/cusrl_heightmap_relate/heightmap_encoder_mlp.py`
3. 修改 `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/agents/cusrl_ppo_cfg.py`
4. 修改 `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/__init__.py`

## 1. 新增 CusRL 专用 backbone

文件：

- [heightmap_encoder_mlp.py](</home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/cusrl_heightmap_relate/heightmap_encoder_mlp.py>)

核心思路：

- 输入仍然是完整拼接后的 observation
- 默认从末尾切出 `187` 维 `height_scan`
- 将 `187` 维 reshape 成 `1 x 11 x 17`
- 用 CNN 压缩成 `64` 维 latent
- 再与 proprio 部分拼接
- 最后送入一个 `MLP [512, 256, 128]`

关键代码结构：

```python
class HeightMapEncoderMlpFactory(ModuleFactory["HeightMapEncoderMlp"]):
    hidden_dims: Iterable[int] = (512, 256, 128)
    activation_fn: str | type[nn.Module] = "ELU"
    ends_with_activation: bool = True
    dropout: float = 0.0
    heightmap_shape: tuple[int, int] = (11, 17)
    heightmap_channels: int = 1
    heightmap_latent_dim: int = 64
```

```python
class HeightMapEncoderMlp(Module):
    def forward(self, observation: torch.Tensor, **kwargs) -> torch.Tensor:
        leading_shape = observation.shape[:-1]
        flat_observation = observation.reshape(-1, observation.shape[-1])

        proprio_obs = flat_observation[:, : self.proprio_dim]
        height_scan = flat_observation[:, self.proprio_dim :]
        height_scan = height_scan.reshape(-1, 1, 11, 17)

        height_latent = self.height_projector(self.height_encoder(height_scan))
        latent = self.mlp(torch.cat([proprio_obs, height_latent], dim=-1))
        return latent.reshape(*leading_shape, self.output_dim)
```

这里刻意支持 `[B, D]` 和 `[T, B, D]` 两种输入形状，方便 CusRL rollout、mini-batch 和导出流程直接复用。

## 2. 修改 Go2W CusRL 配置

文件：

- [cusrl_ppo_cfg.py](</home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/agents/cusrl_ppo_cfg.py>)

新增 import：

```python
from robot_lab.cusrl_heightmap_relate import HeightMapEncoderMlp
```

新增配置类：

```python
@dataclass
class UnitreeGo2WRoughTrainerHeightMapEncoderCfg(TrainerCfg):
    max_iterations = 20000
    save_interval = 1000
    experiment_name = "unitree_go2w_rough_cusrl_heightmap_encoder"
```

这个配置和现有 `cusrl_rsl_aligned` 的区别只在 backbone：

- actor 用 `HeightMapEncoderMlp.Factory(...)`
- critic 用 `HeightMapEncoderMlp.Factory(...)`
- 其他 PPO hooks、优化器、采样器保持一致

配置片段：

```python
agent_factory = cusrl.ActorCritic.Factory(
    num_steps_per_update=24,
    actor_factory=cusrl.Actor.Factory(
        backbone_factory=HeightMapEncoderMlp.Factory(
            hidden_dims=[512, 256, 128],
            activation_fn="ELU",
            ends_with_activation=True,
            heightmap_shape=(11, 17),
            heightmap_channels=1,
            heightmap_latent_dim=64,
        ),
        distribution_factory=cusrl.NormalDist.Factory(),
    ),
    critic_factory=cusrl.Value.Factory(
        backbone_factory=HeightMapEncoderMlp.Factory(
            hidden_dims=[512, 256, 128],
            activation_fn="ELU",
            ends_with_activation=True,
            heightmap_shape=(11, 17),
            heightmap_channels=1,
            heightmap_latent_dim=64,
        ),
    ),
)
```

说明：

- CusRL 的 actor/critic 是两个独立模块，所以这里 actor 和 critic 各自拥有一套 height encoder。
- 这和 RSL-RL 的“完整 ActorCritic 类”不同，CusRL 只需要替换 backbone。

## 3. 修改 rough 任务 registry

文件：

- [unitree_go2w/__init__.py](</home/applepie/project_for_test/go2w_demo/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/__init__.py>)

新增 entry point：

```python
"cusrl_heightmap_cfg_entry_point": (
    f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2WRoughTrainerHeightMapEncoderCfg"
),
```

作用：

- 让 `--agent cusrl_heightmap_cfg_entry_point` 能直接加载新的训练配置
- 不影响现有 `cusrl_cfg_entry_point` 和 `cusrl_rsl_aligned_cfg_entry_point`

## 4. 使用方式

训练命令：

```bash
python scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_heightmap_cfg_entry_point \
  --num_envs 4096 \
  --max_iterations 20000 \
  --headless
```

回放命令：

```bash
python scripts/reinforcement_learning/cusrl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_heightmap_cfg_entry_point \
  --checkpoint logs/cusrl/unitree_go2w_rough_cusrl_heightmap_encoder/<run_dir> \
  --num_envs 1 \
  --debug_obs
```

## 5. 验证建议

建议先做三步：

1. 先验证新增/修改的 Python 文件语法
2. 再验证 backbone 的 shape
3. 最后跑小规模训练 smoke test

语法验证：

```bash
conda run -n env_isaaclab python -m py_compile \
  source/robot_lab/robot_lab/cusrl_heightmap_relate/heightmap_encoder_mlp.py \
  source/robot_lab/robot_lab/cusrl_heightmap_relate/__init__.py \
  source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/agents/cusrl_ppo_cfg.py \
  source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/__init__.py
```

纯 Python 环境里直接 `import robot_lab...` 可能会触发项目顶层 `robot_lab.__init__`，进而导入 IsaacLab/pxr。若只是验证新 backbone 的 shape，可以直接加载新增文件：

```bash
conda run -n env_isaaclab python - <<'PY'
import importlib.util
import sys

import torch

path = "source/robot_lab/robot_lab/cusrl_heightmap_relate/heightmap_encoder_mlp.py"
spec = importlib.util.spec_from_file_location("heightmap_encoder_mlp_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

net = module.HeightMapEncoderMlp(input_dim=254, hidden_dims=[512, 256, 128], activation_fn=torch.nn.ELU)
print(net(torch.zeros(4, 254)).shape)
print(net(torch.zeros(24, 4, 254)).shape)
PY
```

小规模训练 smoke test：

```bash
python scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_heightmap_cfg_entry_point \
  --num_envs 64 \
  --max_iterations 2 \
  --headless
```

## 6. 风险点

- 这个方案不是默认 `cusrl_cfg_entry_point`，而是新的实验入口。
- 强依赖 `height_scan` 在 observation 末尾。
- 强依赖 `187 = 11 x 17 x 1` 这个尺寸。
- 新 checkpoint 不能直接兼容旧 MLP/LSTM checkpoint。
- 如果后面要把 `height_scan` 的顺序改掉，必须同步改 backbone 的切片逻辑。

## 7. 结论

本次代码已经把“MLP + 高程图编码”落到了 CusRL 的正确接入点上：

- env 不变
- reward 不变
- CusRL 训练脚本不变
- 只新增一个独立 rough agent entry point
- actor/critic 各自使用 heightmap encoder backbone
