# Go2W 高程图输入与 LSTM 网络改动记录

本文记录当前 Go2W 高程图实验的主要代码改动。

## 1. 高程图观测预处理

文件：

```text
source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/observations.py
```

`height_scan_norm` 原来使用 batch 级标准化：

```python
(height_scan_raw - torch.mean(height_scan_raw)) / torch.std(height_scan_raw)
```

现在改为：

```python
height_scan = torch.nan_to_num(height_scan_raw, nan=0.0, posinf=clip[1], neginf=clip[0])
height_scan = torch.clamp(height_scan, min=clip[0], max=clip[1])
return height_scan - height_scan.mean(dim=-1, keepdim=True)
```

含义：

- 保留 `sensor_z - terrain_z - 0.5` 的原始高度定义
- 将 `nan/inf` 替换成正常数值
- 将高程图裁剪到 `[-1.0, 1.0]`
- 对每个环境自己的 187 个高程点单独减均值
- 不再除以当前 batch 或当前高程图的 std

原因：

CusRL 后面已经有：

```python
ObservationNormalization(renormalize=True)
```

它会做 running mean/std 归一化。如果在环境侧先除以当前 std，平地时 std 很小，容易把微小噪声放大。

## 2. 高程图网络结构

文件：

```text
source/robot_lab/robot_lab/cusrl_heightmap_relate/heightmap_encoder_mlp.py
```

类名仍然保持为 `HeightMapEncoderMlp`，这样不用改已有 import 和 agent 入口。

内部结构从：

```text
height_scan -> CNN -> 64维 height latent
proprioception + height latent -> MLP
```

改为：

```text
height_scan -> CNN -> 64维 height latent
proprioception + height latent -> LSTM
```

当前 LSTM 参数：

```text
hidden_size = 256
num_layers = 2
```

## 3. Agent 配置

文件：

```text
source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/agents/cusrl_ppo_cfg.py
```

高程图训练入口仍然是：

```text
cusrl_heightmap_cfg_entry_point
```

actor 和 critic 都使用：

```python
HeightMapEncoderMlp.Factory(
    activation_fn="ELU",
    hidden_size=256,
    num_layers=2,
    heightmap_shape=(11, 17),
    heightmap_channels=1,
    heightmap_latent_dim=64,
)
```

## 4. 训练命令

当前主 Slurm 文件：

```text
scripts/job.slurm
```

提交训练：

```bash
cd ~/workspace/robotlab_train
sbatch scripts/job.slurm
```

当前 run name：

```text
hmap_lstm_new_cfg
```

