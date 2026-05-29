from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from cusrl.module.module import Module, ModuleFactory
from cusrl.module.rnn import Lstm


@dataclass(slots=True)
class HeightMapEncoderMlpFactory(ModuleFactory["HeightMapEncoderMlp"]):
    activation_fn: str | type[nn.Module] = "ELU"
    hidden_size: int = 256
    num_layers: int = 2
    bias: bool = True
    dropout: float = 0.0
    heightmap_shape: tuple[int, int] = (11, 17)
    heightmap_channels: int = 1
    heightmap_latent_dim: int = 64

    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        # CusRL 在创建 actor/critic 时会把 observation 维度传进 Factory，
        # 这里保持和 cusrl.Mlp.Factory 相同的调用方式，方便直接替换 backbone。
        assert input_dim is not None
        return HeightMapEncoderMlp(
            input_dim=input_dim,
            output_dim=output_dim,
            activation_fn=self._resolve_activation_fn(self.activation_fn),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            dropout=self.dropout,
            heightmap_shape=self.heightmap_shape,
            heightmap_channels=self.heightmap_channels,
            heightmap_latent_dim=self.heightmap_latent_dim,
        )


class HeightMapEncoderMlp(Module):
    """Encode trailing height_scan, then feed proprioception + height latent into LSTM."""

    Factory = HeightMapEncoderMlpFactory

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        activation_fn: type[nn.Module] = nn.ELU,
        hidden_size: int = 256,
        num_layers: int = 2,
        bias: bool = True,
        dropout: float = 0.0,
        heightmap_shape: tuple[int, int] = (11, 17),
        heightmap_channels: int = 1,
        heightmap_latent_dim: int = 64,
    ):
        # Go2W rough 环境中 height_scan 默认拼在 observation 末尾：
        # 187 = 1 * 11 * 17。剩余前半部分视为 proprio/body observation。
        heightmap_shape = tuple(heightmap_shape)
        heightmap_size = heightmap_channels * math.prod(heightmap_shape)
        proprio_dim = input_dim - heightmap_size
        if proprio_dim <= 0:
            raise ValueError(
                "HeightMapEncoderMlp requires a trailing height_scan. "
                f"input_dim={input_dim}, heightmap_size={heightmap_size}"
            )

        lstm_output_dim = output_dim if output_dim is not None else hidden_size
        super().__init__(input_dim=input_dim, output_dim=lstm_output_dim, is_recurrent=True)

        self.proprio_dim = proprio_dim
        self.heightmap_shape = heightmap_shape
        self.heightmap_channels = heightmap_channels
        self.heightmap_size = heightmap_size
        self.heightmap_latent_dim = heightmap_latent_dim

        # 结构对齐 RSL-RL 版本的高程图编码器：将 1x11x17 的 height_scan
        # 先用轻量 CNN 提取空间特征，再压缩成固定长度 latent。
        self.height_encoder = nn.Sequential(
            nn.Conv2d(heightmap_channels, 16, kernel_size=3, stride=1, padding=1),
            activation_fn(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            activation_fn(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            activation_fn(),
            nn.Flatten(),
        )
        with torch.no_grad():
            # 用虚拟输入推导 CNN 展平后的维度，避免手写卷积输出尺寸导致后续改 shape 时出错。
            sample = torch.zeros(1, heightmap_channels, heightmap_shape[0], heightmap_shape[1])
            encoded_size = self.height_encoder(sample).shape[-1]

        self.height_projector = nn.Sequential(
            nn.Linear(encoded_size, heightmap_latent_dim),
            activation_fn(),
        )
        self.lstm = Lstm(
            input_dim=proprio_dim + heightmap_latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            output_dim=output_dim,
        )

    def forward(
        self,
        observation: torch.Tensor,
        memory=None,
        done: torch.Tensor | None = None,
        sequential: bool = True,
        **kwargs,
    ):
        if observation.shape[-1] != self.input_dim:
            raise ValueError(f"Expected observation dim {self.input_dim}, got {observation.shape[-1]}.")

        # 兼容 CusRL 可能传入的 [B, D] 或 [T, B, D]，先展平前导维度，最后再恢复。
        leading_shape = observation.shape[:-1]
        flat_observation = observation.reshape(-1, observation.shape[-1])

        # 约定 height_scan 位于 observation 末尾；如果环境侧 observation 顺序变化，
        # 这里的切片逻辑也必须同步调整。
        proprio_obs = flat_observation[:, : self.proprio_dim]
        height_scan = flat_observation[:, self.proprio_dim :]
        height_scan = height_scan.reshape(
            -1,
            self.heightmap_channels,
            self.heightmap_shape[0],
            self.heightmap_shape[1],
        )

        height_latent = self.height_projector(self.height_encoder(height_scan))
        lstm_input = torch.cat([proprio_obs, height_latent], dim=-1).reshape(*leading_shape, -1)
        return self.lstm(lstm_input, memory=memory, done=done, sequential=sequential, **kwargs)
