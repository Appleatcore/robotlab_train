"""CusRL PPO agent configuration for B2 omni-directional locomotion."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch import nn

import cusrl
from cusrl.environment.isaaclab import TrainerCfg

cusrl.config.enable_flash_attention(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

B2_OMNI_AUX_TARGET_SLICE = slice(55, 59)
B2_OMNI_AUX_TARGET_NAMES = (
    "base_lin_vel_b_x",
    "base_lin_vel_b_y",
    "pitch_error_to_terrain",
    "base_height_relative_to_terrain",
)


class ElevationEncoder(nn.Module):
    """Convolutional encoder for the local terrain patch."""

    def __init__(
        self,
        input_shape: tuple[int, int],
        hidden_channels: tuple[int, int] = (16, 32),
    ):
        super().__init__()
        self.input_shape = (1, *input_shape)
        self.elevation_layers = nn.Sequential(
            nn.Conv2d(1, hidden_channels[0], kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.SiLU(),
            nn.Conv2d(hidden_channels[0], hidden_channels[1], stride=2, kernel_size=3),
        )

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        input = input.unflatten(-1, self.input_shape)
        batch_dims = input.shape[:-3]
        if batch_dims:
            input = input.flatten(0, -4)

        elevation = input[..., [0], :, :]
        output = self.elevation_layers(elevation)
        if batch_dims:
            output = output.unflatten(0, batch_dims)
        return output


@dataclass
class HybridAttnEncoderFactory(cusrl.ModuleFactory):
    """Cross-attention fusion of proprioception and local terrain patch."""

    proprioception_dim: int
    elevation_shape: tuple[int, int]
    num_lstm_layers: int = 2
    lstm_hidden_dim: int = 256

    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        assert input_dim == self.proprioception_dim + math.prod(self.elevation_shape)
        return GatedHybridAttnEncoder(
            self.proprioception_dim,
            self.elevation_shape,
            num_lstm_layers=self.num_lstm_layers,
            lstm_hidden_dim=self.lstm_hidden_dim,
        )


class GatedHybridAttnEncoder(cusrl.Module):
    """Migrated hybrid attention encoder from ray_lab, aligned to leo_lab obs layout."""

    def __init__(
        self,
        proprioception_dim: int,
        elevation_shape: tuple[int, int],
        activation_fn: type[nn.Module] = nn.SiLU,
        num_lstm_layers: int = 2,
        lstm_hidden_dim: int = 256,
    ):
        self.proprioception_dim = proprioception_dim
        self.exteroception_dim = math.prod(elevation_shape)
        input_dim = self.proprioception_dim + self.exteroception_dim
        super().__init__(input_dim, lstm_hidden_dim, is_recurrent=True)

        self.proprioception_encoder = cusrl.Mlp(proprioception_dim, hidden_dims=[256, 128], activation_fn=activation_fn)
        self.proprioception_feat_dim = self.proprioception_encoder.output_dim
        self.exteroception_encoder = ElevationEncoder(input_shape=elevation_shape)

        with torch.no_grad():
            dummy_exteroception_feat = self.exteroception_encoder(torch.zeros(1, self.exteroception_dim))
            h_feat, w_feat = dummy_exteroception_feat.shape[-2:]
            dummy_exteroception_tokens = dummy_exteroception_feat.flatten(-2).transpose(-1, -2).flatten(0, -3)
        self.attn_kdim: int = dummy_exteroception_tokens.size(-1)

        self.q_norm = nn.LayerNorm(self.proprioception_feat_dim)
        self.positional_encoding = cusrl.nn.LearnablePositionalEncoding2D(self.attn_kdim, h_feat, w_feat)
        self.kv_norm = nn.LayerNorm(self.attn_kdim)
        self.fusor = cusrl.MultiheadCrossAttention(
            self.proprioception_feat_dim, num_heads=4, kv_dim=self.attn_kdim, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(self.proprioception_feat_dim)
        self.ffn = cusrl.FeedForward(self.proprioception_feat_dim, activation_fn=cusrl.nn.SwiGlu)

        self.lstm_norm = nn.LayerNorm(self.proprioception_feat_dim)
        self.historical_encoder = cusrl.Lstm(
            self.proprioception_feat_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers
        )

    def forward(self, input: torch.Tensor, memory=None, done=None, sequential: bool = True):
        proprioception, exteroception = self.split_input(input)
        proprioception_feat = self.proprioception_encoder(proprioception)
        exteroception_feat = self.exteroception_encoder(exteroception)

        proprioception_token = proprioception_feat.flatten(0, -2).unsqueeze(-2)
        exteroception_tokens = self.positional_encoding(exteroception_feat)
        exteroception_tokens = exteroception_tokens.flatten(-2).transpose(-1, -2).flatten(0, -3)

        batch_dims = input.shape[:-1]
        attn_out = self.fusor(self.q_norm(proprioception_token), self.kv_norm(exteroception_tokens))
        fused_feat = (attn_out + proprioception_token).squeeze(-2).unflatten(0, batch_dims)
        fused_feat = fused_feat + self.ffn(self.ffn_norm(fused_feat))
        return self.historical_encoder(self.lstm_norm(fused_feat), memory=memory, done=done, sequential=sequential)

    def split_input(self, input: torch.Tensor):
        return input.split((self.proprioception_dim, self.exteroception_dim), dim=-1)


class B2OmniAuxiliaryEstimator(nn.Module):
    """Predicts privileged terrain-aware state targets from the actor recurrent latent."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = len(B2_OMNI_AUX_TARGET_NAMES),
        hidden_dims: tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.SiLU()])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent.float())


class B2OmniAuxiliaryEstimatorHook(cusrl.Hook):
    """Auxiliary loss for estimating velocity and terrain-relative base state."""

    def __init__(self, loss_weight: float = 0.05):
        super().__init__()
        self.loss_weight = loss_weight
        self.register_mutable("loss_weight")

    def init(self):
        self.register_module("estimator", B2OmniAuxiliaryEstimator(self.agent.actor.latent_dim))

    @staticmethod
    def _mirror_target(target: torch.Tensor) -> torch.Tensor:
        mirrored = target.clone()
        mirrored[..., 1] *= -1.0
        return mirrored

    def _unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        try:
            observation_normalization = self.agent.hook["observation_normalization"]
        except KeyError:
            return state

        state_rms = getattr(observation_normalization, "state_rms", None)
        if state_rms is None:
            return state
        return state_rms.unnormalize(state)

    def _target_from_batch(self, batch) -> torch.Tensor | None:
        if (original_state := batch.get("original_state")) is not None:
            target = original_state[..., B2_OMNI_AUX_TARGET_SLICE]
            observation = batch.get("observation")
            if observation is not None and observation.shape[:-1] != target.shape[:-1]:
                obs_batch_shape = observation.shape[:-1]
                target_batch_shape = target.shape[:-1]
                for dim, size in enumerate(obs_batch_shape):
                    if size == 2 and obs_batch_shape[:dim] + obs_batch_shape[dim + 1 :] == target_batch_shape:
                        target = torch.stack((target, self._mirror_target(target)), dim=dim)
                        break
            return target

        state = batch.get("state")
        if state is None:
            return None
        return self._unnormalize_state(state)[..., B2_OMNI_AUX_TARGET_SLICE]

    def objective(self, batch):
        if self.loss_weight <= 0.0:
            return None

        latent = self.agent.actor.intermediate_repr.get("backbone.output")
        target = self._target_from_batch(batch)
        if latent is None or target is None:
            return None

        target = target.detach()
        with self.agent.autocast():
            prediction = self.estimator(latent)
            if prediction.shape != target.shape:
                raise RuntimeError(
                    f"B2 omni auxiliary estimator shape mismatch: prediction={prediction.shape}, target={target.shape}"
                )
            raw_loss = F.mse_loss(prediction, target)
            aux_loss = self.loss_weight * raw_loss

        self.agent.record(
            b2_omni_aux_loss=aux_loss,
            b2_omni_aux_loss_raw=raw_loss.detach(),
        )
        with torch.no_grad():
            abs_error = torch.abs(prediction.detach() - target)
            self.agent.record(
                **{
                    f"b2_omni_aux_error.{name}": abs_error[..., idx]
                    for idx, name in enumerate(B2_OMNI_AUX_TARGET_NAMES)
                }
            )
        return aux_loss


@dataclass
class B2OmniTrainerCfg(TrainerCfg):
    """Default trainer aligned with the ray_lab attention-based B2 omni policy."""

    max_iterations = 30000
    save_interval = 1000
    experiment_name = "b2_omni"
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=HybridAttnEncoderFactory(
                proprioception_dim=48, elevation_shape=(17, 13), lstm_hidden_dim=256, num_lstm_layers=2
            ),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=HybridAttnEncoderFactory(
                proprioception_dim=59, elevation_shape=(17, 13), lstm_hidden_dim=256, num_lstm_layers=2
            ),
        ),
        optimizer_factory=cusrl.OptimizerFactory(torch.optim.AdamW, defaults=dict(lr=1e-3)),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
        hooks=[
            cusrl.hook.ModuleInitialization(),
            cusrl.hook.ObservationNormalization(),
            cusrl.hook.ValueComputation(),
            cusrl.hook.GeneralizedAdvantageEstimation(),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.ValueLoss(),
            cusrl.hook.SymmetricDataAugmentation(),
            cusrl.hook.OnPolicyPreparation(),
            B2OmniAuxiliaryEstimatorHook(),
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.EntropyLoss(0.005),
            cusrl.hook.GradientClipping(1.0),
            cusrl.hook.OnPolicyStatistics(cusrl.sampler.AutoMiniBatchSampler(num_mini_batches=2)),
            cusrl.hook.AdaptiveLRSchedule(0.012),
            cusrl.hook.EmptyCudaCache(),
        ],
    )


@dataclass
class B2OmniLstmTrainerCfg(TrainerCfg):
    """Fallback plain-LSTM trainer kept for quick ablations/debugging."""

    max_iterations = 30000
    save_interval = 1000
    experiment_name = "b2_omni_lstm"
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Lstm.Factory(hidden_size=256, num_layers=2),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Lstm.Factory(hidden_size=256, num_layers=2),
        ),
        optimizer_factory=cusrl.OptimizerFactory(torch.optim.AdamW, defaults=dict(lr=1e-3)),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
        hooks=[
            cusrl.hook.ModuleInitialization(),
            cusrl.hook.ObservationNormalization(),
            cusrl.hook.ValueComputation(),
            cusrl.hook.GeneralizedAdvantageEstimation(),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.ValueLoss(),
            cusrl.hook.SymmetricDataAugmentation(),
            cusrl.hook.OnPolicyPreparation(),
            B2OmniAuxiliaryEstimatorHook(),
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.EntropyLoss(0.005),
            cusrl.hook.GradientClipping(1.0),
            cusrl.hook.OnPolicyStatistics(cusrl.sampler.AutoMiniBatchSampler()),
            cusrl.hook.AdaptiveLRSchedule(0.015),
            cusrl.hook.EmptyCudaCache(),
        ],
    )
