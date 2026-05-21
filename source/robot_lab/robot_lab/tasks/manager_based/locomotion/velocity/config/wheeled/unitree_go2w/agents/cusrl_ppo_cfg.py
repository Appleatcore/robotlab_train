# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import cusrl
from cusrl.environment.isaaclab import TrainerCfg

from robot_lab.cusrl_heightmap_relate import HeightMapEncoderMlp


@dataclass
class UnitreeGo2WRoughTrainerCfg(TrainerCfg):
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_go2w_rough"
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Lstm.Factory(
                hidden_size=256,
                num_layers=2,
            ),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Lstm.Factory(
                hidden_size=256,
                num_layers=2,
            ),
        ),
        optimizer_factory=cusrl.OptimizerFactory("AdamW", defaults={"lr": 1.0e-3}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
        hooks=[
            cusrl.hook.ValueComputation(),
            cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.99, lamda=0.95),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.ValueLoss(),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.EntropyLoss(weight=0.008),
            cusrl.hook.GradientClipping(max_grad_norm=1.0),
            cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
            cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence=0.01),
        ],
    )


@dataclass
class UnitreeGo2WFlatTrainerCfg(UnitreeGo2WRoughTrainerCfg):
    max_iterations = 5000
    experiment_name = "unitree_go2w_flat"


@dataclass
class UnitreeGo2WRoughTrainerRslAlignedCfg(TrainerCfg):
    max_iterations = 20000
    save_interval = 1000
    experiment_name = "unitree_go2w_rough_cusrl_rsl_aligned"
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
        ),
        optimizer_factory=cusrl.OptimizerFactory("Adam", defaults={"lr": 1.0e-3}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
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
        ],
    )


@dataclass
class UnitreeGo2WRoughTrainerHeightMapEncoderCfg(TrainerCfg):
    # 新实验入口：只替换 actor/critic 的 backbone，PPO 采样、loss、优化器配置沿用 rsl_aligned 版本。
    max_iterations = 20000
    save_interval = 1000
    experiment_name = "unitree_go2w_rough_cusrl_heightmap_encoder"
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            # actor 和 critic 在 CusRL 中是两个独立模块，因此各自拥有一套 heightmap encoder。
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
        optimizer_factory=cusrl.OptimizerFactory("Adam", defaults={"lr": 1.0e-3}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
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
        ],
    )


@dataclass
class UnitreeGo2WFlatTrainerRslAlignedCfg(UnitreeGo2WRoughTrainerRslAlignedCfg):
    max_iterations = 5000
    experiment_name = "unitree_go2w_flat_cusrl_rsl_aligned"
