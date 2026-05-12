from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP
from rsl_rl.utils import resolve_nn_activation


class HeightMapEncoderMlp(nn.Module):
    """Encode a trailing heightmap observation before an MLP head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        proprio_dim: int,
        heightmap_shape: tuple[int, int],
        heightmap_channels: int,
        heightmap_latent_dim: int,
        hidden_dims: list[int],
        activation: str,
    ):
        super().__init__()
        self.in_features = input_dim
        self.proprio_dim = proprio_dim
        self.heightmap_shape = heightmap_shape
        self.heightmap_channels = heightmap_channels
        self.heightmap_size = heightmap_channels * math.prod(heightmap_shape)
        self.heightmap_latent_dim = heightmap_latent_dim

        def make_activation():
            return resolve_nn_activation(activation)

        self.height_encoder = nn.Sequential(
            nn.Conv2d(heightmap_channels, 16, kernel_size=3, stride=1, padding=1),
            make_activation(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            make_activation(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            make_activation(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, heightmap_channels, heightmap_shape[0], heightmap_shape[1])
            encoded_size = self.height_encoder(sample).shape[-1]
        self.height_projector = nn.Sequential(
            nn.Linear(encoded_size, heightmap_latent_dim),
            make_activation(),
        )
        self.mlp = MLP(proprio_dim + heightmap_latent_dim, output_dim, hidden_dims, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proprio_obs = obs[:, : self.proprio_dim]
        heightmap = obs[:, self.proprio_dim :]
        heightmap = heightmap.reshape(
            -1,
            self.heightmap_channels,
            self.heightmap_shape[0],
            self.heightmap_shape[1],
        )
        height_latent = self.height_projector(self.height_encoder(heightmap))
        return self.mlp(torch.cat([proprio_obs, height_latent], dim=-1))


class HeightMapActorCritic(nn.Module):
    """RSL-RL actor-critic that compresses a trailing 17 x 11 heightmap to 64 dims."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        heightmap_shape=(11, 17),
        heightmap_channels: int = 1,
        heightmap_latent_dim: int = 64,
        **kwargs,
    ):
        if kwargs:
            print(
                "HeightMapActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.heightmap_shape = tuple(heightmap_shape)
        self.heightmap_channels = heightmap_channels
        self.heightmap_latent_dim = heightmap_latent_dim
        self.heightmap_size = heightmap_channels * math.prod(self.heightmap_shape)

        num_actor_obs = self._sum_obs_dims(obs, obs_groups["policy"])
        num_critic_obs = self._sum_obs_dims(obs, obs_groups["critic"])
        self.actor_proprio_dim = num_actor_obs - self.heightmap_size
        self.critic_proprio_dim = num_critic_obs - self.heightmap_size
        if self.actor_proprio_dim <= 0 or self.critic_proprio_dim <= 0:
            raise ValueError(
                "HeightMapActorCritic requires height_scan to be the trailing observation term. "
                f"actor_obs={num_actor_obs}, critic_obs={num_critic_obs}, heightmap_size={self.heightmap_size}"
            )

        self.actor = nn.Sequential(
            HeightMapEncoderMlp(
                input_dim=num_actor_obs,
                output_dim=num_actions,
                proprio_dim=self.actor_proprio_dim,
                heightmap_shape=self.heightmap_shape,
                heightmap_channels=heightmap_channels,
                heightmap_latent_dim=heightmap_latent_dim,
                hidden_dims=actor_hidden_dims,
                activation=activation,
            )
        )
        self.critic = nn.Sequential(
            HeightMapEncoderMlp(
                input_dim=num_critic_obs,
                output_dim=1,
                proprio_dim=self.critic_proprio_dim,
                heightmap_shape=self.heightmap_shape,
                heightmap_channels=heightmap_channels,
                heightmap_latent_dim=heightmap_latent_dim,
                hidden_dims=critic_hidden_dims,
                activation=activation,
            )
        )

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        print(
            "HeightMapActorCritic: "
            f"heightmap_size={self.heightmap_size}, "
            f"heightmap_shape={self.heightmap_shape}, "
            f"heightmap_latent_dim={heightmap_latent_dim}, "
            f"actor_proprio_dim={self.actor_proprio_dim}, "
            f"critic_proprio_dim={self.critic_proprio_dim}"
        )
        print(f"Actor heightmap encoder MLP: {self.actor}")
        print(f"Critic heightmap encoder MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _sum_obs_dims(obs, obs_groups: list[str]) -> int:
        dim = 0
        for obs_group in obs_groups:
            assert len(obs[obs_group].shape) == 2, "HeightMapActorCritic only supports 1D observation groups."
            dim += obs[obs_group].shape[-1]
        return dim

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs):
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs):
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
