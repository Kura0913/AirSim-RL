import torch
import torch.nn as nn
import gymnasium as gym
import gymnasium.spaces as spaces
from typing import Dict, Any, Callable, Optional, List, Type
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy, Actor, ContinuousCritic
from Model.FeaturesExtractor.CustomFeaturesExtractor import CustomFeaturesExtractor

class CustomActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super(CustomActor, self).__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
        )
        
        self.features_dim = features_dim

        self.mu = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.shape[0]),
            nn.Tanh()
        )

    def forward(self, obs):
        features = self.extract_features(obs, self.features_extractor)

        return self.mu(features)

class CustomCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(CustomCritic, self).__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            n_critics,
            share_features_extractor
        )
        self.features_dim = features_dim

        self.q_networks = nn.ModuleList([nn.Sequential(
            nn.Linear(self.features_dim + self.action_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ) for _ in range(self.n_critics)])

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        q_values = torch.cat([q_net(torch.cat([features, actions], dim=1)) for q_net in self.q_networks], dim=1)
        return q_values

class MixedInputDDPGPolicy(TD3Policy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        config: Dict[str, Any],
        net_arch: Optional[Dict[str, List[int]]] = None,
        activation_fn: Type[torch.nn.Module] = torch.nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CustomFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        **kwargs: Any,
    ):
        self.config = config
        features_extractor_kwargs = {"config": config}

        super(MixedInputDDPGPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(
            self.observation_space,
            self.action_space,
            net_arch=self.net_arch,
            features_extractor=actor_kwargs["features_extractor"],
            features_dim=actor_kwargs["features_extractor"].features_dim,
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images
        ).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomCritic(
            self.observation_space,
            self.action_space,
            net_arch=self.net_arch,
            features_extractor=critic_kwargs["features_extractor"],
            features_dim=critic_kwargs["features_extractor"].features_dim,
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images,
            share_features_extractor=self.share_features_extractor
        ).to(self.device)

    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def extract_features(self, observation: torch.Tensor) -> torch.Tensor:
        return self.features_extractor(observation)
