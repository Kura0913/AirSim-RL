import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, Any, Callable, Optional, List, Type
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy, Actor, ContinuousCritic
from Model.FeaturesExtractor.CustomFeaturesExtractor import CustomFeaturesExtractor

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

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return super().make_actor(features_extractor=actor_kwargs["features_extractor"])

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return super().make_critic(features_extractor=critic_kwargs["features_extractor"])

    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def extract_features(self, observation: torch.Tensor) -> torch.Tensor:
        return self.features_extractor(observation)