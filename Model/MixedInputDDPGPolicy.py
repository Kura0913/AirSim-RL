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
        **kwargs: Any,
    ):
        self.config = config

        # Set default network architecture
        if net_arch is None:
            net_arch = dict(pi=[400, 300], qf=[400, 300])
        
        if isinstance(net_arch, List):
            net_arch = dict(pi=net_arch, qf=net_arch)

        super(MixedInputDDPGPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            **kwargs,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DDPGActor(
            observation_space=self.observation_space,
            action_space=self.action_space,
            net_arch=self.net_arch['pi'],
            features_extractor=actor_kwargs["features_extractor"],
            features_dim=actor_kwargs["features_dim"],
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images,
            config=self.config
        ).to(self.device)


    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(
            observation_space=self.observation_space,
            action_space=self.action_space,
            net_arch=self.net_arch['qf'],
            features_extractor=critic_kwargs["features_extractor"],
            features_dim=critic_kwargs["features_dim"],
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images,
        ).to(self.device)

    def _build_mlp_extractor(self) -> None:
        self.features_extractor = CustomFeaturesExtractor(self.observation_space, self.config)

    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def extract_features(self, observation: torch.Tensor) -> torch.Tensor:
        return self.features_extractor(observation)
    

class DDPGActor(Actor):
    def __init__(
        self,
        observation_space,
        action_space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        config=None
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images
        )
        self.config = config