import torch
import gymnasium as gym
from typing import Dict, Any, Callable, Optional, List
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy, Actor, ContinuousCritic
from Model.FeaturesExtractor.CustomFeaturesExtractor import CustomFeaturesExtractor
import numpy as np

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
    def __init__(self, *args, **kwargs):
        self.config = kwargs.pop('config', {})
        super(DDPGActor, self).__init__(*args, **kwargs)

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the observation if needed and extract features.
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = self.preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        
        # Output direction and speed
        direction = self.mu(latent_pi)[:, :3]
        speed = torch.sigmoid(self.mu(latent_pi)[:, 3])
        
        # Normalize direction
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-8)
        
        # Combine direction and speed
        action = direction * speed.unsqueeze(1)
        
        return action

    def preprocess_obs(self, obs: torch.Tensor, observation_space: gym.spaces.Space, normalize_images: bool = True) -> torch.Tensor:
        """
        Preprocess observation to be to a neural network.
        For images, it normalizes the values by dividing them by 255 if normalize_images is True.
        """
        if isinstance(observation_space, gym.spaces.Box):
            if normalize_images and observation_space.shape[-1] == 3:
                return obs.float() / 255.0
        return obs.float()