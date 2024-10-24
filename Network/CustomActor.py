import torch
import torch.nn as nn
from stable_baselines3.td3.policies import Actor
from stable_baselines3.common.preprocessing import get_action_dim
from typing import List, Type
import gymnasium.spaces as spaces

class CustomActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            net_arch = net_arch,
            features_dim = features_dim,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        
        custom_net_arch = [256, 256, 256]
        
        actor_net = [
            nn.Linear(features_dim, custom_net_arch[0]),
            nn.ReLU(),
            nn.Linear(custom_net_arch[0], custom_net_arch[1]),
            nn.Hardswish(),
            nn.Linear(custom_net_arch[1], custom_net_arch[2]),
            nn.Hardswish(),
            nn.Linear(custom_net_arch[2], action_dim),            
        ]
        
        # Deterministic action
        self.mu = nn.Sequential(*actor_net)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs, self.features_extractor)
        return torch.tanh(self.mu(features))