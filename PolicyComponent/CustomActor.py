import torch.nn as nn
from stable_baselines3.td3.policies import Actor
from stable_baselines3.common.preprocessing import get_action_dim
from typing import List, Type
import gymnasium.spaces as spaces
from Network.ActionNetwork import ActionNetwork
    
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
            net_arch=net_arch,
            features_dim=features_dim,
        )

        self.features_dim = features_dim
        self.action_dim = get_action_dim(self.action_space)

        actor_net = ActionNetwork(self.features_dim, self.action_dim)
        self.mu = actor_net