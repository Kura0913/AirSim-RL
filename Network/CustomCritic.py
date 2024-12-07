import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from typing import List, Type, Tuple
import gymnasium.spaces as spaces

class CustomCriticNetwork(nn.Module):
    def __init__(self, features_dim: int, action_dim: int):
        super().__init__()
        
        # Layer definitions
        self.fc1 = nn.Linear(features_dim + action_dim, 256)  # 32(state) + 2(action)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.hard_tanh = nn.Tanh()
    
    def forward(self, x):
        # Layer 1: FC + ReLU
        x = self.relu(self.fc1(x))
        
        # Layer 2: FC + ReLU
        x = self.relu(self.fc2(x))
        
        # Layer 3: FC + Hard Tanh
        x = self.hard_tanh(self.fc3(x))
        
        return x
    
class CustomCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(ContinuousCritic, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        self.share_features_extractor = share_features_extractor

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []
        for idx in range(n_critics):
            q_net = CustomCriticNetwork(features_dim, action_dim)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)