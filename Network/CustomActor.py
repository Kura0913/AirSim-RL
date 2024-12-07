import torch.nn as nn
import torch as th
from stable_baselines3.td3.policies import Actor
from stable_baselines3.common.preprocessing import get_action_dim
from typing import List, Type
import gymnasium.spaces as spaces

class CustomActorNetwork(nn.Module):
    def __init__(
        self,
        features_dim: int,
        action_dim: int = 2
    ):
        super(CustomActorNetwork, self).__init__()
        self.features_dim = features_dim
        self.action_dim = action_dim

         # Layer definitions
        self.fc1 = nn.Linear(self.features_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, action_dim)
         
         # Activation functions
        self.relu = nn.ReLU()
        self.hard_swish = nn.Hardswish()
        self.hard_tanh = nn.Tanh()
    
    def forward(self, features: th.Tensor) -> th.Tensor:        
        # Layer 1: FC + ReLU
        x = self.relu(self.fc1(features))
        
        # Layer 2: FC + Hard Swish
        x = self.hard_swish(self.fc2(x))
        
        # Layer 3: FC + Hard Swish
        x = self.hard_swish(self.fc3(x))
        
        # Layer 4: FC + Hard Tanh
        x = self.hard_tanh(self.fc4(x))

        return x
    
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

        actor_net = CustomActorNetwork(self.features_dim, self.action_dim)
        self.mu = actor_net