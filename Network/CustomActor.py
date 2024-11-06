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
            net_arch=net_arch,
            features_dim=features_dim,
        )

        self.features_dim = features_dim
        action_dim = get_action_dim(self.action_space)
        
        self.fc1 = nn.Linear(features_dim, 32)  # First FC layer
        self.fc2 = nn.Linear(32, 32)            # Second FC layer
        self.fc3 = nn.Linear(32, action_dim)    # Output layer (3 dimensions for your case)
        
        # Use Softmax as the activation function of the middle layer
        self.softmax = nn.Softmax(dim=-1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights using He initialization"""
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='linear')
        
        # Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Extract features from the observation
        features = self.extract_features(obs, self.features_extractor)
        
        # Apply network layers with activations exactly as in the paper
        x = self.softmax(self.fc1(features))
        x = self.softmax(self.fc2(x))
        # Final layer uses linear activation (no activation function)
        actions = self.fc3(x)
        
        # Bound the actions to [-1, 1] using tanh
        # This is a common practice in continuous action spaces
        return torch.tanh(actions)
    
    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Scale the actions to the correct range if needed"""
        action_space_low = torch.tensor(self.action_space.low, device=action.device)
        action_space_high = torch.tensor(self.action_space.high, device=action.device)
        
        return 0.5 * (action_space_high - action_space_low) * action + 0.5 * (action_space_high + action_space_low)