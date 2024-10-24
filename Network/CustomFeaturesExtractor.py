import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict
import gymnasium as gym
from torchvision.models import resnet18, ResNet18_Weights

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=512+32+32, config=None):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.config = config
        # ResNet18 for depth image
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        # Linear layers for position and distance
        self.position_net = nn.Linear(3, 32)
        self.distance_net = nn.Linear(1, 32)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        depth_features = self.resnet(observations["depth_image"])
        depth_features = depth_features.view(depth_features.size(0), -1)
        
        position_features = self.position_net(observations["position"])
        distance_features = self.distance_net(observations["distance"])
        
        return torch.cat([depth_features, position_features, distance_features], dim=1)