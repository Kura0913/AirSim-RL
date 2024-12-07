import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict
import gymnasium as gym
from torchvision.models import resnet18, ResNet18_Weights

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=32, config=None):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.config = config
        # ResNet18 for depth image
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        depth_features = self.resnet(observations["depth_image"])
        depth_features = depth_features.view(depth_features.size(0), -1)

        reduced_features = self.fc(depth_features)

        return reduced_features