import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict
import gymnasium as gym
from Network.ModifiedResnet import ModifiedResNet18

class DDPGFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=32, config=None):
        super(DDPGFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.config = config
        # Initialize the modified ResNet18
        self.depth_extractor = ModifiedResNet18(self.features_dim)

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        return self.depth_extractor(observations["depth_image"])