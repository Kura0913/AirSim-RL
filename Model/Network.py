import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomNetwork, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        # Define the layers used to extract features
        self.linear = nn.Sequential(
            nn.Linear(64 * 7 * 7, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))
