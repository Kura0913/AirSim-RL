from torchvision import models
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomNetwork, self).__init__(observation_space, features_dim)
        
        # Load a pretrained ResNet101 model
        self.cnn = models.resnet152(pretrained=True)
        
        # Remove the final fully connected layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

        # Define a new fully connected layer to match the feature dimensions
        self.linear = nn.Sequential(
            nn.Linear(self.cnn[-1].in_features, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        x = self.cnn(observations)
        x = x.view(x.size(0), -1)  # Flatten
        return self.linear(x)
