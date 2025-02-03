import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ModifiedResNet18(nn.Module):
    """A modified ResNet18 model for processing depth images."""
    def __init__(self, output_dim: int):
        super(ModifiedResNet18, self).__init__()
        # Load pretrained ResNet18
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Modify first conv layer to accept single-channel input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Add custom fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)
