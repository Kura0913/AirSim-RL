import torch as th
import torch.nn as nn

class ValueNetwork(nn.Module):
    def __init__(self, features_dim):
        super().__init__()
        
        # Layer definitions
        self.fc1 = nn.Linear(features_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.hard_tanh = nn.Tanh()
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        # Layer 1: FC + ReLU
        x = self.relu(self.fc1(x))
        
        # Layer 2: FC + ReLU
        x = self.relu(self.fc2(x))
        
        # Layer 3: FC + Hard Tanh
        x = self.hard_tanh(self.fc3(x))
        
        return x