import torch as th
import torch.nn as nn

class ActionNetwork(nn.Module):
    def __init__(
        self,
        features_dim: int,
        action_dim: int = 2
    ):
        super(ActionNetwork, self).__init__()
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