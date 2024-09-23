import torch
import torch.nn as nn

class VelocityAdjustment(nn.Module):
    def __init__(self):
        super(VelocityAdjustment, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(291, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    
    def forward(self, initial_velocity, obstacle_features):
        combined = torch.cat((initial_velocity, obstacle_features), dim=1)
        adjustment = self.fc(combined)
        return initial_velocity[:, :3] + adjustment