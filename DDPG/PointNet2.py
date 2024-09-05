import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet2(nn.Module):
    def __init__(self, input_channels):
        super(PointNet2, self).__init__()
        # Define the layers for PointNet++
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 1024)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, dim=2, keepdim=False)[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x