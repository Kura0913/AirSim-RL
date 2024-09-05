import torch
import torch.nn as nn
import torch.nn.functional as F
from DDPG.Deeplabv3 import DeepLabV3
from DDPG.PointNet2 import PointNet2

class Actor(nn.Module):
    def __init__(self, lidar_channels, depth_channels, action_dim, max_action):
        super(Actor, self).__init__()
        
        # PointNet++ for LIDAR data
        self.pointnet2 = PointNet2(input_channels=lidar_channels)
        
        # DeepLabV3 for Depth image data
        self.deeplabv3 = DeepLabV3(num_classes=depth_channels)
        
        # Fusion Branch
        self.fc1 = nn.Linear(1024 + 21*256*256, 512)  # Assuming DeepLabV3 outputs 256x256
        self.fc2 = nn.Linear(512, action_dim)
        self.max_action = max_action

    def forward(self, lidar_data=None, depth_data=None):
        if lidar_data is not None:
            lidar_features = self.pointnet2(lidar_data)
        else:
            lidar_features = torch.zeros((1, 1024))  # Placeholder for missing lidar features

        if depth_data is not None:
            depth_features = self.deeplabv3(depth_data)
            depth_features = torch.flatten(depth_features, start_dim=1)
        else:
            depth_features = torch.zeros((1, 21*256*256))  # Placeholder for missing depth features
        
        # Fusion
        fused_features = torch.cat([lidar_features, depth_features], dim=1)
        x = F.relu(self.fc1(fused_features))
        x = torch.tanh(self.fc2(x))
        return x * self.max_action