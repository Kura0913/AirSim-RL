import torch
import torch.nn as nn
import torch.nn.functional as F
from DDPG.Deeplabv3 import DeepLabV3
from DDPG.PointNet2 import PointNet2

class Critic(nn.Module):
    def __init__(self, state_dim, lidar_dim, depth_dim, lidar_channels=3, depth_channels=1):
        super(Critic, self).__init__()
        self.lidar_dim = lidar_dim
        self.depth_dim = depth_dim
        # PointNet++ for LIDAR data
        self.pointnet2 = PointNet2(input_channels=lidar_channels)
        
        # DeepLabV3 for Depth image data
        self.deeplabv3 = DeepLabV3(num_classes=depth_channels)
        
        # Fusion Branch
        self.fc1 = nn.Linear(state_dim, 512)  # Assuming DeepLabV3 outputs 256x256
        self.fc2 = nn.Linear(512, 1)

    def forward(self, state, action, lidar_data, depth_data):
        # 处理 LIDAR 数据
        lidar_feature = self.pointnet2(lidar_data)
        
        # 处理深度图像数据
        depth_feature = self.deeplabv3(depth_data)
        
        # 将状态、动作和特征进行拼接
        fusion_input = torch.cat([state, action, lidar_feature, depth_feature], dim=1)
        
        # 经过融合层
        x = F.relu(self.fc1(fusion_input))
        q_value = self.fc2(x)
        return q_value