import torch
import torch.nn as nn
import torch.nn.functional as F
from DDPG.Deeplabv3 import DeepLabV3
from DDPG.PointNet2 import PointNet2

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, points_dim, depth_dim):
        super(Actor, self).__init__()
        self.points_dim = points_dim
        self.depth_dim = depth_dim
        # PointNet++ for LIDAR data
        self.pointnet2 = PointNet2(input_channels=3)
        
        # DeepLabV3 for Depth image data
        self.deeplabv3 = DeepLabV3(num_classes=1)
        
        # Fusion Branch
        self.fc1 = nn.Linear(state_dim, 512)  # Assuming DeepLabV3 outputs 256x256
        self.fc2 = nn.Linear(512, action_dim)
        self.max_action = max_action

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # 解析 LIDAR 数据和深度图像数据
        lidar_data = state[:, :self.points_dim].reshape(-1, self.points_dim // 3, 3)  # 恢复为点云的形状
        depth_data = state[:, self.points_dim:].reshape(-1, self.depth_dim)  # 恢复为深度图像的形状

        # 处理 LIDAR 数据
        lidar_feature = self.pointnet2(lidar_data)
        
        # 处理深度图像数据
        depth_feature = self.deeplabv3(depth_data)
        
        # 将提取的特征进行拼接
        fusion_input = torch.cat([lidar_feature, depth_feature], dim=1)
        
        # 经过融合层
        x = F.relu(self.fc1(fusion_input))
        action = self.max_action * torch.tanh(self.fc2(x))
        return action