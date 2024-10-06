import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from typing import Dict, Any
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from Model.FeaturesExtractor.PointNetFeatureExtractor import PointNetFeatureExtractor

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, config: Dict[str, Any]):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim=1)
        self.config = config
        
        # Calculate sizes
        self.point_cloud_size = config['point_numbers'] * 3
        self.depth_image_size = config['resize'][0] * config['resize'][1]
        self.position_size = 6  # drone_pos (3) + target_pos (3)

        # Position processing (always active)
        self.position_net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        
        # Point cloud processing (using PointNet)
        self.pointnet = PointNetFeatureExtractor(num_points=config['point_numbers'])
        
        # Depth image processing
        self.depth_net = nn.Sequential(
            resnet18(weights=ResNet18_Weights.DEFAULT),
            nn.Linear(1000, 256)
        )
        self.depth_net[0].conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Calculate maximum total features dimension
        self.max_features_dim = 35 + 1024 + 256  # position_net + pointnet + depth

        # Integration network
        self.integration_net = nn.Sequential(
            nn.Linear(self.max_features_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self._features_dim = 128

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observations
        point_cloud, depth_image, positions = torch.split(
            observations,
            [self.point_cloud_size, self.depth_image_size, self.position_size],
            dim=1
        )
        
        # Process position (always active)
        drone_pos, target_pos = torch.split(positions, [3, 3], dim=1)
        position_features = self.process_position(drone_pos, target_pos)
        
        # Process point cloud
        point_cloud = point_cloud.view(-1, self.config['point_numbers'], 3).transpose(1, 2)
        pointcloud_features = self.pointnet(point_cloud)
        
        # Process depth image
        depth_image = depth_image.view(-1, 1, *self.config['resize'])
        depth_features = self.depth_net(depth_image)

        # Combine features based on mode
        if self.config['mode'] == 'lidar_mode':
            combined_features = torch.cat([position_features, pointcloud_features, torch.zeros_like(depth_features)], dim=1)
        elif self.config['mode'] == 'camera_mode':
            combined_features = torch.cat([position_features, torch.zeros_like(pointcloud_features), depth_features], dim=1)
        else:  # 'all_sensors'
            combined_features = torch.cat([position_features, pointcloud_features, depth_features], dim=1)
        
        # Integrate features
        integrated_features = self.integration_net(combined_features)
        
        return integrated_features
    
    def process_position(self, drone_pos, target_pos):
        combined_pos = torch.cat((drone_pos, target_pos), dim=-1)
        diff_vector = target_pos - drone_pos
        initial_velocity = F.normalize(diff_vector, dim=-1)
        processed_pos = F.relu(self.position_net(combined_pos))

        return torch.cat((initial_velocity, processed_pos), dim=-1)
    
    