import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from Model.FeaturesExtractor.PointNetFeatureExtractor import PointNetFeatureExtractor

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, config):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim=1)
        
        self.config = config
        self.mode = config['mode']
        
        # Calculate the sizes of different parts of the observation
        self.point_cloud_size = self.config['point_numbers'] * 3
        self.depth_image_size = self.config['resize'][0] * self.config['resize'][1]
        self.position_size = 6  # drone_pos (3) + target_pos (3)

        # SubNet for position data
        self.position_net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # SubNet for point cloud (using PointNet)
        self.pointnet = PointNetFeatureExtractor(num_points=config['point_numbers'])
        
        # SubNet for depth image (using a smaller ResNet)
        self.depth_net = nn.Sequential(
            resnet50(weights=ResNet50_Weights.DEFAULT),
            nn.Linear(1000, 256)  # Adjust ResNet output to match other subnets
        )
        # Modify the first conv layer to accept single-channel input
        self.depth_net[0].conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Calculate subnet output dimensions
        self.position_dim = 256
        self.pointcloud_dim = 1024  # PointNet typically outputs 1024-dim vector
        self.depth_dim = 256
        
        # Integration network
        total_features_dim = self.position_dim + self.pointcloud_dim + self.depth_dim
        self.integration_net = nn.Sequential(
            nn.Linear(total_features_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self._features_dim = 128  # Output size of integration_net

    @property
    def features_dim(self):
        return self._features_dim

    def forward(self, observations):
        # Use torch.split to correctly split the tensor
        point_cloud, depth_image, positions = torch.split(
            observations,
            [self.point_cloud_size, self.depth_image_size, self.position_size],
            dim=1
        )
        
        # Reshape the split tensors
        point_cloud = point_cloud.view(-1, self.config['point_numbers'], 3).transpose(1, 2)
        depth_image = depth_image.view(-1, 1, *self.config['resize'])
        
        # Process each part through its subnet
        position_features = self.position_net(positions)
        pointcloud_features = self.pointnet(point_cloud)
        depth_features = self.depth_net(depth_image)
        
        # Concatenate all features
        combined_features = torch.cat([position_features, pointcloud_features, depth_features], dim=1)
        
        # Pass through integration network
        integrated_features = self.integration_net(combined_features)
        
        return integrated_features

    def compute_direction(self, drone_pos, target_pos):
        direction = target_pos - drone_pos
        distance = torch.norm(direction, dim=1, keepdim=True)
        normalized_direction = direction / (distance + 1e-8)  # avoid division by zero
        return normalized_direction, distance