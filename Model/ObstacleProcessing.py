import torch
import torch.nn as nn
from torchvision import models

class ObstacleProcessing(nn.Module):
    def __init__(self, point_cloud_numbers, resize, mode):
        super(ObstacleProcessing, self).__init__()
        self.mode = mode
        self.point_cloud_mlp = nn.Sequential(
            nn.Linear(point_cloud_numbers * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)
        
    def forward(self, point_cloud, depth_image):
        if self.mode in ['lidar_mode', 'all_sensors']:
            pc_features = self.point_cloud_mlp(point_cloud.view(point_cloud.size(0), -1))
        else:
            pc_features = torch.zeros(point_cloud.size(0), 128, device=point_cloud.device)

        if self.mode in ['camera_mode', 'all_sensors']:
            depth_image = depth_image.permute(0, 3, 1, 2)
            depth_features = self.cnn(depth_image.expand(-1, 3, -1, -1))
        else:
            depth_features = torch.zeros(depth_image.size(0), 128, device=depth_image.device)

        return torch.cat((pc_features, depth_features), dim=1)