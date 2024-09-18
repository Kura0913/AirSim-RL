import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.td3.policies import TD3Policy
from torchvision import models
import json

class MixedInputPPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(MixedInputPPOPolicy, self).__init__(
            observation_space, action_space, lr_schedule, net_arch=[64, 64], **kwargs
        )
        # get config setting
        self.config = self.load_config()
        self.point_cloud_numbers = self.config['point_numbers'] # number of point clouds
        self.resize = self.config["resize"] # size of depth image

        # CNN for depth image
        self.cnn = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)  # Using ResNet152 as it's smaller
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # Remove the final fully connected layer
        # Get the number of features from the CNN
        dummy_input = torch.zeros(1, 3, self.resize[0], self.resize[1])
        with torch.no_grad():
            cnn_output = self.cnn(dummy_input)
        self.cnn_feature_dim = cnn_output.view(1, -1).size(1)

        # Fully connected layer for CNN output
        self.cnn_fc = nn.Linear(self.cnn_feature_dim, 256)

        # MLP for point cloud data
        self.point_cloud_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.point_cloud_numbers * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Fully connected layer for drone position
        self.drone_fc = nn.Sequential(
            nn.Linear(3, 32),  # 3 dimensions for drone_position
            nn.ReLU()
        )

        # Fully connected layer for target position
        self.target_fc = nn.Sequential(
            nn.Linear(3, 32),  # 3 dimensions for target position
            nn.ReLU()
        )

        # Combined feature layer
        self.fc_combined = nn.Sequential(
            nn.Linear(self.cnn_feature_dim + 128 + 32 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Policy network (outputs mean actions)
        self.policy_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space.shape[0])  # For continuous actions
        )

        # Value network (outputs state value)
        self.value_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # State value
        )

        # Standard deviation parameter for Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(self.action_space.shape[0]), requires_grad=True)

    def load_config(self):
        with open('config.json', 'r') as file:
            return json.load(file)
        
    def forward(self, obs: torch.Tensor) -> tuple:
        try:
            point_cloud_size = self.point_cloud_numbers * 3
            depth_image_size = self.resize[0] * self.resize[1]
            drone_position_size = 3
            target_position_size = 3

            total_size = point_cloud_size + depth_image_size + drone_position_size + target_position_size
            if obs.size(1) != total_size:
                raise ValueError(f"Expected observation size {total_size}, but got {obs.size(1)}")

            # Extract individual parts
            obs_split = torch.split(obs, [point_cloud_size, depth_image_size, drone_position_size, target_position_size], dim=1)
            point_cloud = obs_split[0].view(-1, self.point_cloud_numbers, 3)  # Reshape to (batch_size, point_cloud_numbers, 3)
            depth_image = obs_split[1].view(-1, 1, self.resize[0], self.resize[1])  # Reshape to (batch_size, 1, resize[0], resize[1])
            drone_position = obs_split[2]  # (batch_size, 3)
            target_position = obs_split[3]  # (batch_size, 3)

            # Convert depth_image to float32
            depth_image = depth_image.float()
            point_cloud = point_cloud.float()
            drone_position = drone_position.float()
            target_position = target_position.float()

            # Process depth image
            depth_image = depth_image.expand(-1, 3, -1, -1)  # Expand to 3 channels
            cnn_features = self.cnn(depth_image)
            cnn_features = cnn_features.view(cnn_features.size(0), -1)  # Flatten
            # cnn_features = F.relu(self.cnn_fc(cnn_features.view(cnn_features.size(0), -1)))  # Flatten and apply FC

            # Process point cloud
            point_cloud_features = self.point_cloud_mlp(point_cloud.view(-1, self.point_cloud_numbers * 3))

            # Process target position
            drone_position_features = self.drone_fc(drone_position)

            # Process target position
            target_position_features = self.target_fc(target_position)

            # Combine features
            combined_features = torch.cat((cnn_features, point_cloud_features, drone_position_features, target_position_features), dim=1)
            combined_features = self.fc_combined(combined_features)

            # Policy network output
            mean_actions = self.policy_net(combined_features)
            log_std = self.log_std.expand_as(mean_actions)
            std = torch.exp(log_std)
            distribution = DiagGaussianDistribution(mean_actions.shape[-1]).proba_distribution(mean_actions, std)

            # Sample actions
            actions = distribution.get_actions()
            log_probs = distribution.log_prob(actions)

            # Value network output
            values = self.value_net(combined_features)

            return actions, values, log_probs
        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise

    def _predict(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.forward(observation)
        return features
    

class MixedInputDDPGPolicy(TD3Policy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(MixedInputDDPGPolicy, self).__init__(
            observation_space, action_space, lr_schedule, net_arch=[64, 64], **kwargs
        )
        # get config setting
        self.config = self.load_config()
        self.point_cloud_numbers = self.config['point_numbers'] # number of point clouds
        self.resize = self.config["resize"] # size of depth image
        # CNN for depth image
        self.cnn = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)  # Using ResNet152 as it's smaller
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # Remove the final fully connected layer
        # Get the number of features from the CNN
        dummy_input = torch.zeros(1, 3, self.resize[0], self.resize[1])
        with torch.no_grad():
            cnn_output = self.cnn(dummy_input)
        self.cnn_feature_dim = cnn_output.view(1, -1).size(1)

        # MLP for point cloud data
        self.point_cloud_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.point_cloud_numbers * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Fully connected layer for drone position
        self.drone_fc = nn.Sequential(
            nn.Linear(3, 32),  # 3 dimensions for drone_position
            nn.ReLU()
        )

        # Fully connected layer for target position
        self.target_fc = nn.Sequential(
            nn.Linear(3, 32),  # 3 dimensions for target position
            nn.ReLU()
        )

        # Combined feature layer
        self.fc_combined = nn.Sequential(
            nn.Linear(self.cnn_feature_dim + 128 + 32 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

    def load_config(self):
        with open('config.json', 'r') as file:
            return json.load(file)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Assuming obs is a flattened tensor containing all data
        point_cloud_size = self.point_cloud_numbers * 3
        depth_image_size = self.resize[0] * self.resize[1]
        drone_position_size = 3
        target_position_size = 3

        # Check that the total size matches
        total_size = point_cloud_size + depth_image_size + drone_position_size + target_position_size
        if obs.size(1) != total_size:
            raise ValueError(f"Expected observation size {total_size}, but got {obs.size(1)}")

        # Extract individual parts
        obs_split = torch.split(obs, [point_cloud_size, depth_image_size, drone_position_size, target_position_size], dim=1)
        point_cloud = obs_split[0].view(-1, self.point_cloud_numbers, 3)  # Reshape to (batch_size, 512, 3)
        depth_image = obs_split[1].view(-1, 1, self.resize[0], self.resize[1])  # Reshape to (batch_size, 1, 64, 64)
        drone_position = obs_split[2]  # (batch_size, 3)
        target_position = obs_split[3]  # (batch_size, 3)

        # Convert depth_image to float32
        depth_image = depth_image.float()
        point_cloud = point_cloud.float()
        drone_position = drone_position.float()
        target_position = target_position.float()

        # Process depth image
        depth_image = depth_image.expand(-1, 3, -1, -1)  # Expand to 3 channels
        cnn_features = self.cnn(depth_image)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)  # Flatten

        # Process point cloud
        point_cloud_features = self.point_cloud_mlp(point_cloud)

        # Process drone position
        drone_position_features = self.drone_fc(drone_position)

        # Process target position
        target_position_features = self.target_fc(target_position)

        # Combine features
        combined_features = torch.cat((cnn_features, point_cloud_features, drone_position_features, target_position_features), dim=1)
        combined_features = self.fc_combined(combined_features)

        return combined_features

    def _predict(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.forward(observation)
        return features