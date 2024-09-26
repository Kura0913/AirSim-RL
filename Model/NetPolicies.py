from Model.ObstacleProcessing import ObstacleProcessing
from Model.PositionProcessing import PositionProcessing
from Model.VelocityAdjustment import VelocityAdjustment
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.td3.policies import TD3Policy
import torch
import torch.nn as nn
import json
import numpy as np

class MixedInputPPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(MixedInputPPOPolicy, self).__init__(
            observation_space, action_space, lr_schedule, net_arch=[64, 64], **kwargs
        )
        self.config = self.load_config()
        self.point_cloud_numbers = self.config['point_numbers']
        self.resize = self.config["resize"]
        self.mode = self.config["mode"]
        
        # Position Processing
        self.position_processing = PositionProcessing()
        
        # Obstacle Processing
        self.obstacle_processing = ObstacleProcessing(self.point_cloud_numbers, self.resize, self.mode)
        
        # Velocity Adjustment
        self.velocity_adjustment = VelocityAdjustment()

        # Value network (outputs state value)
        self.value_net = self.create_value_net()

        # Standard deviation parameter for Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(self.action_space.shape[0]), requires_grad=True)

    def load_config(self):
        with open('config.json', 'r') as file:
            return json.load(file)
        
    def create_value_net(self):
        return nn.Sequential(
            nn.Linear(291, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _process_observation(self, obs):
        if isinstance(obs, dict):
            return obs
        batch_size = 1 if obs.dim() == 1 else obs.size(0)

        point_cloud_size = self.point_cloud_numbers * 3
        depth_image_size = self.resize[0] * self.resize[1]
        
        point_cloud, depth_image, positions = torch.split(obs.view(batch_size, -1), [point_cloud_size, depth_image_size, 6], dim=1)
        
        point_cloud = point_cloud.view(batch_size, self.point_cloud_numbers, 3)
        depth_image = depth_image = depth_image.view(batch_size, 1, *self.resize)
        drone_position, target_position = torch.split(positions, [3, 3], dim=1)
        
        return {
            "point_cloud": point_cloud,
            "depth_image": depth_image,
            "drone_position": drone_position,
            "target_position": target_position
        }

    def forward(self, obs):
        obs_dict = self._process_observation(obs)
        point_cloud = obs_dict["point_cloud"]
        depth_image = obs_dict["depth_image"]
        drone_position = obs_dict["drone_position"]
        target_position = obs_dict["target_position"]

        # Process positions to get initial velocity
        initial_velocity = self.position_processing(drone_position, target_position)
        
        # Process obstacle information
        obstacle_features = self.obstacle_processing(point_cloud, depth_image)
        
        # Adjust velocity based on obstacle information
        final_velocity = self.velocity_adjustment(initial_velocity, obstacle_features)
        
        # Compute value
        combined_features = torch.cat((initial_velocity, obstacle_features), dim=1)
        value = self.value_net(combined_features)

        # Create distribution
        log_std = self.log_std.expand_as(final_velocity)
        std = torch.exp(log_std)
        distribution = DiagGaussianDistribution(final_velocity.shape[-1]).proba_distribution(final_velocity, std)
        
        # Sample actions
        actions = distribution.get_actions()
        log_probs = distribution.log_prob(actions)

        return actions, value, log_probs

    def evaluate_actions(self, obs: dict, actions: torch.Tensor) -> tuple:
        point_cloud = obs["point_cloud"]
        depth_image = obs["depth_image"]
        drone_position = obs["drone_position"]
        target_position = obs["target_position"]

        initial_velocity = self.position_processing(drone_position, target_position)
        obstacle_features = self.obstacle_processing(point_cloud, depth_image)
        final_velocity = self.velocity_adjustment(initial_velocity, obstacle_features)

        combined_features = torch.cat((initial_velocity, obstacle_features), dim=1)
        value = self.value_net(combined_features)

        log_std = self.log_std.expand_as(final_velocity)
        std = torch.exp(log_std)
        distribution = DiagGaussianDistribution(final_velocity.shape[-1]).proba_distribution(final_velocity, std)

        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return value, log_probs, entropy

    def _predict(self, observation: dict, deterministic: bool = False) -> torch.Tensor:
        actions, _, _ = self.forward(observation)
        if deterministic:
            return actions.mean
        actions = actions.clamp(-1, 1)  # Clamp actions to [-1, 1]
        actions = (actions * 100).round() / 100  # Round to nearest 0.01
        return actions
    

class MixedInputDDPGPolicy(TD3Policy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(MixedInputDDPGPolicy, self).__init__(
            observation_space, action_space, lr_schedule, net_arch=[64, 64], **kwargs
        )
        self.config = self.load_config()
        self.point_cloud_numbers = self.config['point_numbers']
        self.resize = self.config["resize"]
        self.mode = self.config["mode"]

        # Position Processing
        self.position_processing = PositionProcessing()
        
        # Obstacle Processing
        self.obstacle_processing = ObstacleProcessing(self.point_cloud_numbers, self.resize, self.mode)
        
        # Velocity Adjustment
        self.velocity_adjustment = VelocityAdjustment()

    def load_config(self):
        with open('config.json', 'r') as file:
            return json.load(file)

    def forward(self, obs) -> torch.Tensor:
        if isinstance(obs, dict):
            obs_dict = obs
        else:
            obs_dict = self._process_observation(obs)

        point_cloud = obs_dict["point_cloud"]
        depth_image = obs_dict["depth_image"]
        drone_position = obs_dict["drone_position"]
        target_position = obs_dict["target_position"]

        # Process positions to get initial velocity
        initial_velocity = self.position_processing(drone_position, target_position)
        
        # Process obstacle information
        obstacle_features = self.obstacle_processing(point_cloud, depth_image)
        
        # Adjust velocity based on obstacle information
        final_velocity = self.velocity_adjustment(initial_velocity, obstacle_features)

        return final_velocity

    def _predict(self, observation: dict, deterministic: bool = False) -> torch.Tensor:
        if not isinstance(observation, dict):
            observation = self._process_observation(observation)
        actions = self.forward(observation)
        if deterministic:
            return actions.mean
        actions = actions.clamp(-1, 1)  # Clamp actions to [-1, 1]
        actions = (actions * 100).round() / 100  # Round to nearest 0.01
        return actions
    
    def _process_observation(self, obs):
        if isinstance(obs, dict):
            return obs
        batch_size = 1 if obs.dim() == 1 else obs.size(0)

        point_cloud_size = self.point_cloud_numbers * 3
        depth_image_size = self.resize[0] * self.resize[1]
        
        point_cloud, depth_image, positions = torch.split(obs.view(batch_size, -1), [point_cloud_size, depth_image_size, 6], dim=1)
        
        point_cloud = point_cloud.view(batch_size, self.point_cloud_numbers, 3)
        depth_image = depth_image = depth_image.view(batch_size, 1, *self.resize)
        drone_position, target_position = torch.split(positions, [3, 3], dim=1)
        
        return {
            "point_cloud": point_cloud,
            "depth_image": depth_image,
            "drone_position": drone_position,
            "target_position": target_position
        }