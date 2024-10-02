from Model.ObstacleProcessing import ObstacleProcessing
from Model.PositionProcessing import PositionProcessing
from Model.VelocityAdjustment import VelocityAdjustment
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.td3.policies import TD3Policy
import torch
import torch.nn as nn
import json


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

        self.mlp_extractor = MlpExtractor(
            feature_dim=291,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device
        )

        self.action_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space.shape[0])
        )

        # Standard deviation parameter for Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(self.action_space.shape[0]))

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            291,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def load_config(self):
        with open('config.json', 'r') as file:
            return json.load(file)
        
    def create_value_net(self):
        return nn.Sequential(
            nn.Linear(64, 128),
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
    
    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
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

        return combined_features
    
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        mean_actions = self.action_net(latent_pi)
        
        if mean_actions.dim() == 1:
            mean_actions = mean_actions.unsqueeze(0)
        log_std = self.log_std.expand_as(mean_actions)

        return self.action_dist.proba_distribution(mean_actions, log_std)

    def forward(self, obs, deterministic: bool = False):
        combined_features = self.extract_features(obs)
        
        latent_pi, latent_vf = self.mlp_extractor(combined_features)

        distribution = self._get_action_dist_from_latent(latent_pi)

        actions = distribution.get_actions(deterministic=deterministic)

        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple:
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()

        return values, log_prob, entropy
    
    # def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
    #     features = self.extract_features(observation)
    #     latent_pi, _ = self.mlp_extractor(features)
    #     mean_actions = self.action_net(latent_pi)
    #     return self.action_dist.get_actions(mean_actions, deterministic=deterministic)

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