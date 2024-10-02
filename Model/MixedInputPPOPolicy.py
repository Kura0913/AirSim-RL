from stable_baselines3.common.policies import ActorCriticPolicy
from Model.FeaturesExtractor.CustomFeaturesExtractor import CustomFeaturesExtractor
from torch.distributions import Normal
import torch
from typing import Dict, Any, Tuple
import numpy as np

class MixedInputPPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, config, **kwargs):
        self.config = config
        super(MixedInputPPOPolicy, self).__init__(
            observation_space, 
            action_space, 
            lr_schedule, 
            # Use net_arch parameter if provided in kwargs, otherwise use default
            net_arch=kwargs.get('net_arch', dict(pi=[256, 128], vf=[256, 128])),
            **kwargs
        )

    def _build_mlp_extractor(self) -> None:
        self.features_extractor = CustomFeaturesExtractor(self.observation_space, self.config)
        self.features_dim = self.features_extractor.features_dim
        super()._build_mlp_extractor()

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        
        # Get action distribution
        mean_actions = self.action_net(latent_pi)
        
        # We assume the action std is fixed here, but you might want to make it learnable
        action_std = torch.ones_like(mean_actions) * 0.5
        distribution = Normal(mean_actions, action_std)
        
        if deterministic:
            actions = mean_actions
        else:
            actions = distribution.sample()
        
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        
        return actions, values, log_prob
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        
        latent_vf = self.mlp_extractor.forward_critic(features)        
        values = self.value_net(latent_vf)
        
        return values
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions).sum(axis=-1)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy().sum(axis=-1)
        return values, log_prob, entropy

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Normal:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        action_std = torch.ones_like(mean_actions) * 0.5  # You might want to make this learnable
        return Normal(mean_actions, action_std)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        actions, _, _ = self.forward(observation, deterministic)
        return actions