from typing import Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy, Actor
from stable_baselines3.common.policies import ContinuousCritic
from typing import List, Type
import gymnasium.spaces as spaces
import torch.nn as nn
from stable_baselines3.common.preprocessing import get_action_dim
from Network.ActionNetwork import ActionNetwork
from Network.ValueNetwork import ValueNetwork

class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomCritic(**critic_kwargs).to(self.device)
    
class CustomActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            net_arch=net_arch,
            features_dim=features_dim,
        )

        self.features_dim = features_dim
        self.action_dim = get_action_dim(self.action_space)

        actor_net = ActionNetwork(self.features_dim, self.action_dim)
        self.mu = actor_net

class CustomCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(ContinuousCritic, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        self.share_features_extractor = share_features_extractor

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []
        for idx in range(n_critics):
            q_net = ValueNetwork(features_dim + action_dim)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)