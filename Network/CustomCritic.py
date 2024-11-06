import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from typing import List, Type, Tuple
import gymnasium.spaces as spaces


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
        self.n_critics = n_critics
        self.q_networks = nn.ModuleList()

        for idx in range(n_critics):
            q_net = QNetwork(features_dim, action_dim)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)


class QNetwork(nn.Module):
    def __init__(self, features_dim: int, action_dim: int):
        super().__init__()
        
        self.features_net = nn.Sequential(
            nn.Linear(features_dim, 48),
            nn.ReLU()
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, 48),
            nn.Tanh()
        )
        
        # Q net structure
        self.q_net = nn.Sequential(
            nn.Linear(48, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='linear')
            nn.init.zeros_(module.bias)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        # split features adn actions
        features = x[:, :-self.action_net[0].in_features]  # get features
        actions = x[:, -self.action_net[0].in_features:]   # get actions
        
        # Process features to 48 dimensions
        processed_features = self.features_net(features)
        
        # Process actions to 48 dimensions
        processed_actions = self.action_net(actions)
        
        # Combine processed features mand processed actions
        combined = processed_features + processed_actions
        
        return self.q_net(combined)