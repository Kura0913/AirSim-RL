from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium.spaces as spaces
import torch as th
from torch import nn
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(
        self,
        config,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):
        self.config = kwargs.pop('config')
        super(CustomPPOPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


class CustomNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super(CustomNetwork, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Hardswish(),
            nn.Linear(256, 256),
            nn.Hardswish(),
            nn.Linear(256, 256),
            nn.Hardswish()
        )
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.policy_net(features), self.value_net(features)