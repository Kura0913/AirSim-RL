from typing import Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy, Actor
from stable_baselines3.common.policies import ContinuousCritic
from PolicyComponent.CustomActor import CustomActor
from PolicyComponent.CustomCritic import CustomCritic

class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        self.config = kwargs.pop('config')
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomCritic(**critic_kwargs).to(self.device)