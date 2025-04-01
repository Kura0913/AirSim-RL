from stable_baselines3.common.policies import ActorCriticPolicy
from Network.ActionNetwork import ActionNetwork
from Network.ValueNetwork import ValueNetwork
from stable_baselines3.common.preprocessing import get_action_dim

class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPPOPolicy, self).__init__(*args, **kwargs)

        self.action_dim = get_action_dim(self.action_space)
        self.action_net = ActionNetwork(self.features_dim, self.action_dim)
        self.value_net = ValueNetwork(self.features_dim)