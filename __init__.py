from Agent import available_agent
from Callback import available_callbacks
from Env import available_envs
from Network import available_networks
from Policy import available_policy
from FeatureExtractor import available_policy_component

available_classes = {
    **available_callbacks,
    **available_agent,
    **available_envs,
    **available_networks,
    **available_policy, 
    **available_policy_component
}