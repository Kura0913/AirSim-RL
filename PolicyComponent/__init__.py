from PolicyComponent.DDPGFeaturesExtractor import DDPGFeaturesExtractor
from PolicyComponent.PPOFeaturesExtractor import PPOFeaturesExtractor


available_policy_component = {
    'DDPGFeaturesExtractor': DDPGFeaturesExtractor,
    'PPOFeaturesExtractor': PPOFeaturesExtractor
}