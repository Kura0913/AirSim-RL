from FeatureExtractor.DDPGFeaturesExtractor import DDPGFeaturesExtractor
from FeatureExtractor.PPOFeaturesExtractor import PPOFeaturesExtractor


available_policy_component = {
    'DDPGFeaturesExtractor': DDPGFeaturesExtractor,
    'PPOFeaturesExtractor': PPOFeaturesExtractor
}