from Network.CustomFeaturesExtractor import CustomFeaturesExtractor
from Policy.CustomTD3Policy import CustomTD3Policy
from stable_baselines3 import DDPG, PPO

class DDPGAgent:
    def __init__(self, env, config):
        self.config = config
        policy_kwargs = dict(
        features_extractor_class=CustomFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512+32+32),
        net_arch=dict(pi=[256, 256, 256], qf=[256, 256]),
        config=config
        )
        self.model = DDPG(
            CustomTD3Policy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_starts=20000,
            gamma=0.9999,
            learning_rate=self.config["learning_rate"],
            device = self.config['device']

        )

    def train(self, total_timesteps, callback):
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path):
        self.model.save(path)

class PPOAgent:
    def __init__(self, env, config):
        self.config = config
        policy_kwargs = dict(
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=32),
            net_arch=[dict(pi=[256, 256, 256], vf=[256, 256])],
            config=config
        )
        self.model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            gamma=0.9999,
            learning_rate=self.config["learning_rate"],
            device = self.config['device']
        )

    def train(self, total_timesteps, callback):
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path):
        self.model.save(path)