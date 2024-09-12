from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from Model.Network import CustomNetwork
from datetime import datetime
import os

class RLModel:
    def __init__(self, config, env):
        self.config = config
        self.env = DummyVecEnv([lambda: env])  # Wrap Airsim environment as vectorized environment
        self.model = self.initialize_model()

    def initialize_model(self):
        if self.config['rl_algorithm'] == 'PPO':
            return PPO(CustomNetwork, self.env, verbose=1)
        elif self.config['rl_algorithm'] == 'DDPG':
            return DDPG(CustomNetwork, self.env, verbose=1)
        else:
            raise ValueError(f"Unsupported RL algorithm: {self.config['rl_algorithm']}")

    def train(self):
        mode = self.config["mode"]
        save_dir = self.config["train"][mode]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, timestamp)
        os.makedirs(save_path, exist_ok=True) 
        self.model.learn(total_timesteps=self.config['episodes'] * self.config['max_steps'])
        self.model.save("model_path")  # Could use a better path

    def test(self):
        obs = self.env.reset()
        for _ in range(self.config['max_steps']):
            action, _ = self.model.predict(obs)
            obs, rewards, done, info = self.env.step(action)
            if done:
                break
    