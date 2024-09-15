from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from Model.BaseCallback import CustomCallback
from Model.Network import CustomNetwork
from datetime import datetime
import torch.optim as optim
from torch import nn
import json
import torch
import os

class RLModel:
    def __init__(self, config, env):
        self.config = config
        self.env = DummyVecEnv([lambda: env])  # Wrap Airsim environment as vectorized environment
        self.model = self.initialize_model()
        # Define the loss function (e.g., MSE for DQN)
        self.criterion = nn.MSELoss()
        
        # Define the optimizer (e.g., Adam optimizer)
        self.optimizer = optim.Adam(self.model.get_parameters(), lr=config['learning_rate'])

    def initialize_model(self):
        if self.config['rl_algorithm'] == 'PPO':
            return PPO(CustomNetwork, self.env, verbose=1, learning_rate=self.config['learning_rate'])
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
        callback = CustomCallback(self.config['rl_algorithm'], save_path)
        self.model.learn(total_timesteps=self.config['timesteps'], callback=callback)
        self.model.save(save_path + "model.pth")  # Could use a better path
        with open(save_path + 'config.json', 'w') as f:
            json.dump(self.config, f)

        return save_path

    def test(self):
        obs = self.env.reset()
        for _ in range(self.config['max_steps']):
            action, _ = self.model.predict(obs)
            obs, rewards, done, info = self.env.step(action)
            if done:
                break

    def load_model(self, model_path):
        # 加載模型權重
        if model_path and model_path.endswith('.pth'):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
        else:
            print(f"Invalid model path: {model_path}")