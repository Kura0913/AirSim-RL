from stable_baselines3 import PPO, DDPG
from Model.NetPolicies import MixedInputPPOPolicy, MixedInputDDPGPolicy
from Model.BaseCallback import CustomCallback
from datetime import datetime
import json
import torch
import os

class RLModel:
    def __init__(self, config, env, device):
        self.config = config
        self.env = env
        self.device = device
        self.model = self.initialize_model()
        

    def initialize_model(self):
        if self.config['rl_algorithm'] == 'PPO':
            return PPO(
                MixedInputPPOPolicy,
                self.env,
                verbose=1,
                learning_rate=self.config['learning_rate'],
                device=self.device
            )
        elif self.config['rl_algorithm'] == 'DDPG':
            return DDPG(
                MixedInputDDPGPolicy,
                self.env,
                verbose=1,
                learning_rate=self.config['learning_rate'],
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported RL algorithm: {self.config['rl_algorithm']}")

    def train(self):        
        mode = self.config["mode"]
        save_dir = self.config["train"][mode]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, timestamp)
        os.makedirs(save_path, exist_ok=True)
        save_path = save_path + '/'
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
        # load model weight
        if model_path and model_path.endswith('.pth'):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
        else:
            print(f"Invalid model path: {model_path}")