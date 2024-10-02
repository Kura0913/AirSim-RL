from stable_baselines3 import PPO, DDPG
from Model.MixedInputPPOPolicy import MixedInputPPOPolicy
from Model.MixedInputDDPGPolicy import MixedInputDDPGPolicy
from Model.CustomCallback import CustomCallback
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
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
                gamma=0.9999,
                n_steps = 2048,
                batch_size=256,
                ent_coef=0.01,
                learning_rate=self.config['learning_rate'],
                device=self.device,
                policy_kwargs={"config": self.config}
            )
        elif self.config['rl_algorithm'] == 'DDPG':
            return DDPG(
                MixedInputDDPGPolicy,
                self.env,
                verbose=1,
                gamma=0.9999,
                learning_rate=self.config['learning_rate'],
                device=self.device,
                replay_buffer_class=ReplayBuffer,
                policy_kwargs={"config": self.config}
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
        callback = CustomCallback(self.config['rl_algorithm'], save_path, verbose=1)
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