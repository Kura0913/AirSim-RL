import json
from Env import AirSimMultiDroneEnv
from Model.RLModel import RLModel
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import numpy as np
import matplotlib.pyplot as plt

def save_training_data(save_path):
    rewards = np.load(save_path + "episode_rewards.npy")
    losses = np.load(save_path + "episode_losses.npy")

    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Episode Rewards Over Time')
    plt.savefig(save_path + 'episode_rewards.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(losses)
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.savefig(save_path + 'episode_losses.png')
    plt.close()

def load_config():
    with open('config.json', 'r') as file:
        return json.load(file)

def make_env(env_id, config, drone_list):
    """Create subprocess environment function"""
    def _init():
        env = AirSimMultiDroneEnv(config, drone_list)
        return env
    return _init

def get_drone_names(settings_path):
        with open(settings_path, "r") as file:
            data = json.load(file)
        drone_names = list(data.get("Vehicles", {}).keys())  # Get all keys of "Vehicles" as drone names
        print(f"drone list: {drone_names}")
        return drone_names

def run_training():
    # load config
    config = load_config()
    
    # create env for multiple drone
    num_drones = get_drone_names(os.path.expanduser("~\\Documents\\AirSim\\settings.json"))
    env = SubprocVecEnv([make_env(i, config, num_drones) for i in range(len(num_drones))])  # Create a multi-process environment

    # initial RL model
    rl_model = RLModel(config, env)

    # start training
    print(f"Training with RL algorithm: {config['rl_algorithm']}")
    save_path = rl_model.train()
    print("Training completed.")
    save_training_data(save_path)

if __name__ == "__main__":
    run_training()