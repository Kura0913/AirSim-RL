import json
from Env import AirSimMultiDroneEnv, AirSimEnv
from Model.RLModel import RLModel
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

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

def make_env(env_id, config, drone_list, device):
    """Create subprocess environment function"""
    def _init():
        env = AirSimMultiDroneEnv(config, drone_list, device)
        return env
    return _init

def get_drone_names(settings_path):
        with open(settings_path, "r") as file:
            data = json.load(file)
        drone_names = list(data.get("Vehicles", {}).keys())  # Get all keys of "Vehicles" as drone names
        print(f"drone list: {drone_names}")
        return drone_names

def run_training_single():
    # load config
    config = load_config()
    if config['device'] == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)  # Output: "cuda" if GPU is available, otherwise "cpu"

    # Create a single drone environment
    drone_list = get_drone_names(os.path.expanduser("~\\Documents\\AirSim\\settings.json"))
    drone_name = drone_list[0]  # Use only the first drone for single drone training
    env = AirSimEnv(drone_name, config, device)  # Directly create the environment

    # Initialize RL model
    rl_model = RLModel(config, env, device)

    # Start training
    print(f"Training with RL algorithm: {config['rl_algorithm']}")
    save_path = rl_model.train()
    print("Training completed.")
    save_training_data(save_path)

if __name__ == "__main__":
    run_training_single()