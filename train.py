from Env import AirsimEnv
from Agent import DDPGAgent, PPOAgent
from CustomCallback import CustomCallback
from datetime import datetime
import json
import os

def train_ddpg(drone_name, config, folder_name):
    env = AirsimEnv(drone_name, config)
    agent = DDPGAgent(env, config)
    callback_class = CustomCallback(config, folder_name)
    agent.train(total_timesteps=config['episodes'] * config['max_steps'], callback=callback_class)
    agent.save(f"{config['train']}{folder_name}/ddpg_model.pth")

def train_ppo(drone_name, config, folder_name):
    env = AirsimEnv(drone_name, config)
    agent = PPOAgent(env, config)
    callback_class = CustomCallback(config, folder_name)
    agent.train(total_timesteps=config['episodes'] * config['max_steps'], callback=callback_class)
    agent.save(f"{config['train']}{folder_name}/ppo_model.pth")

def load_config():
    with open('config.json', 'r') as file:
        return json.load(file)

def load_drone_name():
    drone_list = get_drone_names(os.path.expanduser("~\\Documents\\AirSim\\settings.json"))
    drone_name = drone_list[0]  # Use only the first drone for single drone training

    return drone_name

def get_drone_names(settings_path):
        with open(settings_path, "r") as file:
            data = json.load(file)
        drone_names = list(data.get("Vehicles", {}).keys())  # Get all keys of "Vehicles" as drone names
        print(f"drone list: {drone_names}")
        return drone_names

def main():
    drone_name = load_drone_name()
    config = load_config()
    folder_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    if config["rl_algorithm"] == "DDPG":
        train_ddpg(drone_name, config, folder_name)
    elif config["rl_algorithm"] == "PPO":
        train_ppo(drone_name, config, folder_name)

if __name__ == "__main__":
    main()