from Env import AirsimEnv
from Agent import DDPGAgent, PPOAgent
from CustomCallback import DDPGCustomCallback, PPOCustomCallback
from datetime import datetime
import json
import os

def train_ddpg(drone_name, config, folder_name):
    env = AirsimEnv(drone_name, config)
    agent = DDPGAgent(env, config)

    # Create full path for saving
    save_path = os.path.join(config['train'], folder_name)
    os.makedirs(save_path, exist_ok=True)

    # Load pretrained model if specified
    if config['load_model']:
        try:
            print(f"Loading pretrained model from {config['model_load_path']}")
            model_code = input("Please enter the model code (press Enter to use default): ").strip()
            if len(model_code) <= 0:
                agent.load(config['model_load_path'])
            else:
                agent.load(config['model_load_path'], model_code)
            # Save pretrained model information
            save_pretrain_info(save_path, config['model_load_path'])
            print("Pretrained model loaded successfully")
        except Exception as e:
            print(f"Error loading pretrained model: {str(e)}")
            print("Training will continue with freshly initialized model")

    callback_class = DDPGCustomCallback(config, folder_name)
    agent.train(total_timesteps=config['episodes'] * config['max_steps'], callback=callback_class)
    agent.save(f"{config['train']}{folder_name}/")

def train_ppo(drone_name, config, folder_name):
    env = AirsimEnv(drone_name, config)
    agent = PPOAgent(env, config)

    # Create full path for saving
    save_path = os.path.join(config['train'], folder_name)
    os.makedirs(save_path, exist_ok=True)

    # Load pretrained model if specified
    if config['load_model']:
        try:
            print(f"Loading pretrained model from {config['model_load_path']}")
            agent.load(config['model_load_path'])
            # Save pretrained model information
            save_pretrain_info(save_path, config['model_load_path'])
            print("Pretrained model loaded successfully")
        except Exception as e:
            print(f"Error loading pretrained model: {str(e)}")
            print("Training will continue with freshly initialized model")

    callback_class = PPOCustomCallback(config, folder_name)
    agent.train(total_timesteps=config['episodes'] * config['max_steps'], callback=callback_class)
    agent.save(f"{save_path}/")

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

def save_pretrain_info(folder_path, model_path):
    """Save pretrained model information to a file"""
    with open(os.path.join(folder_path, "pretrain_model.txt"), 'w') as f:
        f.write(f"Pretrained model loaded from: {model_path}\n")

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