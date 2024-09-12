import json
from Env import AirSimMultiDroneEnv
from Model.RLModel import RLModel
from stable_baselines3.common.vec_env import SubprocVecEnv
import os

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
    env = SubprocVecEnv([make_env(i, config, num_drones) for i in range(num_drones)])  # Create a multi-process environment

    # initial RL model
    rl_model = RLModel(config, env)

    # start training
    print(f"Training with RL algorithm: {config['rl_algorithm']}")
    rl_model.train()
    print("Training completed.")

if __name__ == "__main__":
    run_training()