import json
from Env import AirSimEnv
from Model.RLModel import RLModel

def get_test_file_path():
    with open('config.json', 'r') as file:
        config = json.load(file)
    testing_file_path = config["test_path"]

    return testing_file_path

def run_testing():
    file_path = get_test_file_path()

    with open(file_path + 'config.json', 'r') as file:
        config = json.load(file)    
    
    env = AirSimEnv(config)
    
    rl_model = RLModel(config, env)
    model_path = file_path + 'model.pth'
    rl_model.load_model(model_path)
    # start testing
    print(f"Testing with RL algorithm: {config['rl_algorithm']}")
    rl_model.test()
    print("Testing completed.")

if __name__ == "__main__":
    run_testing()
