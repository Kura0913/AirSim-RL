import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from Agent import DDPGAgent, PPOAgent  # Import your agent classes
from Env import AirsimEnv

def create_test_folder():
    """Create a test folder with datetime name"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return current_time

def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_results(folder_path, test_path, avg_reward, completion_rate):
    """Save test results to result.txt"""
    with open(os.path.join(folder_path, "result.txt"), 'w') as f:
        f.write(f"Model path: {test_path}\n")
        f.write(f"Average Reward: {avg_reward:.2f}\n")
        f.write(f"Completion Rate: {completion_rate:.2f}%\n")

def create_plots(episode_rewards, folder_path, completed_episodes):
    """Create and save test result plots"""
    episodes = np.arange(1, len(episode_rewards) + 1)
    
    # 1. Episode Rewards Plot
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_rewards, color='#1f77b4', linewidth=1.5)
    plt.title('Test Episode Rewards', fontsize=12, pad=10)
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Reward', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(folder_path, "episode_rewards.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Moving Average Rewards Plot
    plt.figure(figsize=(12, 6))
    window_size = min(20, len(episode_rewards))
    if window_size > 0:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        ma_episodes = np.arange(window_size, len(episode_rewards) + 1)
        plt.plot(ma_episodes, moving_avg, color='#ff7f0e', linewidth=1.5,
                label=f'{window_size}-Episode Moving Average')
        plt.plot(episodes, episode_rewards, color='#1f77b4', alpha=0.3,
                linewidth=1, label='Original Rewards')
        plt.title('Moving Average Reward', fontsize=12, pad=10)
        plt.xlabel('Episode', fontsize=10)
        plt.ylabel('Reward', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(folder_path, "moving_average_rewards.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Completion Rate Plot
    plt.figure(figsize=(12, 6))
    completion_counts = [sum(1 for x in completed_episodes if x <= i) 
                        for i in range(1, len(episode_rewards) + 1)]
    completion_rates = [count / episode * 100 for episode, count 
                       in enumerate(completion_counts, 1)]
    
    plt.plot(episodes, completion_rates, color='#2ca02c', linewidth=1.5)
    plt.title('Task Completion Rate', fontsize=12, pad=10)
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Completion Rate (%)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 110)
    plt.savefig(os.path.join(folder_path, "completion_rate.png"), dpi=300, bbox_inches='tight')
    plt.close()

def test_model(env:AirsimEnv, config, test_episodes):
    """Test the loaded model"""
    # Initialize agent based on algorithm type
    if config["rl_algorithm"].upper() == "DDPG":
        agent = DDPGAgent(env, config)
    elif config["rl_algorithm"].upper() == "PPO":
        agent = PPOAgent(env, config)
    else:
        raise ValueError(f"Unsupported algorithm: {config['rl_algorithm']}")

    model_code = input("Please enter the model code (press Enter to use default): ").strip()
    # Load the trained model
    if len(model_code) <= 0:
        agent.load(config["test_path"])
    else:
        agent.load(config['test_path'], model_code)

    # Run test episodes
    episode_rewards = []
    completed_episodes = []
    
    for episode in range(test_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = agent.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done:
                if info.get('completed', False):
                    completed_episodes.append(episode + 1)
                print(f"Episode {episode + 1}/{test_episodes}: Reward = {episode_reward:.2f}")
        
        episode_rewards.append(episode_reward)

    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    completion_rate = (len(completed_episodes) / test_episodes) * 100

    return episode_rewards, completed_episodes, avg_reward, completion_rate

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
    # Load configuration
    config = load_config()
    drone_name = load_drone_name()
    
    # Create test folder
    folder_name = create_test_folder()
    test_folder = os.path.join(config['test'], folder_name)
    os.makedirs(test_folder, exist_ok=True)

    # Initialize your environment here
    env = AirsimEnv(drone_name, config)  # You need to implement this
    
    # Set number of test episodes
    test_episodes = 100  # You can modify this or make it configurable
    
    # Run tests
    episode_rewards, completed_episodes, avg_reward, completion_rate = test_model(
        env, config, test_episodes)
    
    # Create plots
    create_plots(episode_rewards, test_folder, completed_episodes)
    
    # Save results
    save_results(test_folder, config['test_path'], avg_reward, completion_rate)
    
    print(f"\nTest results saved to: {test_folder}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Completion Rate: {completion_rate:.2f}%")

if __name__ == "__main__":
    main()