import json
import torch
import numpy as np
import os
from DDPG.Agent import DDPGAgent
from DDPG.AirsimEnv import AirSimEnv
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import sys
import threading
import keyboard

def plot_rewards_and_losses(episodes, rewards, average_losses, save_path):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot rewards as bars
    ax1.bar(episodes, rewards, color='blue', alpha=0.6, label='Rewards')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Rewards', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis for average losses
    ax2 = ax1.twinx()
    ax2.plot(episodes, average_losses, color='red', label='Average Loss')
    ax2.set_ylabel('Average Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add legends and grid
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
    ax1.grid(True)

    # Save and show the plot
    plt.savefig(save_path)
    plt.show()

def get_drone_names_from_settings(settings_path):
    with open(settings_path, "r") as file:
        data = json.load(file)
    drone_names = list(data.get("Vehicles", {}).keys())  # Get all keys of "Vehicles" as drone names
    print(f"drone list: {drone_names}")
    return drone_names

def preprocess_depth_image(depth_image, resize, max_depth=255.0, min_depth_threshold=1.0, ignore_value=np.nan):
    """
    Preprocess the depth image, setting pixels below a certain threshold to a specified ignore value.

    Args:
    - depth_image (np.ndarray): Raw depth image. If None, a zero matrix is created.
    - resize (tuple): Target size for resizing the image.
    - max_depth (float): Normalized maximum depth value.
    - min_depth_threshold (float): Minimum threshold to ignore depth values.
    - ignore_value (float): The flag value of the ignored pixel.

    Returns:
    - torch.Tensor: Processed depth images, suitable for use in neural networks.
    """
    resize = (resize[0], resize[1])

    # If depth_image is None, create a zero matrix
    if depth_image is None:
        depth_image_resized = np.zeros(resize, dtype=np.float32)
    else:
        # Resize the image
        depth_image_resized = cv2.resize(depth_image, resize)
    
    # Normalized depth value
    depth_image_normalized = depth_image_resized / max_depth
    
    # Set the value to nan if the value is over max_depth
    depth_image_normalized[depth_image_normalized > (min_depth_threshold / max_depth)] = ignore_value    
    
    # Convert to PyTorch tensor format
    depth_image_tensor = torch.from_numpy(depth_image_normalized).unsqueeze(0).float()  # (1, 84, 84)
    
    return depth_image_tensor

def sample_point_cloud(point_cloud, num_points=1024):
    """
    Sample points from a point cloud or return a zero matrix if input is None.

    Args:
    - point_cloud (np.ndarray): Raw point cloud data. If None, a zero matrix is created.
    - num_points (int): Number of points to sample.

    Returns:
    - np.ndarray: Sampled point cloud data.
    """
    # If point_cloud is None, create a zero matrix
    if point_cloud is None:
        return np.zeros((num_points, 3), dtype=np.float32)
    
    num_points_in_cloud = point_cloud.shape[0]
    
    if num_points_in_cloud >= num_points:
        choice = np.random.choice(num_points_in_cloud, num_points, replace=False)
    else:
        choice = np.random.choice(num_points_in_cloud, num_points, replace=True)
    
    return point_cloud[choice, :]

def get_process_data(env, config):
    """
    Process data from sensors based on the given mode.

    Args:
    - env (AirSimEnv): The simulation environment.
    - config (dict): Configuration for data processing.

    Returns:
    - np.ndarray: Processed sensor data.
    """
    if config["mode"] == "lidar_mode":
        lidar_data = env.get_lidar_data()
        processed_data = sample_point_cloud(lidar_data, num_points=config["point_numbers"])

        # Zero padding for depth image data
        depth_image_size = config["resize"][0] * config["resize"][1]
        depth_image_zeros = np.zeros(depth_image_size)
        processed_data = np.concatenate([processed_data.flatten(), depth_image_zeros])
        
    elif config["mode"] == "camera_mode":
        depth_image = env.get_depth_image()
        processed_data = preprocess_depth_image(depth_image, resize=config["resize"]).numpy().flatten()

        # Zero padding for point cloud data
        point_cloud_zeros = np.zeros(config["point_numbers"] * 3)
        processed_data = np.concatenate([point_cloud_zeros, processed_data])

    elif config["mode"] == "all_sensors":
        depth_image = env.get_depth_image()
        lidar_data = env.get_lidar_data()

        processed_depth_image = preprocess_depth_image(depth_image, resize=config["resize"]).numpy().flatten()
        sampled_point_cloud = sample_point_cloud(lidar_data, num_points=config["point_numbers"]).flatten()
        processed_data = np.concatenate([sampled_point_cloud, processed_depth_image])

    return processed_data

def train(envs, agent, config):
    global stop_training
    # Start a separate thread to listen for stop key
    threading.Thread(target=listen_for_stop_key, daemon=True).start()
    episode = 0
    episodes = config["num_episodes"]
    episode_rewards = []
    episode_losses = []

    while episode < episodes and not stop_training:
        states = []
        # Initialize the status of each drone
        for env in envs:
            env.reset()
            states.append(get_process_data(env, config))

        total_rewards = [[] for _ in range(len(envs))]  # Use multiple lists to track rewards for each drone
        total_critic_loss = []
        total_actor_loss = []
        dones = [False] * len(envs)  # initial the done status of each drone

        while not all(dones):  # When the status of all drones is True, the current episode ends
            actions = [agent.act(state) for state in states]

            next_states, rewards = [], []
            for i, env in enumerate(envs):
                if dones[i]:  # If the drone has completed its mission, skip
                    continue

                env.step(actions[i])

                processed_data = get_process_data(env, config)

                reward, done, completed = env.computed_reward(actions[i])  # Get rewards and end signals
                next_states.append(processed_data)
                rewards.append(reward)
                agent.store(states[i], actions[i], reward, processed_data)

                # Set the done state of the drone
                dones[i] = done
                
                # Total reward for recording drones
                total_rewards[i].append(reward)

            # update state
            states = next_states
 
            # Update the agent and collect losses
            critic_loss, actor_loss = agent.update()

            # Only add valid losses (skip when not enough samples in replay buffer)
            if critic_loss is not None and actor_loss is not None:
                total_critic_loss.append(critic_loss)
                total_actor_loss.append(actor_loss)

            # calcuate average reward for current state
            avg_rewards = [np.mean(rewards) for rewards in total_rewards if rewards]
            avg_episode_reward = np.mean(avg_rewards) if avg_rewards else 0

            # Calculate average losses for current state
            avg_critic_loss = np.mean(total_critic_loss) if total_critic_loss else 0
            avg_actor_loss = np.mean(total_actor_loss) if total_actor_loss else 0

            if config[config['mode']]["train_infinite"]:
                if done:
                    if completed:
                        status = (f'Episode: {episode + 1:5d}/N | Reward: {avg_episode_reward:5d} | loss: {(avg_critic_loss + avg_actor_loss) / 2:.4f} | mission_state: success')
                    else:
                        status = (f'Episode: {episode + 1:5d}/N | Reward: {avg_episode_reward:5d} | loss: {(avg_critic_loss + avg_actor_loss) / 2:.4f} | mission_state: failed')
                else:
                    status = (f'Episode: {episode + 1:5d}/N | Reward: {avg_episode_reward:5d} | loss: {(avg_critic_loss + avg_actor_loss) / 2:.4f} | mission_state: running')
            else:
                if done:
                    if completed:
                        status = (f'Episode: {episode + 1:5d}/{episodes} | Reward: {avg_episode_reward:5d} | loss: {(avg_critic_loss + avg_actor_loss) / 2:.4f} | mission_state: success')
                    else:
                        status = (f'Episode: {episode + 1:5d}/{episodes} | Reward: {avg_episode_reward:5d} | loss: {(avg_critic_loss + avg_actor_loss) / 2:.4f} | mission_state: failed')
                else:
                    status = (f'Episode: {episode + 1:5d}/{episodes} | Reward: {avg_episode_reward:5d} | loss: {(avg_critic_loss + avg_actor_loss) / 2:.4f} | mission_state: running')
            
            sys.stdout.write('\r' + status)
            sys.stdout.flush()
            
        print(f'\r')
        # avg_rewards = [np.mean(rewards) for rewards in total_rewards if rewards]
        # avg_episode_reward = np.mean(avg_rewards) if avg_rewards else 0

        # # Calculate average losses for the episode
        # avg_critic_loss = np.mean(total_critic_loss) if total_critic_loss else 0
        # avg_actor_loss = np.mean(total_actor_loss) if total_actor_loss else 0

        episode_rewards.append(avg_episode_reward)
        episode_losses.append((avg_critic_loss + avg_actor_loss) / 2)  # Average of both losses
        episode += 1

    # Plot rewards and losses
    plot_rewards_and_losses(list(range(1, episode + 1)), episode_rewards, episode_losses, save_path="training_results.png")

def listen_for_stop_key():
    """listing 'p' key"""
    global stop_training
    keyboard.wait('p')
    stop_training = True

def main():
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get drone names
    airsim_settings_path = os.path.expanduser("~\\Documents\\AirSim\\settings.json")
    drone_names = get_drone_names_from_settings(airsim_settings_path)

    # Read JSON configuration file
    with open("config.json", "r") as file:
        config = json.load(file)

    # Select the corresponding training configuration according to the mode
    mode = config["mode"]

    # Set save path
    save_dir = config["train"][mode]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, timestamp)
    os.makedirs(save_path, exist_ok=True) 

    # Set the state dimension (state_dim) and other parameters according to the configuration
    if mode == "lidar_mode":
        point_numbers = config["point_numbers"]
        state_dim = point_numbers * 3
    elif mode == "camera_mode":
        resize = tuple(config["resize"])
        state_dim = resize[0] * resize[1]
    elif mode == "all_sensors":
        point_numbers = config["point_numbers"]
        resize = tuple(config["resize"])
        state_dim = (point_numbers * 3) + (resize[0] * resize[1])
    else:
        raise ValueError("Unknow mode or config setting error! Please choose one from the following modes: lidar_mode, camera_mode, all_sensors.")

    action_dim = 3
    max_action = config['max_action']
    # Set up multiple drone environments
    envs = [AirSimEnv(drone_name) for drone_name in drone_names]
    
    agent = DDPGAgent(state_dim, action_dim, device, max_action, config["point_numbers"]*3, config["resize"][0] * config["resize"][1])

    # Load pre-trained model if exists
    if config["load_model"]:
        model_load_path = config.get("model_load_path", None)  # Optional read path
        if model_load_path and os.path.exists(model_load_path):
            agent.load(model_load_path)

    # Start training
    train(envs, agent, config)

    # Save the trained model
    model_save_path = os.path.join(save_path, "trained_model.pth")
    agent.save(model_save_path)

    # Close all environments when finished
    for env in envs:
        env.close()

if __name__ == "__main__":
    stop_training = False
    main()
