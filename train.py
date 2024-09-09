import json
import torch
import numpy as np
import os
from DDPG.Agent import DDPGAgent
from DDPG.AirsimEnv import AirSimEnv
import matplotlib.pyplot as plt
import cv2
import datetime

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

def sample_point_cloud(point_cloud, num_points=1024):
    num_points_in_cloud = point_cloud.shape[0]
    
    if num_points_in_cloud >= num_points:
        choice = np.random.choice(num_points_in_cloud, num_points, replace=False)
    else:
        choice = np.random.choice(num_points_in_cloud, num_points, replace=True)
    
    return point_cloud[choice, :]

def preprocess_depth_image(depth_image, resize, max_depth=255.0, min_depth_threshold=1.0, ignore_value=np.nan):
    """
    Preprocess the depth image, setting pixels below a certain threshold to a specified ignore value.

    Args:
    - depth_image (np.ndarray): Raw depth image.
    - max_depth (float): Normalized maximum depth value.
    - min_depth_threshold (float): Minimum threshold to ignore depth values.
    - ignore_value (float): The flag value of the ignored pixel.

    Returns:
    - torch.Tensor: Processed depth images, suitable for use in neural networks.
    """
    resize = (resize[0], resize[1])
    # resize the image
    depth_image_resized = cv2.resize(depth_image, resize)
    
    # normalized depth value
    depth_image_normalized = depth_image_resized / max_depth
    
    # set the value to nan if the value over max_depth
    depth_image_normalized[depth_image_normalized > (min_depth_threshold / max_depth)] = ignore_value    
    
    # Convert to PyTorch tensor format
    depth_image_tensor = torch.from_numpy(depth_image_normalized).unsqueeze(0).float()  # (1, 84, 84)

    
    return depth_image_tensor

def train(envs, agent, config):
    episode = 0
    episodes = config[config["mode"]]["num_episodes"]
    episode_rewards = []
    episode_losses = []

    while episode < episodes:
        # Initialize the status of each drone
        states = [env.reset() for env in envs]
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

                if config["mode"] == "lidar_mode":
                    lidar_data = env.get_lidar_data()
                    processed_data = sample_point_cloud(lidar_data, num_points=config["lidar_mode"]["target_numbers"])

                elif config["mode"] == "camera_mode":
                    depth_image = env.get_depth_image()
                    processed_data = preprocess_depth_image(depth_image, target_size=config["camera_mode"]["resize"])

                elif config["mode"] == "all_sensors":
                    depth_image = env.get_depth_image()
                    lidar_data = env.get_lidar_data()
                    processed_depth_image = preprocess_depth_image(depth_image, target_size=config["camera_mode"]["resize"])
                    sampled_point_cloud = sample_point_cloud(lidar_data, num_points=config["lidar_mode"]["target_numbers"])
                    processed_data = np.concatenate([processed_depth_image.flatten(), sampled_point_cloud.flatten()])

                reward, done = env.computed_reward(actions[i])  # Get rewards and end signals
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

        avg_rewards = [np.mean(rewards) for rewards in total_rewards if rewards]
        avg_episode_reward = np.mean(avg_rewards) if avg_rewards else 0

        # Calculate average losses for the episode
        avg_critic_loss = np.mean(total_critic_loss) if total_critic_loss else 0
        avg_actor_loss = np.mean(total_actor_loss) if total_actor_loss else 0

        episode_rewards.append(avg_episode_reward)
        episode_losses.append((avg_critic_loss + avg_actor_loss) / 2)  # Average of both losses
        if config[config["mode"]]["train_infinite"]:
            print(f"Episode {episode+1}/N completed. Avg Reward: {avg_episode_reward:.2f}, Avg Critic Loss: {avg_critic_loss:.4f}, Avg Actor Loss: {avg_actor_loss:.4f}")
        else:
            print(f"Episode {episode+1}/{episodes} completed. Avg Reward: {avg_episode_reward:.2f}, Avg Critic Loss: {avg_critic_loss:.4f}, Avg Actor Loss: {avg_actor_loss:.4f}")

    # Plot rewards and losses
    plot_rewards_and_losses(list(range(1, episode + 1)), episode_rewards, episode_losses, save_path="training_results.png")

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
    mode_config = config[mode]

    # Set save path
    save_dir = mode_config["save_path"]["train"]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, timestamp)
    os.makedirs(save_path, exist_ok=True)

    # Set the state dimension (state_dim) and other parameters according to the configuration
    if mode == "lidar_mode":
        target_numbers = mode_config["target_numbers"]
        state_dim = target_numbers * 3
    elif mode == "camera_mode":
        resize = tuple(mode_config["resize"])
        state_dim = resize[0] * resize[1]
    elif mode == "all_sensors":
        target_numbers = mode_config["target_numbers"]
        resize = tuple(mode_config["resize"])
        state_dim = (target_numbers * 3) + (resize[0] * resize[1])
    else:
        raise ValueError("Unknow mode or config setting error! Please choose one from the following modes: lidar_mode, camera_mode, all_sensors.")

    action_dim = 3

    # Set up multiple drone environments
    envs = [AirSimEnv(drone_name) for drone_name in drone_names]

    agent = DDPGAgent(state_dim, action_dim, device)

    # Load pre-trained model if exists
    if mode_config["load_model"]:
        model_load_path = mode_config.get("model_load_path", None)  # Optional read path
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
    main()
