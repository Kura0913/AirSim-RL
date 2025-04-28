import json
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from __init__ import available_classes

class Platform:
    def load_yaml_config(self, conig_path):
        with open(conig_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)

        return config
    
    def save_training_setting_to_yaml(self, training_setting, save_path):
        with open(os.path.join(save_path, "training_setting.yaml"), 'w') as yaml_file:
            yaml.dump(training_setting, yaml_file, default_flow_style=False)

    def create_instances(self, drone_name, training_setting, folder_name):    
        # Get and instantiate the environment class
        env_config = training_setting['Env']
        env_class_name = env_config.pop('env_class')
        
        if env_class_name in available_classes:
            env_class = available_classes[env_class_name]
            env_instance = env_class(drone_name, env_config)
            print(f"Successfully created an instance of {env_class_name}")
        else:
            print(f"Error: Unable to find class {env_class_name}")
            return None
        
        # Get and instantiate the agent class
        algorithm = training_setting['Algorithm']
        algorithm_settings = self.load_yaml_config("./algorithm_settings.yaml")
        agent_config = algorithm_settings[algorithm]
        agent_class_name = agent_config.pop('agent_class')

        if agent_class_name in available_classes:
            # Instantiate the specified class and pass in the remaining configuration parameters
            agent_class = available_classes[agent_class_name]
            agent_instance = agent_class(env_instance, agent_config, training_setting, folder_name)
            print(f"Successfully created an instance of {agent_class_name}")
        else:
            print(f"Error: Unable to find class {agent_class_name}")
            return None
        
        # Get and instantiate the callback class
        
        return agent_instance
    
    def train(self, drone_name, training_setting, folder_name):
        # Create full path for saving
        save_path = os.path.join(training_setting['save_path'], folder_name)
        os.makedirs(save_path, exist_ok=True)
        # Save training settings to result path
        self.save_training_setting_to_yaml(training_setting, save_path)

        agent = self.create_instances(drone_name, training_setting, folder_name)
        # Create full path for saving
        save_path = os.path.join(training_setting['save_path'], folder_name)
        os.makedirs(save_path, exist_ok=True)

        if training_setting['load_pretrain_model']:
            try:
                print(f"Loading pretrained model from {training_setting['pretrain_model_path']}")
                model_code = input("Please enter the model code (press Enter to use default): ").strip()
                if len(model_code) <= 0:
                    agent.load(training_setting['pretrain_model_path'])
                else:
                    agent.load(training_setting['pretrain_model_path'], model_code)
                # Save pretrained model information
                self.save_load_model_info(save_path, training_setting['pretrain_model_path'])
                print("Pretrained model loaded successfully")
            except Exception as e:
                print(f"Error loading pretrained model: {str(e)}")
                print("Training will continue with freshly initialized model")

        agent.train(total_timesteps=training_setting['episodes'] * training_setting['max_steps'])
        agent.save(f"{training_setting['save_path']}{folder_name}/")

    def test(self, drone_name, testing_setting, folder_name):
        # Create full path for saving
        save_path = os.path.join(testing_setting['save_path'], folder_name)
        os.makedirs(save_path, exist_ok=True)

        # Get the previously trained settings of the specified model
        model_setting = self.load_yaml_config(os.path.join(testing_setting['model_path'], 'training_setting.yaml'))

        try:
            print(f"Loading pretrained model from {testing_setting['model_path']}")
            model_code = input("Please enter the model code (press Enter to use default): ").strip()

            # Create agent
            agent = self.create_instances(drone_name, model_setting, folder_name)

            if len(model_code) <= 0:
                agent.load(testing_setting['model_path'])
            else:
                agent.load(testing_setting['model_path'], model_code)
            # Save pretrained model information
            self.save_load_model_info(save_path, testing_setting['model_path'])
            print("Test model loaded successfully")

            # Run tests and get test results
            test_result = agent.evaluate(testing_setting['test_episodes'])

            # Create plots
            self.create_plots(test_result['episode_rewards'], save_path, test_result['completed_episodes'])

            # save row data as json file
            self.save_row_data(save_path, test_result['episode_rewards'], test_result['completed_episodes'])

            # Save results as txt file
            self.save_results(save_path, testing_setting['model_path'], test_result['avg_reward'], test_result['completion_rate'])

            print(f"\nTest results saved to: {save_path}")
            print(f"Average Reward: {test_result['avg_reward']:.2f}")
            print(f"Completion Rate: {test_result['completion_rate']:.2f}%")

        except Exception as e:
            print(f"Error loading test model: {str(e)}")
            return

    def load_drone_name(self):
        drone_list = self.get_drone_names(os.path.expanduser("~\\Documents\\AirSim\\settings.json"))
        drone_name = drone_list[0]  # Use only the first drone for single drone training

        return drone_name

    def get_drone_names(self, settings_path):
            with open(settings_path, "r") as file:
                data = json.load(file)
            drone_names = list(data.get("Vehicles", {}).keys())  # Get all keys of "Vehicles" as drone names
            print(f"drone list: {drone_names}")
            return drone_names

    def save_load_model_info(self, folder_path, model_path):
        """Save pretrained model information to a file"""
        with open(os.path.join(folder_path, "load_model.txt"), 'w') as f:
            f.write(f"Model loaded from: {model_path}\n")

    def save_results(self, save_path, model_path, avg_reward, completion_rate):
        """Save test results to result.txt"""
        with open(os.path.join(save_path, "result.txt"), 'w') as f:
            f.write(f"Model path: {model_path}\n")
            f.write(f"Average Reward: {avg_reward:.2f}\n")
            f.write(f"Completion Rate: {completion_rate:.2f}%\n")

    def create_plots(self, episode_rewards, save_path, completed_episodes):
        """Create and save test result plots"""
        episodes = np.arange(1, len(episode_rewards) + 1)
        
        # 1. Episode Rewards Plot
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, episode_rewards, color='#1f77b4', linewidth=1.5)
        plt.title('Test Episode Rewards', fontsize=12, pad=10)
        plt.xlabel('Episode', fontsize=10)
        plt.ylabel('Reward', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        max_reward = max(episode_rewards) if episode_rewards else 0
        min_reward = min(episode_rewards) if episode_rewards else 0
        stats_text = f'Max Reward: {max_reward:.2f}\nMin Reward: {min_reward:.2f}'
        plt.text(0.02, 0.98, stats_text, 
                transform=plt.gca().transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.savefig(os.path.join(save_path, "episode_rewards.png"), dpi=300, bbox_inches='tight')
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

            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            plt.savefig(os.path.join(save_path, "moving_average_rewards.png"), dpi=300, bbox_inches='tight')
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

        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        final_rate = completion_rates[-1] if completion_rates else 0
        stats_text = f'Final Completion Rate: {final_rate:.2f}%'
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.savefig(os.path.join(save_path, "completion_rate.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def save_row_data(self, path, episode_rewards, completed_episodes):
        stats_data = {
            'episode_rewards': [float(x) for x in episode_rewards],
            'completed_episodes':[int(x) for x in completed_episodes]
        }
        with open(f"{path}/training_stats.json", 'w') as f:
                    json.dump(stats_data, f, indent=4)