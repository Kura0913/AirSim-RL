from stable_baselines3.common.callbacks import BaseCallback
import os
import torch as th
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json

class BaseCustomCallback(BaseCallback):
    def __init__(self, config, folder_name, verbose=0):
        super(BaseCustomCallback, self).__init__(verbose)
        self.config = config
        self.folder_name = folder_name
        self.desired_episodes = self.config['episodes']
        self.save_episodes = self.config['save_episodes']
        self.curr_episode = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.completed_episodes = []

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        done = self.locals['dones'][0]
        if done:            
            self.curr_episode += 1
            completed = self.locals['infos'][0].get('completed', False)
            completion_status = "completed" if completed else "failed"
            self.episode_rewards.append(self.current_episode_reward)

            if completed:
                self.completed_episodes.append(self.curr_episode)

            print(f"Episode: {self.curr_episode:4d}/{self.desired_episodes} end with reward: {self.current_episode_reward:.2f}, mission status: {completion_status}")

            self.current_episode_reward = 0

            if self.curr_episode in self.save_episodes:
                self._save_model()
                self._save_training_plots(str(self.curr_episode))

            if self.curr_episode >= self.desired_episodes:
                self._save_training_plots()
                return False

        return True

    def _save_model(self):
        raise NotImplementedError

    def _save_base_stats(self, base_path):
        training_stats = {
            'episode': self.curr_episode,
            'episode_rewards': self.episode_rewards,
            'completed_episodes': self.completed_episodes,
            'current_reward': self.current_episode_reward,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'average_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
            'max_reward': float(np.max(self.episode_rewards)) if self.episode_rewards else 0,
            'min_reward': float(np.min(self.episode_rewards)) if self.episode_rewards else 0,
            'completion_rate': float(len(self.completed_episodes)/max(1, len(self.episode_rewards))*100)
        }
        
        stats_path = f"{base_path}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(training_stats, f, indent=4)
        print(f"Training statistics saved to {stats_path}")

    def _save_training_plots(self, episode_str=""):
        """
        Generate and save training statistics plots
        1. Episode rewards plot
        2. Moving average rewards plot
        3. Completion rate plot
        4. Moving Completion Rate Plot
        """
        if episode_str != "":
            episode_str = "_" + episode_str
        try:
            path = f"{self.config['save_path']}{self.folder_name}/"
            os.makedirs(path, exist_ok=True)
            
            rewards_path = f"{path}episode_rewards{episode_str}.png"
            avg_path = f"{path}moving_average_rewards{episode_str}.png"
            completion_path = f"{path}completion_rate{episode_str}.png"
            
            # Initialize data variables
            moving_avg = None
            completion_rates = []
            final_rate = 0.0
            
            plt.style.use('default')
            
            # 1. Episode Rewards Plot
            plt.figure(figsize=(12, 6))
            episodes = np.arange(1, len(self.episode_rewards) + 1, dtype=int)
            plt.plot(episodes, self.episode_rewards, color='#1f77b4', linewidth=1.5)
            plt.title('Training Episode Rewards', fontsize=12, pad=10)
            plt.xlabel('Episode', fontsize=10)
            plt.ylabel('Reward', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            max_reward = max(self.episode_rewards) if self.episode_rewards else 0
            min_reward = min(self.episode_rewards) if self.episode_rewards else 0
            stats_text = f'Max Reward: {max_reward:.2f}\nMin Reward: {min_reward:.2f}'
            plt.text(0.02, 0.98, stats_text, 
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.savefig(rewards_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Moving Average Rewards Plot
            plt.figure(figsize=(12, 6))
            
            # Dynamically resize window
            window_size = min(100, len(self.episode_rewards))
            if window_size > 0:
                moving_avg = np.convolve(self.episode_rewards, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
                # The x-axis of the moving average should start from window_size
                ma_episodes = np.arange(window_size, len(self.episode_rewards) + 1, dtype=int)
                plt.plot(ma_episodes, moving_avg, 
                        color='#ff7f0e',  # orange
                        linewidth=1.5,
                        label=f'{window_size}-Episode Moving Average')
                
                # For reference, the original data is also plotted
                plt.plot(episodes, self.episode_rewards, 
                        color='#1f77b4', 
                        alpha=0.3, 
                        linewidth=1,
                        label='Original Rewards')
                
                plt.title(f'Moving Average Reward', fontsize=12, pad=10)
                plt.xlabel('Episode', fontsize=10)
                plt.ylabel('Reward', fontsize=10)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                
            plt.savefig(avg_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Completion Rate Plot
            plt.figure(figsize=(12, 6))
            
            # Completion rate is always calculated even if there are no completed cases
            completion_counts = [sum(1 for x in self.completed_episodes if x <= i) 
                            for i in range(1, len(self.episode_rewards) + 1)]
            completion_rates = [count / episode * 100 for episode, count 
                            in enumerate(completion_counts, 1)]
            
            plt.plot(episodes, completion_rates, 
                    color='#2ca02c',  # green
                    linewidth=1.5)
            plt.title('Task Completion Rate', fontsize=12, pad=10)
            plt.xlabel('Episode', fontsize=10)
            plt.ylabel('Completion Rate (%)', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            # range: 0-100%
            plt.ylim(0, 100)
            
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            # Add final completion rate information
            final_rate = completion_rates[-1] if completion_rates else 0
            stats_text = f'Final Completion Rate: {final_rate:.2f}%'
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.savefig(completion_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 4. Moving Completion Rate Plot
            plt.figure(figsize=(12, 6))

            # Calculate moving completion rate with window size 100
            window_size = min(100, len(completion_rates))
            if window_size > 0:
                moving_completion = np.convolve(completion_rates, 
                                            np.ones(window_size)/window_size, 
                                            mode='valid')
                # The x-axis should start from window_size
                mc_episodes = np.arange(window_size, len(completion_rates) + 1, dtype=int)
                
                # Plot original completion rates with transparency
                plt.plot(episodes, completion_rates, 
                        color='#2ca02c',  # green
                        alpha=0.3, 
                        linewidth=1,
                        label='Original Completion Rate')
                
                # Plot moving completion rate
                plt.plot(mc_episodes, moving_completion, 
                        color='#d62728',  # red
                        linewidth=1.5,
                        label=f'{window_size}-Episode Moving Completion Rate')
                
                plt.title('Moving Task Completion Rate', fontsize=12, pad=10)
                plt.xlabel('Episode', fontsize=10)
                plt.ylabel('Completion Rate (%)', fontsize=10)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.ylim(0, 100)  # range: 0-100%
                plt.legend()
                
                plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                
                # Add final moving completion rate information
                final_moving_rate = moving_completion[-1] if len(moving_completion) > 0 else 0
                stats_text = f'Final Moving Completion Rate: {final_moving_rate:.2f}%'
                plt.text(0.02, 0.98, stats_text,
                        transform=plt.gca().transAxes,
                        verticalalignment='top',
                        fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                moving_completion_path = f"{path}moving_completion_rate{episode_str}.png"
                plt.savefig(moving_completion_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"4. Moving Completion Rate Plot: {moving_completion_path}")
            
            # Save raw data as JSON
            stats_data = {
                'episode_rewards': [float(x) for x in self.episode_rewards],
                'moving_average': [float(x) for x in moving_avg] if moving_avg is not None else [],
                'completed_episodes':[int(x) for x in self.completed_episodes],
                'moving_completion_rate': [float(x) for x in moving_completion],
                'completion_rates': [float(x) for x in completion_rates],
                'max_reward': float(max_reward),
                'min_reward': float(min_reward),
                'final_completion_rate': float(final_rate)
            }
            
            with open(f"{path}training_stats.json", 'w') as f:
                json.dump(stats_data, f, indent=4)
            
            print("\nTraining plots have been saved:")
            print(f"1. Episode Rewards Plot: {rewards_path}")
            print(f"2. Moving Average Plot: {avg_path}")
            print(f"3. Completion Rate Plot: {completion_path}")
            print(f"4. Raw Statistics Data: {path}training_stats.json")
            
        except Exception as e:
            print(f"Error saving training plots: {str(e)}")
            import traceback
            traceback.print_exc()
