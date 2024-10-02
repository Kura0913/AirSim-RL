from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import matplotlib.pyplot as plt

class CustomCallback(BaseCallback):
    def __init__(self, rl_algorithm, save_path, save_freq=1000, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.rl_algorithm = rl_algorithm
        self.save_path = save_path
        self.save_freq = save_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.best_mean_reward = -np.inf
        self.n_steps = 0

    def _on_step(self) -> bool:
        self.n_steps += 1
        
        # Log episode info
        if 'episode' in self.locals:
            episode_info = self.locals['episode']
            self.episode_rewards.append(episode_info['r'])
            self.episode_lengths.append(episode_info['l'])
        
        # Log loss
        self.get_loss_value()
        
        # Save model periodically
        if self.n_steps % self.save_freq == 0:
            self.model.save(os.path.join(self.save_path, f"model_{self.n_steps}"))
        
        # Save best model
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.save_path, "best_model"))
        
        return True

    def _on_training_end(self) -> None:
        # Save final model
        self.model.save(os.path.join(self.save_path, "final_model"))
        
        # Save episode data
        np.save(os.path.join(self.save_path, "episode_rewards.npy"), self.episode_rewards)
        np.save(os.path.join(self.save_path, "episode_lengths.npy"), self.episode_lengths)
        np.save(os.path.join(self.save_path, "episode_losses.npy"), self.episode_losses)
        
        # Plot training curves
        self.plot_training_curves()

    def get_loss_value(self):
        if self.rl_algorithm == 'DDPG':
            if hasattr(self.model, 'critic_loss'):
                critic_loss_value = self.model.critic_loss
                self.episode_losses.append(critic_loss_value)
        elif self.rl_algorithm == 'PPO':
            if hasattr(self.model, 'value_loss') and hasattr(self.model, 'policy_loss'):
                value_loss = self.model.value_loss
                policy_loss = self.model.policy_loss
                total_loss = value_loss + policy_loss
                self.episode_losses.append(total_loss)

    def plot_training_curves(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(132)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        
        plt.subplot(133)
        plt.plot(self.episode_losses)
        plt.title('Episode Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'training_curves.png'))
        plt.close()