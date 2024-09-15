from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CustomCallback(BaseCallback):
    def __init__(self, rl_algorithm, save_path, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.rl_algorithm = rl_algorithm
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_losses = []

    def _on_step(self) -> bool:
        # Assume a monitor environment is used to automatically log returns
        if 'episode' in self.locals:
            episode_reward = self.locals['episode']['r']
            self.episode_rewards.append(episode_reward)
        
        self.get_loss_value()

        return True
        

    def _on_training_end(self) -> None:
        np.save(self.save_path + "episode_rewards.npy", self.episode_rewards)
        np.save(self.save_path + "episode_losses.npy", self.episode_losses)
    
    # get loss according rl-algorithm
    def get_loss_value(self):
        if self.rl_algorithm == 'DDPG':
            if hasattr(self.model, 'critic_loss'):
                critic_loss_value = self.model.critic_loss
                self.episode_losses.append(critic_loss_value)
        if self.rl_algorithm == 'PPO':
            if hasattr(self.model, 'value_loss') and hasattr(self.model, 'policy_loss'):
                value_loss = self.model.value_loss
                policy_loss = self.model.policy_loss
                total_loss = value_loss + policy_loss
                self.episode_losses.append(total_loss)
