from stable_baselines3.common.callbacks import BaseCallback
import os

class CustomCallback(BaseCallback):
    def __init__(self, config, folder_name, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.config = config
        self.folder_name = folder_name
        self.desired_episodes = self.config['episodes']
        self.save_episodes = self.config['save_episodes']
        self.curr_episode = 0
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Update current episode reward
        self.current_episode_reward += self.locals['rewards'][0]
        # Check if the episode is over
        done = self.locals['dones'][0]
        if done:            
            self.curr_episode += 1
            self.episode_rewards.append(self.current_episode_reward)

            # Output the rewards of the current episode on the console
            print(f"Episode: {self.curr_episode:4d}/{self.desired_episodes} finished with reward: {self.current_episode_reward}")

            # Reset current episode reward
            self.current_episode_reward = 0

            # Check if the model needs to be saved
            if self.curr_episode in self.save_episodes:
                path = f"{self.config['train']}{self.folder_name}/"
                os.makedirs(path, exist_ok=True)
                model_path = f"{path}{self.config['algorithm'].lower()}_model_episode_{self.curr_episode}.pth"
                self.model.save(model_path)
                print(f"Model saved at episode {self.curr_episode}")

            # Check whether the specified number of episodes is reached
            if self.curr_episode >= self.desired_episodes:
                return False

        return True

    def get_episode_rewards(self):
        return self.episode_rewards