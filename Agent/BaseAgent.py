from stable_baselines3.common.base_class import BaseAlgorithm
import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, env, agent_config, training_setting, folder_name):
        self.agent_config = agent_config
        self.env = env
        self.device = training_setting['device']
        self.model = BaseAlgorithm
        self.class_component = self._get_agnet_settings(training_setting, folder_name)

    def _get_agnet_settings(self, training_setting, folder_name):
        # get feature extractor class
        from __init__ import available_classes
        feature_extractor_class_name = self.agent_config.pop('features_extractor_class')
        if feature_extractor_class_name in available_classes:
            feature_extractor_class = available_classes[feature_extractor_class_name]
            print(f'Successfully created an instance of {feature_extractor_class}.')
        else:
            print(f'Error: Unable to find class {feature_extractor_class_name}.')
            return None
        
        # get callback class
        callback_class_name = self.agent_config.pop('callback_class')
        if callback_class_name in available_classes:
            callback_class = available_classes[callback_class_name]
            callback = callback_class(training_setting, folder_name)
            print(f'Successsfully created an instance of {callback_class}.')
        else:
            print(f'Error: Unable to find class {callback_class_name}.')

        return {'feature_extractor_class': feature_extractor_class, 'callback': callback}
    
    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps, callback=self.class_component['callback'])

    @abstractmethod
    def save(self, path):
        """Save model components separately instead of saving the whole model"""
        pass
    
    @abstractmethod
    def load(self, path, model_code:str=''):
        """Load model components separately"""
        pass

    def evaluate(self, episodes=5, deterministic=True, render=False):
        episode_rewards = []
        episode_lengths = []
        completed_episodes = []

        for episode in range(episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done:
                    if info.get('completed', False):
                        completed_episodes.append(episode + 1)
                    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, " 
                          f"Length = {episode_length}, Success = {info['completed']}")
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # Output statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        completion_rate = (len(completed_episodes) / episodes) * 100

        print("\nEvaluation Results:")
        print(f"Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
        print(f"Mean Episode Length: {mean_length:.1f}")
        print(f"Completion Rate: {completion_rate:.2f}%")

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'completed_episodes': completed_episodes,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'completion_rate': completion_rate
        }