from PolicyComponent.DDPGFeaturesExtractor import DDPGFeaturesExtractor
from PolicyComponent.PPOFeaturesExtractor import PPOFeaturesExturactor
from Policy.CustomTD3Policy import CustomTD3Policy
from Policy.CustomPPOPolicy import CustomPPOPolicy
from stable_baselines3 import DDPG, PPO
import torch as th
from ReplayBuffer import PrioritizedReplayBuffer
import numpy as np
import json

class DDPGAgent:
    def __init__(self, env, config):
        self.config = config
        policy_kwargs = dict(
            features_extractor_class=DDPGFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=32),
            net_arch=dict(pi=[256, 256, 256], qf=[256, 256]),
            config=config,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs=dict(
                eps=1e-5,
                weight_decay=1e-4,
                amsgrad=True
            )
        )
        self.model = DDPG(
            CustomTD3Policy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            gamma=0.9999,
            learning_starts=1000,
            buffer_size=self.config['buffer_size'],
            learning_rate=self.config["learning_rate"],
            batch_size=64,
            device=th.device(self.config['device'])
        )

        self.model.replay_buffer = PrioritizedReplayBuffer(
            buffer_size=50000,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=th.device(self.config['device']),
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
        )

    def train(self, total_timesteps, callback):
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path, model_code:str=''):
        """Save model components separately instead of saving the whole model"""
        try:
            # Save policy network state dict
            policy_state = self.model.policy.state_dict()
            th.save(policy_state, f"{path}{model_code}_policy.pth")

            # Save actor network state dict
            actor_state = self.model.actor.state_dict()
            th.save(actor_state, f"{path}{model_code}_actor.pth")

            # Save critic network state dict
            critic_state = self.model.critic.state_dict()
            th.save(critic_state, f"{path}{model_code}_critic.pth")

            # Save training parameters
            training_params = {
                'learning_rate': self.model.learning_rate,
                'gamma': self.model.gamma,
                'batch_size': self.model.batch_size,
                'buffer_size': self.model.buffer_size,
                'learning_starts': self.model.learning_starts,
                # Add any other relevant parameters
            }
            
            with open(f"{path}_params.json", 'w') as f:
                json.dump(training_params, f, indent=4)
                
            print(f"Model components saved successfully to {path}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            import traceback
            traceback.print_exc()

    def load(self, path):
        """Load model components separately"""
        try:
            # Load policy network
            policy_state = th.load(f"{path}_policy.pth")
            self.model.policy.load_state_dict(policy_state)

            # Load actor network
            actor_state = th.load(f"{path}_actor.pth")
            self.model.actor.load_state_dict(actor_state)

            # Load critic network
            critic_state = th.load(f"{path}_critic.pth")
            self.model.critic.load_state_dict(critic_state)

            # Load parameters
            with open(f"{path}_params.json", 'r') as f:
                params = json.load(f)
                # Update model parameters if needed
                self.model.learning_rate = params['learning_rate']
                self.model.gamma = params['gamma']
                # ... update other parameters as needed

            print(f"Model components loaded successfully from {path}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()

    def evaluate(self, episodes=5, deterministic=True, render=False):
        episode_rewards = []
        episode_lengths = []
        episode_successes = []

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
                
                if render:
                    self.env.render()
                
                if done:
                    success = info.get('completed', False)
                    episode_successes.append(success)
                    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, " 
                          f"Length = {episode_length}, Success = {success}")
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # Output statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        success_rate = np.mean(episode_successes) * 100

        print("\nEvaluation Results:")
        print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Mean Episode Length: {mean_length:.1f}")
        print(f"Success Rate: {success_rate:.1f}%")

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_successes': episode_successes,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'success_rate': success_rate
        }

class PPOAgent:
    def __init__(self, env, config):
        self.config = config
        self.env = env
        policy_kwargs = dict(
            features_extractor_class=PPOFeaturesExturactor,
            features_extractor_kwargs=dict(features_dim=32),
            net_arch=[],
            config=config,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs=dict(
                eps=1e-5,
                weight_decay=1e-4,
                amsgrad=True
            )
        )
        self.model = PPO(
            CustomPPOPolicy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            gamma=0.9999,
            n_steps=512,
            batch_size=64,
            n_epochs=5,
            learning_rate=self.config["learning_rate"],
            device=th.device(self.config['device'])
        )

    def train(self, total_timesteps, callback):
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path):
        """Save model components separately instead of saving the whole model"""
        try:
            # Save policy network state dict
            policy_state = self.model.policy.state_dict()
            th.save(policy_state, f"{path}_policy.pth")

            # Save value function network state dict
            value_net_state = self.model.policy.value_net.state_dict()
            th.save(value_net_state, f"{path}_value_net.pth")

            # Save training parameters
            training_params = {
                'learning_rate': self.model.learning_rate,
                'gamma': self.model.gamma,
                'batch_size': self.model.batch_size,
                'n_steps': self.model.n_steps,
                'n_epochs': self.model.n_epochs,
                # Add any other relevant parameters
            }
            
            with open(f"{path}_params.json", 'w') as f:
                json.dump(training_params, f, indent=4)
                
            print(f"Model components saved successfully to {path}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            import traceback
            traceback.print_exc()

    def load(self, path, model_code:str=''):
        """Load model components separately"""
        try:
            # Load policy network
            policy_state = th.load(f"{path}{model_code}_policy.pth")
            self.model.policy.load_state_dict(policy_state)

            # Load value function network
            value_net_state = th.load(f"{path}{model_code}_value_net.pth")
            self.model.policy.value_net.load_state_dict(value_net_state)

            # Load parameters
            with open(f"{path}_params.json", 'r') as f:
                params = json.load(f)
                # Update model parameters if needed
                self.model.learning_rate = params['learning_rate']
                self.model.gamma = params['gamma']
                self.model.n_steps = params['n_steps']
                self.model.n_epochs = params['n_epochs']
                # ... update other parameters as needed

            print(f"Model components loaded successfully from {path}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()

    def evaluate(self, episodes=5, deterministic=True, render=False):
        """Evaluate the agent's performance"""
        episode_rewards = []
        episode_lengths = []
        episode_successes = []

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
                
                if render:
                    self.env.render()
                
                if done:
                    success = info.get('completed', False)
                    episode_successes.append(success)
                    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, " 
                          f"Length = {episode_length}, Success = {success}")
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # Output statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        success_rate = np.mean(episode_successes) * 100

        print("\nEvaluation Results:")
        print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Mean Episode Length: {mean_length:.1f}")
        print(f"Success Rate: {success_rate:.1f}%")

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_successes': episode_successes,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'success_rate': success_rate
        }