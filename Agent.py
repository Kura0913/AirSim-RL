from Network.CustomFeaturesExtractor import CustomFeaturesExtractor
from Policy.CustomTD3Policy import CustomTD3Policy
from stable_baselines3 import DDPG, PPO
import torch as th
from ReplayBuffer import PrioritizedReplayBuffer, HumanDemonstrationBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
import numpy as np
import json
from typing import Dict, List, Any, Optional

class DDPGAgent:
    def __init__(self, env, config):
        self.config = config
        policy_kwargs = dict(
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=512+32+32+32),
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
            learning_starts=10000,
            gamma=0.9999,
            buffer_size=500000,
            learning_rate=self.config["learning_rate"],
            batch_size=256,
            device=th.device(self.config['device'])

        )

        self.model.replay_buffer = PrioritizedReplayBuffer(
            buffer_size=500000,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=th.device(self.config['device']),
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
        )

    def train(self, total_timesteps, callback):
        original_train = self.model.learn
        
        def modified_train_step(self, gradient_steps: int) -> None:
            """Modified training steps to support PER"""
            for _ in range(gradient_steps):
                samples, indices, weights = self.replay_buffer.sample(self.batch_size)
                
                # Calculate TD error
                with th.no_grad():
                    next_actions = self.actor_target(samples.next_observations)
                    next_q_values = self.critic_target(samples.next_observations, next_actions)
                    target_q_values = samples.rewards + (1 - samples.dones) * self.gamma * next_q_values
                
                current_q_values = self.critic(samples.observations, samples.actions)
                td_errors = th.abs(target_q_values - current_q_values).detach().cpu().numpy()
                
                # Update priority
                self.replay_buffer.update_priorities(indices, td_errors)
                
                # Apply importance sampling weights
                critic_loss = ((target_q_values - current_q_values) ** 2) * weights
                critic_loss = critic_loss.mean()
                
                # update critic
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
                
                # update actor
                actor_loss = -self.critic(samples.observations, self.actor(samples.observations)).mean()
                
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                
                # update target net
                self._target_soft_update()
        
        # Replace the original train_step method
        self.model._train_step = modified_train_step.__get__(self.model)
        
        # Execute training
        original_train(total_timesteps=total_timesteps, callback=callback)

    def save(self, path):
        """Save model components separately instead of saving the whole model"""
        try:
            # Save policy network state dict
            policy_state = self.model.policy.state_dict()
            th.save(policy_state, f"{path}_policy.pth")

            # Save actor network state dict
            actor_state = self.model.actor.state_dict()
            th.save(actor_state, f"{path}_actor.pth")

            # Save critic network state dict
            critic_state = self.model.critic.state_dict()
            th.save(critic_state, f"{path}_critic.pth")

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
        policy_kwargs = dict(
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=32),
            net_arch=[dict(pi=[256, 256, 256], vf=[256, 256])],
            config=config,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs=dict(
                eps=1e-5,
                weight_decay=1e-4,
                amsgrad=True
            )
        )
        self.model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            gamma=0.9999,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=self.config["learning_rate"],
            device=th.device(self.config['device'])
        )

    def train(self, total_timesteps, callback):
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path):
        self.model.save(path)

class HumanGuidedDDPGAgent(DDPGAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.demo_buffer = HumanDemonstrationBuffer(
            buffer_size=50000,  # Adjust size as needed
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=config['device']
        )
        self.demo_ratio = 0.2  # Ratio of demonstration data in training batch
        
    def add_demonstration(self, obs: Dict[str, np.ndarray], action: np.ndarray, 
                         reward: float, done: bool):
        """Add human demonstration to demo buffer"""
        self.demo_buffer.add(obs, action, reward, done)
        
    def _demo_train_step(self, gradient_steps: int) -> None:
        """Modified training step incorporating demonstration data"""
        for _ in range(gradient_steps):
            # Sample from both buffers
            replay_samples = self.model.replay_buffer.sample(
                int(self.model.batch_size * (1 - self.demo_ratio))
            )
            demo_samples = self.demo_buffer.sample(
                int(self.model.batch_size * self.demo_ratio)
            )
            
            # Combine samples
            combined_samples = ReplayBufferSamples(
                observations={
                    key: th.cat([replay_samples.observations[key], 
                               demo_samples.observations[key]])
                    for key in replay_samples.observations.keys()
                },
                actions=th.cat([replay_samples.actions, demo_samples.actions]),
                next_observations={
                    key: th.cat([replay_samples.next_observations[key], 
                               demo_samples.next_observations[key]])
                    for key in replay_samples.next_observations.keys()
                },
                rewards=th.cat([replay_samples.rewards, demo_samples.rewards]),
                dones=th.cat([replay_samples.dones, demo_samples.dones])
            )
            
            # Calculate losses with demonstration data
            with th.no_grad():
                next_actions = self.model.actor_target(combined_samples.next_observations)
                next_q_values = self.model.critic_target(
                    combined_samples.next_observations, 
                    next_actions
                )
                target_q_values = combined_samples.rewards + \
                    (1 - combined_samples.dones) * self.model.gamma * next_q_values
            
            # Update critic
            current_q_values = self.model.critic(
                combined_samples.observations,
                combined_samples.actions
            )
            critic_loss = ((target_q_values - current_q_values) ** 2).mean()
            
            self.model.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.model.critic.optimizer.step()
            
            # Update actor
            actor_loss = -self.model.critic(
                combined_samples.observations,
                self.model.actor(combined_samples.observations)
            ).mean()
            
            self.model.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.model.actor.optimizer.step()
            
            # Update target networks
            self.model._target_soft_update()