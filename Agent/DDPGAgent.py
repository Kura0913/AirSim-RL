from PolicyComponent.DDPGFeaturesExtractor import DDPGFeaturesExtractor
from Policy.CustomTD3Policy import CustomTD3Policy
from Agent.BaseAgent import BaseAgent
from stable_baselines3 import DDPG
import torch as th
from ReplayBuffer.PrioritizedReplayBuffer import PrioritizedReplayBuffer
import json

class DDPGAgent(BaseAgent):
    def __init__(self, env, device, agent_config, training_setting, folder_name):
        self.agent_config = agent_config
        self.env = env
        self.class_component = self._get_agnet_settings(training_setting, folder_name)
        policy_kwargs = dict(
            features_extractor_class=DDPGFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=32),
            net_arch=dict(pi=[256, 256, 256], qf=[256, 256]),
            config=training_setting,
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
            verbose=self.agent_config['verbose'],
            gamma=self.agent_config['gamma'],
            learning_starts=self.agent_config['learning_starts'],
            buffer_size=self.agent_config['buffer_size'],
            learning_rate=self.agent_config["learning_rate"],
            batch_size=self.agent_config["buffer_size"],
            device=device
        )

        self.model.replay_buffer = PrioritizedReplayBuffer(
            buffer_size=50000,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
        )

    def train(self, total_timesteps, callback):
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

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

    def load(self, path, model_code:str=''):
        """Load model components separately"""
        try:
            # Load policy network
            policy_state = th.load(f"{path}{model_code}_policy.pth")
            self.model.policy.load_state_dict(policy_state)

            # Load actor network
            actor_state = th.load(f"{path}{model_code}_actor.pth")
            self.model.actor.load_state_dict(actor_state)

            # Load critic network
            critic_state = th.load(f"{path}{model_code}_critic.pth")
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
