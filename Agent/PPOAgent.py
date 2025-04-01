from Policy.CustomPPOPolicy import CustomPPOPolicy
from stable_baselines3 import PPO
from Agent.BaseAgent import BaseAgent
import torch as th
import json

class PPOAgent(BaseAgent):
    def __init__(self, env, agent_config, training_setting, folder_name):
        self.agent_config = agent_config
        self.env = env
        self.device = training_setting['device']
        self.class_component = self._get_agnet_settings(training_setting, folder_name)
        policy_kwargs = dict(
            features_extractor_class=self.class_component['feature_extractor_class'],
            features_extractor_kwargs=dict(features_dim=32),
            net_arch=[],
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
            verbose=self.agent_config['verbose'],
            gamma=self.agent_config['gamma'],
            n_steps=self.agent_config['n_steps'],
            batch_size=self.agent_config['batch_size'],
            n_epochs=self.agent_config['n_epochs'],
            learning_rate=self.agent_config["learning_rate"],
            device=self.device
        )

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