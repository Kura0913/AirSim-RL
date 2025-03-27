from Callback.BaseCustomCallback import BaseCustomCallback
import os
import torch as th
import numpy as np
from datetime import datetime
import json

class DDPGCustomCallback(BaseCustomCallback):
    def _save_model(self):
        try:
            path = f"{self.config['save_path']}{self.folder_name}/"
            os.makedirs(path, exist_ok=True)
            base_path = f"{path}{self.curr_episode}"
            
            # 1. Save network weights
            if hasattr(self.model, 'policy'):
                policy_path = f"{base_path}_policy.pth"
                th.save(self.model.policy.state_dict(), policy_path)
                print(f"Policy network saved to {policy_path}")
            
            if hasattr(self.model, 'actor'):
                actor_path = f"{base_path}_actor.pth"
                th.save(self.model.actor.state_dict(), actor_path)
                print(f"Actor network saved to {actor_path}")
            
            if hasattr(self.model, 'critic'):
                critic_path = f"{base_path}_critic.pth"
                th.save(self.model.critic.state_dict(), critic_path)
                print(f"Critic network saved to {critic_path}")
            
            # 2. Save training parameters
            params = {
                'learning_rate': getattr(self.model, 'learning_rate', None),
                'gamma': getattr(self.model, 'gamma', None),
                'batch_size': getattr(self.model, 'batch_size', None),
                'learning_starts': getattr(self.model, 'learning_starts', None),
                'gradient_steps': getattr(self.model, 'gradient_steps', None),
                'action_noise': str(getattr(self.model, 'action_noise', None))
            }
            
            params_path = f"{base_path}_params.json"
            with open(params_path, 'w') as f:
                json.dump({k: v for k, v in params.items() if v is not None}, f, indent=4)
            print(f"Training parameters saved to {params_path}")
            
            # 3. Save training statistics
            self._save_base_stats(base_path)

        except Exception as e:
            print(f"Warning: Failed to save at episode {self.curr_episode}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()