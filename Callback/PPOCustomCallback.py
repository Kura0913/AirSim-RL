from Callback.BaseCustomCallback import BaseCustomCallback
import os
import torch as th
import json

class PPOCustomCallback(BaseCustomCallback):
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
                
                value_net_path = f"{base_path}_value_net.pth"
                th.save(self.model.policy.value_net.state_dict(), value_net_path)
                print(f"Value network saved to {value_net_path}")
            
            # 2. Save training parameters
            params = {
                'learning_rate': getattr(self.model, 'learning_rate', None),
                'gamma': getattr(self.model, 'gamma', None),
                'n_steps': getattr(self.model, 'n_steps', None),
                'n_epochs': getattr(self.model, 'n_epochs', None),
                'batch_size': getattr(self.model, 'batch_size', None),
                'clip_range': str(getattr(self.model, 'clip_range', None)),
                'ent_coef': getattr(self.model, 'ent_coef', None),
                'vf_coef': getattr(self.model, 'vf_coef', None)
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