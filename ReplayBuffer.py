import numpy as np
from typing import Dict, List, Any, Optional
import torch as th
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import ReplayBufferSamples
import gymnasium as gym

class PrioritizedReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        device: str = "auto",
        n_envs: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        **kwargs
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs
        )
        
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_increment = float(beta_increment)
        self.epsilon = float(epsilon)
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.pos = 0
    
    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        max_priority = self.priorities.max() if self.pos > 0 else 1.0
        super().add(obs, next_obs, action, reward, done, infos)
        self.priorities[self.pos - 1] = max_priority
    
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        upper_bound = self.pos if not self.full else self.buffer_size
        
        priorities = self.priorities[:upper_bound]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        indices = np.random.choice(upper_bound, size=batch_size, p=probs)
        weights = (upper_bound * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        data = self._get_samples(indices, env=env)
        
        weights_tensor = th.FloatTensor(weights).to(self.device).reshape(-1, 1)
        
        # Create new ReplayBufferSamples, keeping original observations unchanged
        return ReplayBufferSamples(
            observations=data.observations,
            actions=data.actions,
            next_observations=data.next_observations,
            dones=data.dones,
            rewards=data.rewards * weights_tensor  # Apply weights to rewards
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = np.abs(priorities) + self.epsilon

class HumanDemonstrationBuffer:
    """Buffer for storing human demonstrations"""
    def __init__(self, buffer_size: int, observation_space: Dict, action_space: Any, device: str = "auto"):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = th.device(device)
        self.pos = 0
        self.full = False
        
        # Initialize buffers for each observation component
        self.observations = {
            'depth_image': np.zeros((buffer_size, 1, observation_space['depth_image'].shape[1], 
                                   observation_space['depth_image'].shape[2]), 
                                   dtype=np.float32),
            'position': np.zeros((buffer_size, 3), dtype=np.float32),
            'goal': np.zeros((buffer_size, 3), dtype=np.float32),
            'distance': np.zeros((buffer_size, 1), dtype=np.float32)
        }
        
        self.actions = np.zeros((buffer_size, *action_space.shape), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        
    def add(self, obs: Dict[str, np.ndarray], action: np.ndarray, reward: float, 
            done: bool) -> None:
        # Copy observation components
        for key in self.observations.keys():
            self.observations[key][self.pos] = obs[key]
        
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0
            
    def sample(self, batch_size: int) -> ReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        indices = np.random.randint(0, upper_bound, size=batch_size)
        
        observations = {
            key: th.as_tensor(self.observations[key][indices]).to(self.device)
            for key in self.observations.keys()
        }
        
        return ReplayBufferSamples(
            observations=observations,
            actions=th.as_tensor(self.actions[indices]).to(self.device),
            next_observations=observations,  # For demonstration data, we don't need next obs
            rewards=th.as_tensor(self.rewards[indices]).to(self.device),
            dones=th.as_tensor(self.dones[indices]).to(self.device)
        )