
import gymnasium as gym
from gymnasium import spaces
import airsim
from abc import ABC, abstractmethod

class BaseMultirotorEnv(gym.Env, ABC):
    def __init__(self, drone_name):
        # env variable
        self.drone_name = drone_name
        # set airsim api client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, drone_name)
        self.client.armDisarm(True, drone_name)

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Dict({
        })

    @abstractmethod
    def reset(self, seed=None):
        pass

    @abstractmethod
    def step(self, action):
        pass
    
    def close(self):
        self.client.enableApiControl(False)