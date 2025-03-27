
import gymnasium as gym
from gymnasium import spaces
import airsim

class BaseMultirotorEnv(gym.Env):
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

    def reset(self, seed=None):
        pass

    def step(self, action):
        pass


    def _get_obs(self):
        pass

    def _check_done(self, obs):
        '''
        Return: done(bool), mission_completed(bool), arrive_max_steps(bool)
        '''
        pass
    
    def close(self):
        self.client.enableApiControl(False)