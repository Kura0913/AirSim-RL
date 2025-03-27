import Tools.AirsimTools as airsimtools
from RewardCalculator.RewardCalculator import DroneRewardCalculator
from Env.AirsimBaseEnv import BaseMultirotorEnv
from gymnasium import spaces
import airsim
import numpy as np
import time
import random

class AirsimEnv(BaseMultirotorEnv):
    def __init__(self, drone_name, config:dict, camera_name = "camera", goal_name = "BP_Grid", start_point_name = "BP_StartPoint"):
        # config setting
        self.config = config
        # env variable
        self.drone_name = drone_name
        self.camera_name = camera_name
        self.goal_name = goal_name
        self.start_point_name = start_point_name
        self.target_resize = [config['resize']['width'], config['resize']['height']]
        # set airsim api client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, drone_name)
        self.client.armDisarm(True, drone_name)
        # statistical variables
        self.steps = 0
        self.total_reward = 0
        self.episode = 0
        self.episode_rewards = []
        self.end_eposide = False
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Dict({
            'depth_image': spaces.Box(low=0, high=255, shape=(1, self.target_resize[0], self.target_resize[1]))
        })

        self.start_poses = self._load_start_points() # get poses of all spawn points
        self.map_num = len(self.start_poses) - 1 # get map num
        self.start_pose = self.start_poses[0] # initial start point pose
        self.start_position = [self.start_pose.position.x_val, self.start_pose.position.y_val, self.start_pose.position.z_val] # initial start point position
        self.goal_position = np.array(self._load_goal_position(2))
        # reward calculator
        self.reward_calculator = DroneRewardCalculator(self.client, self.drone_name)

    def reset(self, seed=None):
        reset_flag = True
        map_idx = random.randint(0, self.map_num)
        self.start_pose = self.start_poses[map_idx] # get random map's spawn point
        self.start_position = [self.start_pose.position.x_val, self.start_pose.position.y_val, self.start_pose.position.z_val] # get spawn point position
        print(f'Load map: {map_idx}')
        if seed is not None:
            np.random.seed(seed)
        while reset_flag:
            airsimtools.reset_drone(self.client, self.drone_name)
            self.client.simSetVehiclePose(self.start_pose, True, vehicle_name=self.drone_name)
            self.end_eposide = False
            self.total_reward = 0
            self.steps = 0
            time.sleep(0.5)
            position = self.client.getMultirotorState(self.drone_name).kinematics_estimated.position.to_numpy_array()
            if np.linalg.norm(position - self.start_position) < 5 and  not self.client.simGetCollisionInfo(self.drone_name).has_collided:
                reset_flag = False
        self.episode += 1

        return self._get_obs(), dict()

    def step(self, action):
        self.steps += 1
        # execute action
        if len(action) == 3:
            n, e, d = action
        else:
            e, d = action

        self.client.moveByVelocityAsync(1, float(e), float(d), 1, vehicle_name=self.drone_name).join()
        self.client.moveByVelocityAsync(0.0, 0.0, 0.0, 0.2, vehicle_name=self.drone_name).join()
        
        obs = self._get_obs()
        done, completed = self._check_done(obs)
        reward = self.reward_calculator.compute_reward(action, obs, self.steps, done, completed)
        info = {"completed": completed}
        self.total_reward += reward
        if done:
            self.end_eposide = True
            self.episode_rewards.append(self.total_reward)
        
        return obs, reward, done, done, info
    
    def _get_airsim_client(self):
        return self.client
    
    def _load_start_points(self):
        player_start_objects = self.client.simListSceneObjects(f'{self.start_point_name}[\w]*')
        start_poses =  []
        for player_start_object in player_start_objects:
            start_poses.append(self.client.simGetObjectPose(player_start_object))

        return start_poses

    def _load_goal_position(self, round_decimals):
        goal_objects = self.client.simListSceneObjects(f'{self.goal_name}[\w]*')
        for goal_object in goal_objects:
            goal_position = self.client.simGetObjectPose(goal_object).position
            goal_position = [np.round(goal_position.x_val, round_decimals), np.round(goal_position.y_val, round_decimals), np.round(goal_position.z_val, round_decimals)]
            goal_position = airsimtools.check_negative_zero(goal_position[0], goal_position[1], goal_position[2])
        
        return goal_position

    def _get_obs(self):
        # get depth image
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_image = np.reshape(depth_image, (responses[0].height, responses[0].width))
        depth_image = np.array(depth_image * 255, dtype=np.uint8)
        depth_image_resized = np.resize(depth_image, self.target_resize)
        depth_image_final = np.expand_dims(depth_image_resized, axis=0)

        return {
            'depth_image': depth_image_final
        }

    def _check_done(self, obs):
        '''
        Return: done(bool), mission_completed(bool), arrive_max_steps(bool)
        '''
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided and self.goal_name in collision_info.object_name: # reach destination
            return True, True
        if collision_info.has_collided: # collision happend or steps reach max_steps
            return True, False
        return False, False # episode continue
    
    def close(self):
        self.client.enableApiControl(False)