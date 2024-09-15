import airsim
import numpy as np
import Tools.AirsimTools as airsimtools
from DataProcessor import DataProcessor
import time
import gymnasium as gym
from gymnasium import spaces


class AirSimEnv(gym.Env):
    def __init__(self, drone_name, config, lidar_sensor="lidar", camera = "camera", target_name = "BP_Grid", spawn_object_name = "BP_spawn_point", distance_range=(0, 5), maping_range=(1, 3)):
        super(AirSimEnv, self).__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, drone_name)
        self.client.armDisarm(True, drone_name)
        self.config = config
        self.processor = DataProcessor(config)
        self.drone_name = drone_name
        self.lidar_sensor = lidar_sensor
        self.camera = camera
        self.target_name = target_name
        self.spawn_object_name = spawn_object_name
        self.distance_range = distance_range
        self.maping_range = maping_range 
        self.spawn_points = airsimtools.get_targets(self.client, self.drone_name, self.client.simListSceneObjects(f'{self.spawn_object_name}[\w]*'), 2, 1)
        self.targets = airsimtools.get_targets(self.client, self.drone_name, self.client.simListSceneObjects(f'{self.target_name}[\w]*'), 2, 1)
        self.max_distance_to_target = 500
        self.complited_reward = 100
        self.collision_penalty = -100  # Penalty for collision
        self.distance_reward_factor = 1.0  # Reward scaling for distance to target
        self.smoothness_penalty_factor = -0.1  # Penalty for sudden movement changes

        # Define the observation space based on the config
        point_numbers = self.config["point_numbers"]
        resize = self.config["resize"]
        depth_image_size = resize[0] * resize[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(point_numbers * 3 + depth_image_size + 3,),  # LIDAR points (x, y, z) + Depth image + Target position (x, y, z)
            dtype=np.float32
        )
        # Define the action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # Example action space

    def reset(self):
        airsimtools.reset_drone_to_random_spawn_point(self.client, self.drone_name, self.spawn_points)
        time.sleep(1)
        self.targets = airsimtools.get_targets(self.client, self.drone_name, self.client.simListSceneObjects(f'{self.target_name}[\w]*'), 2, 1)
        

    def takeoff(self):
        self.client.takeoffAsync(1, vehicle_name=self.drone_name).join()

    def get_lidar_data(self):
        lidar_data = self.client.getLidarData(vehicle_name=self.drone_name, lidar_name=self.lidar_sensor)
        if len(lidar_data.point_cloud) < 3:
            return np.array([])
        
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        return points

    def get_depth_image(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        ], vehicle_name=self.drone_name)

        if responses and responses[0].width != 0 and responses[0].height != 0:
            # convert to np.array
            img1d = np.array(responses[0].image_data_float, dtype=np.float32)
            img2d = img1d.reshape(responses[0].height, responses[0].width)
            return img2d
        else:
            return None
    
    def computed_reward(self, velocity):
        """
        Calculate the reward function and determine if the episode should end based on the drone's state.

        return: reward(bool), done(bool), completed(bool)
        """
        state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        position = np.array([state.kinematics_estimated.position.x_val,
                            state.kinematics_estimated.position.y_val,
                            state.kinematics_estimated.position.z_val])
        distance_to_target = np.linalg.norm(position - target_position)

        done, completed = self.check_done(distance_to_target)

        if done and completed:
            return self.complited_reward, done, completed
        elif done and not completed:
            return self.collision_penalty, done, completed
        else:
            target_position = np.array(self.targets[0])

            # Get previous distance to target, if not exist, get -1
            prev_distance = getattr(self, f'prev_distance_{self.drone_name}', -1)
            if distance_to_target < prev_distance or prev_distance == -1:
                distance_reward = 5
            else:
                distance_reward = -3
            # Save previous distance
            setattr(self, f'prev_distance_{self.drone_name}', distance_to_target)

            # Get previous velocity, if not exist, get -1
            prev_velocity = getattr(self, f'prev_velocity_{self.drone_name}', -1)
            # Calculate the rate of change of velocity and apply a smoothness penalty
            if prev_velocity == -1:
                velocity_change_penalty = 0
            else:
                velocity_change_penalty = np.linalg.norm(velocity - prev_velocity) * self.smoothness_penalty_factor

            # Save previous velocity
            setattr(self, f'prev_velocity_{self.drone_name}', velocity)

            # Get final reward
            reward = distance_reward + velocity_change_penalty

            return reward, False, False

    def get_target_list(self, object_name):
        objects = self.client.simListSceneObjects(f'{object_name}[\w]*')
        targets = airsimtools.get_targets(self.client, self.drone_name, objects, 2, 1)

        return targets

    def get_observation(self):
        depth_image = self.get_depth_image()
        lidar_data = self.get_lidar_data()
        if self.targets:
            curr_target = self.targets[0]
        else:
            curr_target = [np.nan, np.nan, np.nan]

        return self.processor.process(lidar_data, depth_image, curr_target)

    def step(self, action):
        n, e, d = action
        prev_distance = getattr(self, f'prev_distance_{self.drone_name}', -1)
        if prev_distance == -1:
            speed = 5
        else:
            speed = airsimtools.map_value(self.distance_range, self.maping_range, prev_distance)
        n, e, d = airsimtools.scale_and_normalize_vector([n, e, d], speed)
        self.client.moveByVelocityAsync(n, e, d, duration=1, vehicle_name=self.drone_name).join()
        next_state = self.get_observation()

        reward, done, completed = self.computed_reward([n, e, d])

        return next_state, reward, done, {}

    def close(self):
        self.client.armDisarm(False, self.drone_name)
        self.client.enableApiControl(False, self.drone_name)

    def check_done(self, distance_to_target):
        '''
        return: done(bool), complete(bool)
        '''
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
        # Calculate distance to the current target
        if len(self.targets) == 0:
            # No more targets, end the episode
            return True, True        
        elif collision_info.has_collided or distance_to_target >= self.max_distance_to_target:
            # Collision or too far from the target, end the episode
            return True, False
        else:
            return False, False

class AirSimMultiDroneEnv(gym.Env):
    def __init__(self, config, drone_list):
        super(AirSimMultiDroneEnv, self).__init__()
        self.config = config
        self.drones = {drone_name: AirSimEnv(drone_name, config) for drone_name in drone_list}
        self.action_space = spaces.Dict({
            drone_name: spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
            for drone_name in drone_list
        })