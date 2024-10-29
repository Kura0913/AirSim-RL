import Tools.AirsimTools as airsimtools
import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np

class AirsimEnv(gym.Env):
    def __init__(self, drone_name, config, camera_name = "camera", goal_name = "BP_Grid", 
                 distance_sensor_list = ["front", "left", "right", "lfront", "rfront", "lfbottom", "rfbottom", "rbbottom", "lbbottom", 'top']):
        # config setting
        self.config = config
        # env variable
        self.drone_name = drone_name
        self.camera_name = camera_name
        self.goal_name = goal_name
        self.distance_sensor_list = distance_sensor_list
        self.target_resize = config['resize']
        self.start_positon = [0, 0, 0]
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
        # define action sapce, observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
        self.observation_space = spaces.Dict({
            'depth_image': spaces.Box(low=0, high=255, shape=(1, self.target_resize[0], self.target_resize[1])),
            'position': spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            'distance': spaces.Box(low=0, high=np.inf, shape=(1,))
        })

        self.goal_position = np.array(self._load_goal_position(2))

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        airsimtools.reset_drone(self.client, self.drone_name)
        self.client.takeoffAsync().join()
        self.end_eposide = False
        self.total_reward = 0
        self.steps = 0
        self.episode += 1
        self.start_positon = airsimtools.get_drone_position_list(self.client, self.drone_name)

        return self._get_obs(), dict()

    def step(self, action):
        self.steps += 1
        # execute action
        n, e, d = action
        self.client.moveByVelocityAsync(float(n), float(e), float(d), 1).join()
        
        obs = self._get_obs()
        done, completed = self._check_done(obs)
        reward = self._compute_reward(action, obs, done, completed)
        info = {"completed": completed}
        self.total_reward += reward
        if done:
            self.end_eposide = True
            self.episode_rewards.append(self.total_reward)
        
        return obs, reward, done, done, info
    
    def _load_goal_position(self, round_decimals):
        goal_objects = self.client.simListSceneObjects(f'{self.goal_name}[\w]*')
        for goal_object in goal_objects:
            goal_position = self.client.simGetObjectPose(goal_object).position
            goal_position = [np.round(goal_position.x_val, round_decimals), np.round(goal_position.y_val, round_decimals), np.round(goal_position.z_val, round_decimals)]
            goal_position = airsimtools.check_negative_zero(goal_position[0], goal_position[1], goal_position[2])
        
        return goal_position

    def _get_obs(self):
        # get depth image
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False)])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_image = np.reshape(depth_image, (responses[0].height, responses[0].width))
        depth_image = np.array(depth_image * 255, dtype=np.uint8)
        depth_image_resized = np.resize(depth_image, self.target_resize)
        depth_image_final = np.expand_dims(depth_image_resized, axis=0)
        # get drone position
        position = self.client.getMultirotorState().kinematics_estimated.position.to_numpy_array()

        # calculate distance to goal position
        distance = np.linalg.norm(position - self.goal_position)

        return {
            'depth_image': depth_image_final,
            'position': position,
            'distance': np.array([distance])
        }

    def _compute_reward(self, action, obs, done:bool, completed:bool, d_soft=2, d_hard=0.5):
        r_t = 0
        
        # R_fly: reward for flying towards destination and following predefined route
        distance_to_destination = np.linalg.norm(obs['position'] - self.goal_position)
        distance_to_route = self._calculate_distance_to_route(obs['position'], self.start_positon, self.goal_position)

        max_possible_distance = np.linalg.norm(self.goal_position - self.start_positon)
        progress = 1.0 - (distance_to_destination / max_possible_distance)
        route_adherence = 1.0 - min(1.0, distance_to_route / d_soft)
        
        R_fly = 5.0 * (progress + route_adherence)
        
        # R_goal: reward for reaching the destination
        R_goal = 100 if done and completed else 0
        
        # R_collision: penalty for collision
        R_collision = -100 if done and not completed else 0
        
        # R_margin: penalty for getting too close to obstacles
        C1, C2 = 10, 20  # Constants to be tuned empirically
        d_obstacle = self._get_min_distance_sensor_value()
        if d_obstacle < d_hard:
            R_margin = -C2 / d_obstacle
        elif d_obstacle < d_soft:
            R_margin = -C1 * (1 - d_obstacle / d_soft)
        else:
            R_margin = 0
        
        r_t = R_fly + R_goal + R_collision + R_margin
        return r_t

    def _check_done(self, obs):
        '''
        Return: done(bool), mission_completed(bool)
        '''
        collision_info = self.client.simGetCollisionInfo()
        distance = obs['distance'][0]
        if collision_info.has_collided or self.steps >= self.config['max_steps']: # collision happend or steps reach max_steps
            return True, False
        if distance < 0.1: # reach destination
            return True, True
        return False, False # episode continue

    def close(self):
        self.client.enableApiControl(False)

    def _get_min_distance_sensor_value(self):
        distance_min_value = -1
        for distance_sensor in self.distance_sensor_list:
            value = self.client.getDistanceSensorData(distance_sensor, self.drone_name).distance
            if value < distance_min_value or distance_min_value < 0:
                distance_min_value = value
        
        return distance_min_value
    
    def _calculate_distance_to_route(self, current_position, start_point, end_point):
        # convet variable to numpy array
        p = np.array(current_position)
        a = np.array(start_point)
        b = np.array(end_point)
        
        ab = b - a
        ap = p - a
        
        # calcuate projection
        projection = np.dot(ap, ab) / np.dot(ab, ab)
        
        # limit the projection on the route
        projection = np.clip(projection, 0, 1)
        
        # calcuate closest point
        closest_point = a + projection * ab
        
        # calcuate distance
        distance = np.linalg.norm(p - closest_point)
        
        return distance
    def _calculate_distance_to_route(self, current_pos, start_pos, goal_pos):
        # convet variable to numpy array
        current_pos = np.array(current_pos)
        start_pos = np.array(start_pos)
        goal_pos = np.array(goal_pos)
        # Vector calculation
        line_vec = goal_pos - start_pos
        point_vec = current_pos - start_pos
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        
        # Calculate projection
        point_proj_len = np.dot(point_vec, line_unitvec)
        
        if point_proj_len < 0:
            # Click in front of the starting point
            return np.linalg.norm(current_pos - start_pos)
        elif point_proj_len > line_len:
            # Click behind the end point
            return np.linalg.norm(current_pos - goal_pos)
        else:
            # Point in the middle of the line segment and calculate the vertical distance
            point_proj = start_pos + line_unitvec * point_proj_len
            return np.linalg.norm(current_pos - point_proj)