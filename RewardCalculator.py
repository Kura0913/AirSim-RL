import numpy as np
import airsim

class DroneRewardCalculator:
    def __init__(self, client: airsim.MultirotorClient, lidar_list, drone_name, start_position, goal_position, center_pixels=(16, 16)):
        self.lidar_list = lidar_list
        self.client = client
        self.drone_name = drone_name
        self.start_positon = start_position
        self.goal_position = goal_position
        # Margin boundaries and constants
        self.d_soft = 2.0
        self.d_hard = 0.5
        self.C1 = 2.0
        self.C2 = 4.0
        self.collision_penalty = -2.0
        self.reach_destination_reward = 2.0
        self.obstacle_avoidance_reward = 0.5
        self.center_height, self.center_width = center_pixels
    
    def compute_reward(self, action, obs, curr_step, done:bool, completed:bool):
        r_t = 0

        # R_collision: penalty for collision
        R_collision = -2 if done and not completed else 0
        
        # R_goal: reward for reaching the destination
        R_goal = 2 if done and completed else 0
        
        r_t += self._get_action_reward(obs)
        # r_t += self._calculate_margin_reward_lidar()

        if done and not completed:
            return R_collision
        
        if done and completed:
            return R_goal
        
        return r_t
    
    def _get_action_reward(self, obs):
        depth_map = obs['depth_image']
        _, h, w = depth_map.shape
        if h < self.center_height or w < self.center_width:
            raise ValueError(
                f"Depth map size ({h}x{w}) is smaller than center region "
                f"size ({self.center_height}x{self.center_width})"
            )
        # Get the center area of ​​the depth map
        center_region = self._get_center_region(depth_map, h, w)

        # Calculate the average depth of the central area and overall
        center_depth = np.mean(center_region)
        overall_depth = np.mean(depth_map)

        depth_max = np.max(depth_map)
        if depth_max > 0:
            normalized_center = center_depth / depth_max
            normalized_overall = overall_depth / depth_max

            # calculate action reward
            if normalized_center > normalized_overall:
                return 0.2  # stay away from obstacles
            else:
                return -0.2  # approaching obstacles

    def _calculate_margin_reward_lidar(self):
        # get point cloud data
        point_cloud = []
        for lidar_name in self.lidar_list:
            point_cloud += self.client.getLidarData(lidar_name, self.drone_name).point_cloud
        points = np.array(point_cloud, dtype=np.float32).reshape(-1, 3)

        # calculate distance for each point
        if points is None or len(points) == 0:
            return 0
        
        distances = np.linalg.norm(points, axis=1)

        d_obstacle = min(distances)

        if d_obstacle < self.d_hard:
            R_margin = -self.C2 / d_obstacle
        elif d_obstacle < self.d_soft:
            R_margin = -self.C1 * (1 - d_obstacle / self.d_soft)
        else:
            R_margin = 0

        return R_margin
        
    def _get_center_region(self, depth_map, h, w):        
        # Make sure the center area is no larger than the original image
        center_h = min(self.center_height, h)
        center_w = min(self.center_width, w)
        
        # Calculate the start and end positions of the center area
        h_start = (h - center_h) // 2
        h_end = h_start + center_h
        w_start = (w - center_w) // 2
        w_end = w_start + center_w

        return depth_map[0, h_start:h_end, w_start:w_end]