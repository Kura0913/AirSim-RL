import numpy as np
import airsim

class DroneRewardCalculator:
    def __init__(self, client: airsim.MultirotorClient, drone_name, center_pixels=(16, 16)):
        self.client = client
        self.drone_name = drone_name
        self.center_height, self.center_width = center_pixels
    
    def compute_reward(self, action, obs, curr_step, done:bool, completed:bool):
        r_t = 0

        # R_collision: penalty for collision
        R_collision = -2 if done and not completed else 0
        
        # R_goal: reward for reaching the destination
        R_goal = 2 if done and completed else 0
        
        action_reward = self._get_action_reward(obs)
        r_t += action_reward
        
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
        else:
            return -0.2
        
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