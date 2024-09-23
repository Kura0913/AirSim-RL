import numpy as np
import cv2
import torch

class DataProcessor:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.point_numbers = config["point_numbers"]
        self.resize_shape = tuple(config["resize"])
        self.mode = config["mode"]

    def process(self, lidar_data, depth_image, drone_position, target_position):
        """
        Process data from sensors based on the given mode.

        Args:
        - lidar_data (np.ndarray): Raw LiDAR point cloud data.
        - depth_image (np.ndarray): Raw depth image.
        - drone_position (np.ndarray): Drone's current position.
        - target_position (np.ndarray): Target position.

        Returns:
        - dict: Processed sensor data.
        """
        processed_data = {}

        if self.mode in ["lidar_mode", "all_sensors"]:
            processed_data["point_cloud"] = self.sample_point_cloud(lidar_data)
        else:
            processed_data["point_cloud"] = torch.zeros((self.point_numbers, 3), device=self.device)

        if self.mode in ["camera_mode", "all_sensors"]:
            processed_data["depth_image"] = self.preprocess_depth_image(depth_image)
        else:
            processed_data["depth_image"] = torch.zeros((1, *self.resize_shape), device=self.device)

        processed_data["drone_position"] = torch.tensor(drone_position, device=self.device).float()
        processed_data["target_position"] = torch.tensor(target_position, device=self.device).float()

        return processed_data

    def preprocess_depth_image(self, depth_image, max_depth=255.0, min_depth_threshold=1.0, ignore_value=-999):
        """
        Preprocess the depth image.

        Args:
        - depth_image (np.ndarray): Raw depth image.
        - max_depth (float): Normalized maximum depth value.
        - min_depth_threshold (float): Minimum threshold to ignore depth values.
        - ignore_value (float): The flag value of the ignored pixel.

        Returns:
        - torch.Tensor: Processed depth image.
        """
        if depth_image is None:
            return torch.zeros((1, *self.resize_shape), device=self.device)

        depth_image_resized = cv2.resize(depth_image, self.resize_shape)
        depth_image_normalized = depth_image_resized / max_depth
        depth_image_normalized[depth_image_normalized > (min_depth_threshold / max_depth)] = ignore_value
        
        return torch.from_numpy(depth_image_normalized).unsqueeze(0).float()
    
    def sample_point_cloud(self, point_cloud):
        """
        Sample points from a point cloud.

        Args:
        - point_cloud (np.ndarray): Raw point cloud data.

        Returns:
        - torch.Tensor: Sampled point cloud data.
        """
        if point_cloud is None:
            return torch.zeros((self.point_numbers, 3), device=self.device)
        
        num_points_in_cloud = point_cloud.shape[0]
        
        if num_points_in_cloud >= self.point_numbers:
            choice = np.random.choice(num_points_in_cloud, self.point_numbers, replace=False)
        else:
            choice = np.random.choice(num_points_in_cloud, self.point_numbers, replace=True)
        
        sampled_point_cloud = point_cloud[choice, :]
        return torch.from_numpy(sampled_point_cloud).float()