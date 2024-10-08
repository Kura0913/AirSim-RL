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
        self.grid_size = config["grid_size"]

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
        Sample points from a point cloud using average sampling.

        Args:
        - point_cloud (np.ndarray): Raw point cloud data.

        Returns:
        - torch.Tensor: Sampled point cloud data.
        """
        if point_cloud is None or point_cloud.shape[0] == 0:
            return torch.zeros((self.point_numbers, 3), device=self.device)

        # calculate point cloud edge
        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)
        
        # calculate grid size
        grid_sizes = (max_coords - min_coords) / self.grid_size
        
        grid_indices = np.floor((point_cloud - min_coords) / grid_sizes).astype(int)
        
        # Create a dictionary to store the points in each grid.
        grid_dict = {}
        for i, point in enumerate(point_cloud):
            grid_key = tuple(grid_indices[i])
            if grid_key not in grid_dict:
                grid_dict[grid_key] = []
            grid_dict[grid_key].append(point)
        
        # Sample points from each grid.
        sampled_points = []
        points_per_grid = max(1, self.point_numbers // len(grid_dict))
        
        for grid_points in grid_dict.values():
            n_points = min(len(grid_points), points_per_grid)
            sampled_points.extend(np.random.choice(grid_points, n_points, replace=False))
        
        # If the sampled points are not enough, add them randomly from all points.
        if len(sampled_points) < self.point_numbers:
            additional_points = np.random.choice(point_cloud, self.point_numbers - len(sampled_points), replace=False)
            sampled_points.extend(additional_points)
        
        # If too many points are sampled, redundant points are randomly deleted.
        if len(sampled_points) > self.point_numbers:
            sampled_points = np.random.choice(sampled_points, self.point_numbers, replace=False)
        
        sampled_point_cloud = np.array(sampled_points)
        
        return torch.from_numpy(sampled_point_cloud).float().to(self.device)