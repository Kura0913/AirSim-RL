import numpy as np
import cv2
import torch

class DataProcessor:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.point_numbers = config["point_numbers"]
        self.resize_shape = tuple(config["resize"])

    def process(self, lidar_data, depth_image, drone_position, target_position):
        """
        Process data from sensors based on the given mode.

        Args:
        - env (AirSimEnv): The simulation environment.
        - config (dict): Configuration for data processing.

        Returns:
        - np.ndarray: Processed sensor data.
        """
        if self.config["mode"] == "lidar_mode":
            processed_data = self.sample_point_cloud(lidar_data, num_points=self.config["point_numbers"])

            # Zero padding for depth image data
            depth_image_size = self.config["resize"][0] * self.config["resize"][1]
            depth_image_zeros = np.zeros(depth_image_size)
            processed_data = np.concatenate([processed_data.flatten(), depth_image_zeros, drone_position, target_position])
            
        elif self.config["mode"] == "camera_mode":
            processed_data = self.preprocess_depth_image(depth_image, resize=self.config["resize"]).numpy().flatten()

            # Zero padding for point cloud data
            point_cloud_zeros = np.zeros(self.config["point_numbers"] * 3)
            processed_data = np.concatenate([point_cloud_zeros, processed_data, drone_position, target_position])

        elif self.config["mode"] == "all_sensors":
            processed_depth_image = self.preprocess_depth_image(depth_image, resize=self.config["resize"]).numpy().flatten()
            sampled_point_cloud = self.sample_point_cloud(lidar_data, num_points=self.config["point_numbers"]).flatten()
            processed_data = np.concatenate([sampled_point_cloud, processed_depth_image, drone_position, target_position])

        # Convert the processed data to a torch tensor and move to the same device as the model
        processed_data_tensor = torch.tensor(processed_data).to(self.device)

        return processed_data_tensor


    def preprocess_depth_image(self, depth_image, resize, max_depth=255.0, min_depth_threshold=1.0, ignore_value=-999):
        """
        Preprocess the depth image, setting pixels below a certain threshold to a specified ignore value.

        Args:
        - depth_image (np.ndarray): Raw depth image. If None, a zero matrix is created.
        - resize (tuple): Target size for resizing the image.
        - max_depth (float): Normalized maximum depth value.
        - min_depth_threshold (float): Minimum threshold to ignore depth values.
        - ignore_value (float): The flag value of the ignored pixel.

        Returns:
        - torch.Tensor: Processed depth images, suitable for use in neural networks.
        """
        resize = (resize[0], resize[1])

        # If depth_image is None, create a zero matrix
        if depth_image is None:
            depth_image_resized = np.zeros(resize, dtype=np.float32)
        else:
            # Resize the image
            depth_image_resized = cv2.resize(depth_image, resize)
        
        # Normalized depth value
        depth_image_normalized = depth_image_resized / max_depth
        
        # Set the value to nan if the value is over max_depth
        depth_image_normalized[depth_image_normalized > (min_depth_threshold / max_depth)] = ignore_value    
        
        # Convert to PyTorch tensor format
        depth_image_tensor = torch.from_numpy(depth_image_normalized).unsqueeze(0).float()
        
        return depth_image_tensor
    
    def sample_point_cloud(self, point_cloud, num_points=1024):
        """
        Sample points from a point cloud or return a zero matrix if input is None.

        Args:
        - point_cloud (np.ndarray): Raw point cloud data. If None, a zero matrix is created.
        - num_points (int): Number of points to sample.

        Returns:
        - np.ndarray: Sampled point cloud data.
        """
        # If point_cloud is None, create a zero matrix
        if point_cloud is None:
            return np.zeros((num_points, 3), dtype=np.float32)
        
        num_points_in_cloud = point_cloud.shape[0]
        
        if num_points_in_cloud >= num_points:
            choice = np.random.choice(num_points_in_cloud, num_points, replace=False)
        else:
            choice = np.random.choice(num_points_in_cloud, num_points, replace=True)
        
        return point_cloud[choice, :]