import airsim
import numpy as np

class AirSimEnv:
    def __init__(self, drone_name, lidar_sensor="lidar", camera = "camera"):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, drone_name)
        self.client.armDisarm(True, drone_name)
        self.drone_name = drone_name
        self.lidar_sensor = lidar_sensor
        self.camera = camera

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
        self.takeoff()

    def takeoff(self):
        self.client.takeoffAsync(1, vehicle_name=self.drone_name).join()

    def get_lidar_data(self):
        lidar_data = self.client.getLidarData(vehicle_name=self.drone_name, lidar_name=self.lidar_sensor)
        if len(lidar_data.point_cloud) < 3:
            return np.array([])
        
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        return points

    def get_camera_img(self):
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


    def step(self, action):
        vx, vy, vz = action
        self.client.moveByVelocityAsync(vx, vy, vz, duration=1, vehicle_name=self.drone_name).join()

    def close(self):
        self.client.armDisarm(False, self.drone_name)
        self.client.enableApiControl(False, self.drone_name)
