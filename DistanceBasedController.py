# HumanController.py
import numpy as np
from typing import Dict, List, Tuple
import airsim
import Tools.AirsimTools as airsimtools

MAX_DISTANCE = 10
DISTANCE_RANGE = (0, 10)
MAPING_RANGE = (0, 1)

class DistanceBasedController:
    def __init__(self, client: airsim.MultirotorClient, drone_name, sensor_list: List[str], 
                 safety_distance: float = 1, goal_name = "BP_Grid"):
        """
        Initialize the distance-based controller
        
        Args:
            client: AirSim client instance
            sensor_list: List of distance sensor names
            safety_distance: Minimum safe distance from obstacles
        """
        self.client = client
        self.drone_name = drone_name
        self.sensor_list = sensor_list
        self.sensor_list += ["front1", "front2"]
        self.safety_distance = safety_distance
        self.rise_velocity = 1.0
        self.goal_distance_limit = 2
        self.goal_name = goal_name
        self.goal_position = np.array(self._load_goal_position(2))
        
        # Define drone limits (safety thresholds)
        self.drone_limits = {
            'front': safety_distance,
            'left': safety_distance,
            'right': safety_distance,
            'top': safety_distance,
            'bottom': safety_distance
        }
        
        # Define mapping ranges for velocity scaling
        self.distance_range = [0, safety_distance]
        self.mapping_range = [0, 1.0]

    def _load_goal_position(self, round_decimals):
        goal_objects = self.client.simListSceneObjects(f'{self.goal_name}[\w]*')
        for goal_object in goal_objects:
            goal_position = self.client.simGetObjectPose(goal_object).position
            goal_position = [np.round(goal_position.x_val, round_decimals), np.round(goal_position.y_val, round_decimals), np.round(goal_position.z_val, round_decimals)]
            goal_position = airsimtools.check_negative_zero(goal_position[0], goal_position[1], goal_position[2])
        
        return goal_position
    
    def get_sensor_readings(self) -> Dict[str, float]:
        """Get all distance sensor readings"""
        readings = {}
        for sensor in self.sensor_list:
            try:
                reading = self.client.getDistanceSensorData(sensor, self.drone_name).distance
                readings[sensor] = reading  # Cap maximum distance
            except:
                readings[sensor] = MAX_DISTANCE  # Default to large value if sensor fails
        return readings
    
    def get_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate control action based on sensor readings
        Returns: action array [x, y, z] in NED coordinate system
        """
        # get drone position
        position = self.client.getMultirotorState().kinematics_estimated.position.to_numpy_array()
        # get original velocity
        velocity = self.goal_position - position
        velocity = velocity / np.linalg.norm(velocity)

        readings = self.get_sensor_readings()
        velocity_factor = 1.0
        self.rise_velocity = 1.0
        # Get all sensor readings
        # front = readings.get('front', MAX_DISTANCE)
        right = readings.get('right', MAX_DISTANCE)
        left = readings.get('left', MAX_DISTANCE)
        top = readings.get('top', MAX_DISTANCE)
        rfront = readings.get('rfront', MAX_DISTANCE)
        lfront = readings.get('lfront', MAX_DISTANCE)
        
        # Get bottom sensors and find minimum
        bottom_sensors = ["lfbottom", "rfbottom", "rbbottom", "lbbottom"]
        front_sensors = ["front", "front1", "frone2"]
        bottom = min(readings.get(sensor, MAX_DISTANCE) for sensor in bottom_sensors)
        front = min(readings.get(sensor, MAX_DISTANCE) for sensor in front_sensors)
        # Initialize action vector [x, y, z]
        # get drone position
        position = self.client.getMultirotorState().kinematics_estimated.position.to_numpy_array()
        action = self.goal_position - position
        action = action / np.linalg.norm(action)
        stop_action = False
        
        # 1. First check vertical clearance
        if bottom < self.drone_limits['bottom'] and top > self.drone_limits['top']:
            # Need to move up
            correct_velocity = [0, 0, -1 + airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, bottom)]  # Negative for up in NED
            stop_action = True
        elif top < self.drone_limits['top'] and bottom > self.drone_limits['bottom']:
            # Need to move down
            correct_velocity = [0, 0, 1 - airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, bottom)]
            stop_action = True
            
        else:
            velocity_factor = 1.0
            if front < self.drone_limits['front'] and left < self.drone_limits['left'] and right < self.drone_limits['right']:
                # All sensors detecting obstacles - move up
                correct_velocity = [0, 0, -1]
                stop_action = True
                
            elif front < self.drone_limits['front'] and left < self.drone_limits['left']:
                # Front and left blocked - move right
                correct_velocity = [0, 1, 0]
                stop_action = True
                
            elif front < self.drone_limits['front'] and right < self.drone_limits['right']:
                # Front and right blocked - move left
                correct_velocity = [0, -1, 0]
                stop_action = True
                
            elif left < self.drone_limits['left']:
                # Left blocked - move right
                correct_velocity = [0, 1, 0]
                velocity_factor = airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, left)
                stop_action = False
                
            elif right < self.drone_limits['right']:
                # Right blocked - move left
                correct_velocity = [0, -1, 0]
                velocity_factor = airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, right)
                stop_action = False
                
            elif front < self.drone_limits['front']:
                # Front blocked - choose better side
                if rfront > lfront and rfront < self.drone_limits['right'] and lfront < self.drone_limits['left']:# there is more space at right side
                    correct_velocity = [0, 1, 0]
                    velocity_factor = airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, right)
                    stop_action = False
                else:# there is more space at left side
                    correct_velocity = [0, -1, 0]
                    velocity_factor = airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, left)
                stop_action = False
            
            else:# normal
                correct_velocity = [0, 0, 0]
                stop_action = False

        if stop_action and self.goal_distance_limit < np.linalg.norm(self.goal_position - position): # drone is very close to obstacles, need to stop moving
            return correct_velocity
        else:# there is enough space to correct the velocity
            velocity = [i * velocity_factor for i in velocity]
            correct_velocity[2] -= (self.rise_velocity - velocity_factor)
            correct_velocity_factor = 1
            if self.goal_distance_limit > np.linalg.norm(self.goal_position - position):
                correct_velocity_factor = (np.linalg.norm(self.goal_position - position) - 0.5) / self.goal_distance_limit
                correct_velocity_factor = float(correct_velocity_factor)
                if correct_velocity_factor < 0:
                    correct_velocity_factor = 0
            correct_velocity = [num * correct_velocity_factor for num in correct_velocity]
            return np.sum([velocity, correct_velocity], axis=0)