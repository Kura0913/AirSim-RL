import numpy as np
import airsim

class DroneRewardCalculator:
    def __init__(self, client: airsim.MultirotorClient, distance_sensor_list, drone_name, start_position, goal_position, max_steps):
        self.distance_sensor_list = distance_sensor_list
        self.client = client
        self.drone_name = drone_name
        self.start_positon = start_position
        self.goal_position = goal_position
        self.max_steps = max_steps
        # Margin boundaries and constants
        self.d_soft = 2.0
        self.d_hard = 0.5
        self.C1 = 10.0
        self.C2 = 20.0
        
        # Action reward constants
        self.LOW_VELOCITY = 0.2
        self.HIGH_VELOCITY = 0.8
        
        # Reward weights
        self.MARGIN_WEIGHT = 1.0
        self.ACTION_WEIGHT = 0.3
        self.STEP_WEIGHT = 0.2
    
    def _get_min_distance_sensor_value(self):
        distance_min_value = -1
        for distance_sensor in self.distance_sensor_list:
            value = self.client.getDistanceSensorData(distance_sensor, self.drone_name).distance
            if value < distance_min_value or distance_min_value < 0:
                distance_min_value = value
        
        return distance_min_value
    
    def _get_all_sensor_distances(self):
        distances = {}
        for sensor in self.distance_sensor_list:
            value = self.client.getDistanceSensorData(sensor, self.drone_name).distance
            distances[sensor] = value
        return distances
    def _calculate_margin_reward_v1(self):
        # R_margin: penalty for getting too close to obstacles        
        d_obstacle = self._get_min_distance_sensor_value()
        if d_obstacle < self.d_hard:
            R_margin = -self.C2 / d_obstacle
        elif d_obstacle < self.d_soft:
            R_margin = -self.C1 * (1 - d_obstacle / self.d_soft)
        else:
            R_margin = 5

        return R_margin
    
    def _calculate_margin_reward_v2(self, sensor_distances: dict):
        """
        Optimized V2 version of margin reward calculation
        - More linear scaling
        - Clear reward of 5.0 for safe conditions
        - Bounded minimum reward at -100
        """
        num_sensors = len(sensor_distances)
        
        # Check if all sensors are in safe zone
        if all(distance >= self.d_soft for distance in sensor_distances.values()):
            return 5.0
        
        violations = []
        for sensor, distance in sensor_distances.items():
            if distance < self.d_hard:
                # Linear scaling for hard boundary violations
                severity = max(0.1, distance) / self.d_hard
                violations.append(('hard', severity))
            elif distance < self.d_soft:
                # Linear scaling for soft boundary violations
                severity = (distance - self.d_hard) / (self.d_soft - self.d_hard)
                violations.append(('soft', severity))
        
        if not violations:
            return 5.0
        
        # Calculate penalty based on violation type and severity
        total_penalty = 0
        num_violations = len(violations)
        
        for violation_type, severity in violations:
            if violation_type == 'hard':
                # Hard violations scale from -100 to -50
                penalty = -100 + (50 * severity)
            else:
                # Soft violations scale from -50 to 0
                penalty = -50 + (50 * severity)
            total_penalty += penalty
        
        # Average the penalties and apply violation density scaling
        avg_penalty = total_penalty / num_violations
        violation_ratio = num_violations / num_sensors
        
        # Linear combination of average penalty and violation density
        final_penalty = avg_penalty * (0.7 + 0.3 * violation_ratio)
        
        return max(-100.0, min(5.0, final_penalty))
    
    def _calculate_action_reward(self, action, margin_reward):
        """Calculate action efficiency reward"""
        # Convert margin_reward to safety_level [0, 1]
        safety_level = (margin_reward + 100) / 105.0
        safety_level = max(0.0, min(1.0, safety_level))
        
        action_magnitude = np.linalg.norm(action)
        
        if action_magnitude < self.LOW_VELOCITY:
            return -2.0
        elif action_magnitude > self.HIGH_VELOCITY:
            return 3.0
        return 1.0        
        
    def _calculate_step_reward(self, current_step: int, max_steps: int):
        """Calculate step efficiency reward"""
        progress_ratio = current_step / max_steps
        
        if progress_ratio < 0.5:
            return 0.0
        elif progress_ratio < 0.8:
            step_penalty_factor = (progress_ratio - 0.5) / 0.3
            return -3.0 * step_penalty_factor
        else:
            return -5.0
    
    def _calculate_fly_reward(self, obs):
        # R_fly: reward for flying towards destination and following predefined route
        distance_to_destination = np.linalg.norm(obs['position'] - self.goal_position)
        distance_to_route = self._calculate_distance_to_route(
            obs['position'], self.start_positon, self.goal_position)

        max_possible_distance = np.linalg.norm(self.goal_position - self.start_positon)
        progress = 1.0 - (distance_to_destination / max_possible_distance)
        route_adherence = 1.0 - min(1.0, distance_to_route / self.d_soft)
        
        R_fly = 10.0 * (0.7 * progress + 0.3 * route_adherence)

        return R_fly

    def _calculate_velocity_reward(self, action):
        # R_movement: reward/penalty for movement vector characteristics
        LOW_HORIZONTAL = 0.5
        HIGH_VERTICAL = 0.5

        
        horizontal_movement = np.sqrt(action[0]**2 + action[1]**2)
        vertical_movement = abs(action[2])

        if horizontal_movement > LOW_HORIZONTAL:
            if vertical_movement > HIGH_VERTICAL:
                vertical_penalty = -5.0 * (vertical_movement - HIGH_VERTICAL)
            else:
                vertical_penalty = 5.0 * (1.0 - vertical_movement/HIGH_VERTICAL)
            R_movement = vertical_penalty * (horizontal_movement / 1.0)
        else:
            R_movement = 0.0

            return R_movement
        
    def _adjust_weights(self, margin_reward):
        """Dynamically adjust weights based on safety level"""
        safety_level = (margin_reward + 100) / 105.0
        safety_level = max(0.0, min(1.0, safety_level))
        
        if safety_level > 0.8:
            return 1.0, 0.7, 0.3  # margin, action, step
        elif safety_level > 0.5:
            return 1.0, 0.3, 0.2
        else:
            return 1.0, 0.0, 0.1
        
    def _compute_reward(self, action, obs, curr_step, done:bool, completed:bool, arrive_max_steps:bool):
        # R_collision: penalty for collision
        R_collision = -100 if done and not completed and not arrive_max_steps else 0
        
        # Get the distance values ​​of all sensors
        sensor_distances = self._get_all_sensor_distances()        
        
        # Get fly reward
        R_fly = self._calculate_fly_reward(obs)
        
        # R_goal: reward for reaching the destination
        R_goal = 100 if done and completed else 0
        
        # R_margin: margin reward
        # R_margin = self._calculate_margin_reward_v1() # margin rewardV1
        R_margin = self._calculate_margin_reward_v2(sensor_distances)  # V2
        # R_margin = self._calculate_margin_reward_v3(sensor_distances)  # V3

        margin_weight, action_weight, step_weight = self._adjust_weights(R_margin)

        # Calculate individual rewards
        action_reward = self._calculate_action_reward(action, R_margin)
        # step_reward = self._calculate_step_reward(curr_step, self.max_steps)
        
        # calculate total reward
        r_t = (
            R_fly +
            margin_weight * R_margin +
            action_weight * action_reward +
            # step_weight * step_reward +
            R_goal +
            R_collision
        )
        # r_t = (
        #     R_fly +
        #     R_margin +
        #     R_goal +
        #     R_collision
        # )
        # print(f"total_reward: {r_t}")
        return r_t
    
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