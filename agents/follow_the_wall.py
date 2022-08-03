from dataclasses import dataclass
from typing import Tuple

import numpy as np

from agents.agent import Agent, State, ObservationDict, ActionDict


class PID:
    @dataclass
    class PIDConfig:
        kp: float = 1.0
        ki: float = 0.0
        kd: float = 0.0

    def __init__(self, config: PIDConfig, init_control=0):
        self._c = config
        self.init_control = init_control
        self.prev_error = 0
        self.integral = 0

    def control(self, target, measurement):
        error = target - measurement
        self.integral += error
        output_control = self.init_control + \
                         self._c.kp * error + \
                         self._c.ki * self.integral + \
                         self._c.kd * (error - self.prev_error)
        self.prev_error = error
        return output_control


class FollowTheWall(Agent):
    @dataclass
    class Config:
        # scan configuration
        scan_field: str = "lidar"  # field in observation corresponding to the lidar scan
        scan_fov_degree: int = 270  # field-of-view of lidar sensor in degree
        scan_size: int = 1080  # field-of-view of lidar sensor in degree
        # actuators limits
        max_steering: float = 0.42  # max steering (in absolute value, assuming symmetric steering)
        min_speed: float = 0.0  # min speed (m/s)
        max_speed: float = 3.5  # max speed (m/s)

    def __init__(self,
                 target_distance_left: float = 0.5,
                 target_speed: float = 1.0,
                 reference_angle: float = 60):
        # config controllers
        self._c = self.Config()
        self._steer_ctrl = PID(
            PID.PIDConfig(0.8 * 16.0, 0.0, 0.1 * 0.4 * 16.0),   # tuned with ZN method (Ku=16, Tu=0.5)
            init_control=0.0)

        # target variables
        self._target_distance_left = target_distance_left
        self._target_speed = target_speed

        # internal params
        self._alpha = reference_angle  # degrees
        self._n_beams_for_dist = 3  # nr averaged beams to estimate wall distance
        self._cos_alpha = np.cos(self._alpha * np.pi / 180)
        self._sin_alpha = np.sin(self._alpha * np.pi / 180)
        # reference rays
        self._left_index = self._get_beam_id_from_angle(angle_deg=-90)
        self._right_index = self._get_beam_id_from_angle(angle_deg=90)
        self._alpha_index = self._get_beam_id_from_angle(angle_deg=-self._alpha)  # second ray: alpha angle

    def _get_beam_id_from_angle(self, angle_deg: int):
        assert - self._c.scan_fov_degree / 2 <= angle_deg <= self._c.scan_fov_degree / 2, f"invalid angle {angle_deg}"
        angles = np.linspace(0, self._c.scan_fov_degree, self._c.scan_size)
        differences = (angles - (self._c.scan_fov_degree / 2 + angle_deg))**2
        return np.argmin(differences)

    def get_action(self, observation: ObservationDict,
                   state: State = None,
                   return_norm_actions: bool = True) -> Tuple[ActionDict, State]:
        assert "velocity" in observation and "lidar" in observation, f"unvalid observation, keys: {observation.keys()}"
        # process observation
        current_speed = observation["velocity"][0]
        scan = observation["lidar"]

        # compute speed, steering targets
        predicted_distance = self._compute_distance(scan)
        steering = self._steer_ctrl.control(self._target_distance_left, predicted_distance)
        speed = self._target_speed

        # convert to normalized scale
        if return_norm_actions:
            steering = self._linear_scaling(steering, [-self._c.max_steering, self._c.max_steering], [-1, +1],
                                            clip=True)
            speed = self._linear_scaling(speed, [self._c.min_speed, self._c.max_speed], [-1, +1], clip=True)
        else:
            # simply clip to ensure within bounds
            steering = np.clip(steering, -self._c.max_steering, self._c.max_steering)
            speed = np.clip(speed, self._c.min_speed, self._c.max_speed)
        return {"steering": steering, "speed": speed}, None

    def _compute_distance(self, scan: np.array):
        # https://f1tenth-coursekit.readthedocs.io/en/latest/assignments/labs/lab3.html
        left_dist = self._average_distance_around_beam(scan, self._left_index, n=self._n_beams_for_dist)
        alpha_dist = self._average_distance_around_beam(scan, self._alpha_index, n=self._n_beams_for_dist)
        theta = np.arctan((alpha_dist * self._cos_alpha - left_dist) / (alpha_dist * self._sin_alpha))  # in radians
        dist_to_wall = left_dist * np.cos(theta)
        naive_predicted_distance = 0.5 * np.sin(theta)  # this might benefit from some multiplicative scalar
        return dist_to_wall + naive_predicted_distance

    @staticmethod
    def _average_distance_around_beam(data, index, n=3):
        """
        Reads lidar data from data[index] and returns an average of
        n values of data, centered around index
        """
        return np.mean(data[index - n // 2:index + n // 2 + 1])

    @staticmethod
    def _linear_scaling(v, from_range, to_range, clip=True):
        if clip:
            v = np.clip(v, from_range[0], from_range[1], dtype=np.float32)  # clip it
        new_v = (v - from_range[0]) / (from_range[1] - from_range[0])  # norm in 0,1
        new_v = to_range[0] + (to_range[1] - to_range[0]) * new_v  # map it to target range
        return new_v

    def reset(self) -> State:
        # stateless controller
        pass
