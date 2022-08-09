import dataclasses
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np

from racecar_gym.agents.agent import Agent, State, ObservationDict, ActionDict


class PID:
    @dataclass
    class PIDConfig:
        kp: float = 1.0
        ki: float = 0.0
        kd: float = 0.0

    def __init__(self, config: Dict = {}):
        self._c = self.PIDConfig(**config)
        self.prev_error = 0
        self.integral = 0

    def control(self, target, measurement):
        error = target - measurement
        self.integral += error
        output_control = self._c.kp * error + self._c.ki * self.integral + self._c.kd * (error - self.prev_error)
        self.prev_error = error
        return output_control


class FollowTheWall(Agent):
    """
    Follow The Wall Controller which keep a fixed distance to the Left wall.
    The velocity profile depends on the steering angle.
    It works decently in smooth tracks (no 90 degree or hairpin curves).

    Note: PID parameters *must* be tuned according to the track, target distance and target speed.
    """

    @dataclass
    class Config:
        # scan configuration (used to process observation and compute angles)
        scan_field: str = "lidar"  # field in observation corresponding to the lidar scan
        scan_fov_degree: int = 270  # field-of-view of lidar sensor in degree
        scan_size: int = 1080  # field-of-view of lidar sensor in degree
        # actuators limits (used to normalize actions)
        max_steering: float = 0.42  # max steering (in absolute value, assuming symmetric steering)
        min_speed: float = 0.0  # min speed (m/s)
        max_speed: float = 3.5  # max speed (m/s)
        # controller params
        target_distance_left: float = 0.5  # distance to the left wall (m)
        max_deviation: float = 100  # max deviation of target distance from current distance (m), avoid u-turns
        alpha: float = 60  # reference angle for the 2nd beam
        n_beams_for_dist: int = 3  # used to estimate distance by averaging this nr of beams
        pid_config: Dict[str, float] = dataclasses.field(default_factory=lambda : {"kp": 1.0, "ki": 0.0, "kd": 0.0})  # important: pid for lateral control
        base_speed: float = 1.0  # base speed in longitudinal control (ms)
        variable_speed: float = 1.0  # variable speed in longitudinal control (ms)

    def __init__(self, config: Dict = {}):
        # config controllers
        self._c = self.Config(**config)
        self._steer_ctrl = PID(config=self._c.pid_config)

        # internal params
        self._n_beams_for_dist = 3  # nr averaged beams to estimate wall distance
        self._cos_alpha = np.cos(self._c.alpha * np.pi / 180)
        self._sin_alpha = np.sin(self._c.alpha * np.pi / 180)
        # reference rays
        self._left_index = self._get_beam_id_from_angle(angle_deg=-90)
        self._right_index = self._get_beam_id_from_angle(angle_deg=90)
        self._alpha_index = self._get_beam_id_from_angle(angle_deg=-self._c.alpha)  # second ray: alpha angle

    def _get_beam_id_from_angle(self, angle_deg: int):
        assert - self._c.scan_fov_degree / 2 <= angle_deg <= self._c.scan_fov_degree / 2, f"invalid angle {angle_deg}"
        angles = np.linspace(0, self._c.scan_fov_degree, self._c.scan_size)
        differences = (angles - (self._c.scan_fov_degree / 2 + angle_deg)) ** 2
        return np.argmin(differences)

    def get_action(self, observation: ObservationDict,
                   state: State = None,
                   return_norm_actions: bool = True) -> Tuple[ActionDict, State]:
        assert self._c.scan_field in observation, f"unvalid observation, keys: {observation.keys()}"
        # process observation
        scan = observation[self._c.scan_field]

        # compute speed, steering targets
        current_distance, predicted_distance = self._compute_distances(scan)

        # target correction
        # problem: when starting too far from the target distance, the agent causes u-shape turns
        # solution: control the target distance incrementally to avoid too aggressive steering
        distance_error = np.clip(self._c.target_distance_left - current_distance,
                                 -self._c.max_deviation, self._c.max_deviation)
        target_distance = current_distance + distance_error

        unclip_steering = self._steer_ctrl.control(target_distance, predicted_distance)
        steering = np.clip(unclip_steering, -self._c.max_steering, self._c.max_steering)
        speed = self.control_speed(steering=steering)

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

    def _compute_distances(self, scan: np.array) -> Tuple[float, float]:
        # https://f1tenth-coursekit.readthedocs.io/en/latest/assignments/labs/lab3.html
        left_dist = self._average_distance_around_beam(scan, self._left_index, n=self._n_beams_for_dist)
        alpha_dist = self._average_distance_around_beam(scan, self._alpha_index, n=self._n_beams_for_dist)
        theta = np.arctan((alpha_dist * self._cos_alpha - left_dist) / (alpha_dist * self._sin_alpha))  # in radians
        dist_to_wall = left_dist * np.cos(theta)
        naive_predicted_distance = 0.5 * np.sin(theta)  # this might benefit from some multiplicative scalar
        return left_dist, dist_to_wall + naive_predicted_distance

    @staticmethod
    def _average_distance_around_beam(data, index, n=3):
        """
        Reads lidar data from data[index] and returns an average of
        n values of data, centered around index
        """
        return np.mean(data[index - n // 2:index + n // 2 + 1])

    def control_speed(self, steering: float) -> float:
        speed = self._c.base_speed + (1 - abs(steering) / self._c.max_steering) * self._c.variable_speed
        speed = np.clip(speed, self._c.min_speed, self._c.max_speed)
        return speed

    @staticmethod
    def _linear_scaling(v, from_range, to_range, clip=True):
        if clip:
            v = np.clip(v, from_range[0], from_range[1], dtype=np.float32)  # clip it
        new_v = (v - from_range[0]) / (from_range[1] - from_range[0])  # norm in 0,1
        new_v = to_range[0] + (to_range[1] - to_range[0]) * new_v  # map it to target range
        return new_v

    def reset(self, config: Dict = None) -> State:
        # ftw is state-less -> do nothing a part from changing params
        if config is not None:
            self._c = FollowTheWall.Config(**config)
        pass
