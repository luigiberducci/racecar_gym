from dataclasses import dataclass
from typing import Any, Tuple, List, Dict

import numpy as np

from racecar_gym.agents.agent import Agent, ObservationDict, ActionDict, State


class FollowTheGap(Agent):
    """
    Follow the Gap controller.

    It implements a simple reactive controller with the following control laws:
        Lateral control:
            steering(t) = steering gain * angle_{center-gap}
        Longitudinal control:
            speed(t) = base_speed + variable speed * (1 - |steering(t)|) / max_steering
    """
    @dataclass
    class Config:
        # scan configuration (used to process observations)
        scan_field: str = "lidar"   # field in observation corresponding to the laser scan
        scan_fov_degree: int = 270  # field-of-view of lidar sensor in degree
        # actuators limits
        max_steering: float = 0.42  # max steering (in absolute value, assuming symmetric steering)
        min_speed: float = 0.0  # min, max speed
        max_speed: float = 3.5
        # controller params
        gap_threshold: float = 3.0  # important: min threshold for gap detection (m)
        steering_gain: float = 1.0  # gain on steering to steer more or less aggressively towards the gap (scalar)
        base_speed: float = 1.0     # base speed in longitudinal control (ms)
        variable_speed: float = 1.0 # variable speed in longitudinal control (ms)

    def __init__(self, config: Dict = {}):
        self._c = FollowTheGap.Config(**config)

    def get_action(self, observation: ObservationDict, state: Any = None,
                   return_norm_actions: bool = True) -> ActionDict:
        original_scan = observation[self._c.scan_field]
        scan_proc = self.preprocess(original_scan)

        gaps = self.find_gaps(scan_proc)
        if len(gaps) > 0:
            gap = self.select_gap(gaps)
            steering = self.control_steering(gap, n_beams=len(scan_proc))
            speed = self.control_speed(steering)
        else:
            steering = 0.0
            speed = self._c.base_speed

        if return_norm_actions:
            steering = -1 + 2 * (steering + self._c.max_steering) / (2 * self._c.max_steering)
            speed = -1 + 2 * (speed - self._c.min_speed) / (self._c.max_speed - self._c.min_speed)
        return {"steering": steering, "speed": speed}, None

    @staticmethod
    def preprocess(scan: np.ndarray) -> np.ndarray:
        return scan

    def find_gaps(self, scan: np.ndarray) -> List[Tuple[int, int]]:
        gaps = []

        gap_init, gap_end = -1, -1
        for i, beam in enumerate(scan):
            if beam > self._c.gap_threshold:
                if gap_init < 0:
                    gap_init = i
            else:
                if gap_init >= 0 and gap_end < 0:
                    gap_end = i
                    gap = (gap_init, gap_end)
                    gaps.append(gap)
                    gap_init = gap_end = -1
        if gap_init >= 0 and gap_end < 0:
            gap_end = len(scan) - 1
            gap = (gap_init, gap_end)
            gaps.append(gap)

        return gaps

    @staticmethod
    def select_gap(gaps: List[Tuple[int, int]]) -> Tuple[int, int]:
        best_gap_id = np.argmax([gap_end - gap_init for gap_init, gap_end in gaps])
        return gaps[best_gap_id]

    def control_steering(self, gap: Tuple[int, int], n_beams: int) -> float:
        center_gap = gap[0] + (gap[1] - gap[0]) / 2
        steering = np.deg2rad(-self._c.scan_fov_degree / 2) + center_gap * np.deg2rad(self._c.scan_fov_degree / n_beams)
        steering = np.clip(self._c.steering_gain * steering, -self._c.max_steering, self._c.max_steering)
        return steering

    def control_speed(self, steering: float) -> float:
        speed = self._c.base_speed + (1 - abs(steering) / self._c.max_steering) * self._c.variable_speed
        speed = np.clip(speed, self._c.min_speed, self._c.max_speed)
        return speed

    def reset(self, config: Dict = None) -> State:
        # ftg is state-less -> do nothing a part from changing params
        if config is not None:
            self._c = FollowTheGap.Config(**config)
        pass
