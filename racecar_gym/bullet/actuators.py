from abc import ABC
from dataclasses import dataclass
from typing import Tuple, TypeVar, List

import gym
import numpy as np
import pybullet

from racecar_gym.core import actuators

T = TypeVar('T')


class BulletActuator(actuators.Actuator[T], ABC):
    def __init__(self, name: str):
        super().__init__(name)
        self._body_id = None
        self._joint_indices = []

    def reset(self, body_id: int, joint_indices: List[int] = None):
        self._body_id = body_id
        self._joint_indices = joint_indices

    @property
    def body_id(self) -> int:
        return self._body_id

    @property
    def joint_indices(self) -> List[int]:
        return self._joint_indices


class Motor(BulletActuator[Tuple[float, float]]):
    @dataclass
    class Config:
        velocity_multiplier: float
        max_velocity: float
        max_force: float

    def __init__(self, name: str, config: Config):
        super().__init__(name)
        self._config = config

    def control(self, command: Tuple[float, float]) -> None:
        velocity, force = command
        for index in self.joint_indices:
            pybullet.setJointMotorControl2(self.body_id,
                                           index, pybullet.VELOCITY_CONTROL,
                                           targetVelocity=velocity * self._config.velocity_multiplier,
                                           force=force)

    def space(self) -> gym.Space:
        return gym.spaces.Box(low=np.array([-self._config.max_velocity, 0.0]),
                              high=np.array([self._config.max_velocity, self._config.max_force]),
                              shape=(2,))


class SteeringWheel(BulletActuator[float]):
    @dataclass
    class Config:
        steering_multiplier: float
        max_steering_angle: float

    def __init__(self, name: str, config: Config):
        super().__init__(name)
        self._config = config

    def control(self, command: float) -> None:
        for joint in self.joint_indices:
            pybullet.setJointMotorControl2(self.body_id,
                                           joint,
                                           pybullet.POSITION_CONTROL,
                                           targetPosition=-command * self._config.steering_multiplier)

    def space(self) -> gym.Space:
        return gym.spaces.Box(low=-self._config.max_steering_angle,
                              high=self._config.max_steering_angle,
                              shape=(1,))