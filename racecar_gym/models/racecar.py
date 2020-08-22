from typing import Dict, Tuple, List, Any, Optional, Set

import gym
import numpy as np
from pybullet_utils.bullet_client import BulletClient

from racecar_gym.definitions import Pose
from racecar_gym.models import Map
from racecar_gym.models.configs import VehicleConfig, VehicleJointConfig
from racecar_gym.models.lidar import Lidar


class RaceCar:

    def __init__(self, client: BulletClient, map: Map, config: VehicleConfig):
        self._client = client
        self._config = config
        self._id = None
        self._joint_dict = {}
        self._lidar = None
        self._on_finish = False
        self._lap = 0
        self._map = map

        self.action_space = gym.spaces.Box(
            low=np.array([-config.max_speed, -config.max_steering_angle, 0]),
            high=np.array([config.max_speed, config.max_steering_angle, config.max_force]),
        )

    @property
    def id(self) -> int:
        """
        Get the id of the object in the current simulation.
        Returns:
            id
        """
        return self._id

    @property
    def config(self) -> VehicleConfig:
        return self._config

    def step(self, velocity: float, steering_angle: float, force: float) -> None:
        joints = self._config.joints
        motorized = [self._joint_dict[joint] for joint in joints.motorized_joints]
        steering = [self._joint_dict[joint] for joint in joints.steering_joints]

        for wheel in motorized:
            self._client.setJointMotorControl2(self._id, wheel, self._client.VELOCITY_CONTROL,
                                               targetVelocity=velocity * self._config.speed_multiplier, force=force)

        for steer in steering:
            self._client.setJointMotorControl2(self._id, steer, self._client.POSITION_CONTROL,
                                               targetPosition=-steering_angle * self._config.steering_multiplier)

        self._update_lap()

    def observe(self, sensors: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        observations = {}
        info = {}
        for sensor in sensors:
            if sensor == 'odometry':
                observations['pose'], observations['velocity'] = self._odometry()
            if sensor == 'lidar':
                observations[sensor] = self._lidar_scan()
        observations['lap'] = self._lap
        info['collisions'] = self._check_collisions()
        observations['collision'] = len(info['collisions']) > 0
        return observations, info

    def _odometry(self) -> Tuple[np.ndarray, np.ndarray]:
        position, orientation = self._client.getBasePositionAndOrientation(self._id)
        euler = self._client.getEulerFromQuaternion(orientation)
        pose = np.append(position, euler)
        velocities = self._client.getBaseVelocity(self._id)
        velocities = np.array(velocities[0] + velocities[1])
        return pose, velocities

    def _lidar_scan(self) -> np.ndarray:
        return self._lidar.scan()

    def reset(self, pose: Pose):
        self._id = self._load_model(self._config.urdf_file, pose)
        self._joint_dict = self._load_joint_indices(self._config.joints)
        self._setup_constraints()
        if not self._lidar:
            self._lidar = Lidar(client=self._client,
                                id=self._joint_dict[self._config.joints.lidar_joint[0]],
                                car_id=self._id,
                                config=self._config.sensors.lidar)
        self._lidar.reset()

    def _load_joint_indices(self, config: VehicleJointConfig) -> Dict[str, int]:
        available_joints = config.motorized_joints \
                           + config.steering_joints \
                           + config.lidar_joint \
                           + config.camera_joint

        joint_dict = {}

        for joint_index in range(self._client.getNumJoints(self._id)):
            joint_name = self._client.getJointInfo(self._id, joint_index)[1].decode('UTF-8')
            if joint_name in available_joints:
                joint_dict[joint_name] = joint_index

        return joint_dict

    def _load_model(self, model: str, initial_pose: Pose) -> int:
        position, orientation = initial_pose
        orientation = self._client.getQuaternionFromEuler(orientation)
        id = self._client.loadURDF(model, position, orientation)
        return id

    def _setup_constraints(self):

        for wheel in range(self._client.getNumJoints(self._id)):
            self._client.setJointMotorControl2(self._id, wheel, self._client.VELOCITY_CONTROL, targetVelocity=0,
                                               force=0)
            self._client.getJointInfo(self._id, wheel)

        # self._client.setJointMotorControl2(self._id,10,self._client.VELOCITY_CONTROL,targetVelocity=1,force=10)
        c = self._client.createConstraint(self._id, 9, self._id, 11, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = self._client.createConstraint(self._id, 10, self._id, 13, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._client.createConstraint(self._id, 9, self._id, 13, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._client.createConstraint(self._id, 16, self._id, 18, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = self._client.createConstraint(self._id, 16, self._id, 19, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._client.createConstraint(self._id, 17, self._id, 19, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._client.createConstraint(self._id, 1, self._id, 18, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
        c = self._client.createConstraint(self._id, 3, self._id, 19, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

    def _update_lap(self):
        closest_points = self._client.getClosestPoints(self._id, self._map.finish_id, 0.05)
        if len(closest_points) > 0:
            if not self._on_finish:
                self._on_finish = True
                self._lap += 1
        else:
            if self._on_finish:
                self._on_finish = False

    def _check_collisions(self) -> Set[int]:
        collisions = set([c[2] for c in self._client.getContactPoints(self._id)])
        collisions_without_floor = collisions - {self._map.floor_id, self._map.finish_id}
        return collisions_without_floor