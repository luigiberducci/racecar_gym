from typing import Optional

import numpy as np
import pybullet
from nptyping import NDArray, Shape, Float

from racecar_gym.core import Agent

def get_velocity(id: int) -> NDArray[Shape["6"], Float]:
    linear, angular = pybullet.getBaseVelocity(id)
    position, orientation = pybullet.getBasePositionAndOrientation(id)
    rotation = pybullet.getMatrixFromQuaternion(orientation)
    rotation = np.reshape(rotation, (-1, 3)).transpose()
    linear = rotation.dot(linear)
    angular = rotation.dot(angular)
    return np.append(linear, angular)


def get_pose(id: int) -> Optional[NDArray[Shape["6"], Float]]:
    position, orientation = pybullet.getBasePositionAndOrientation(id)
    if any(np.isnan(position)) or any(np.isnan(orientation)):
        return None
    orientation = pybullet.getEulerFromQuaternion(orientation)
    pose = np.append(position, orientation)
    return pose

def birds_eye(agent: Agent, width=640, height=480) -> np.ndarray:
    #width, height = 2*640, 2*480           # for high-res frames used in fig.1 submission
    position, _ = pybullet.getBasePositionAndOrientation(agent.vehicle_id)
    position = np.array([position[0], position[1], 3.0])
    #position = np.array([9.0, 3.5, 3.5])   # for centering view on top of 90-deg turn of treitlstrasse
    view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=position,
        distance=3.0,
        yaw=0,
        pitch=-90,
        roll=0,
        upAxisIndex=2
    )
    proj_matrix = pybullet.computeProjectionMatrixFOV(
        fov=90,
        aspect=float(width) / height,
        nearVal=0.01,
        farVal=100.0
    )
    _, _, rgb_image, _, _ = pybullet.getCameraImage(
        width=width,
        height=height,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix)

    rgb_array = np.reshape(rgb_image, (height, width, -1))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

def follow_agent(agent: Agent, width=640, height=480) -> np.ndarray:
    position, orientation = pybullet.getBasePositionAndOrientation(agent.vehicle_id)
    _, _, yaw = pybullet.getEulerFromQuaternion(orientation)
    orientation = pybullet.getQuaternionFromEuler((0, 0, yaw))
    rot_matrix = pybullet.getMatrixFromQuaternion(orientation)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    camera_position = position + rot_matrix.dot([-0.8, 0, 0.3])
    up_vector = rot_matrix.dot([0, 0, 1])
    target = position
    view_matrix = pybullet.computeViewMatrix(camera_position, target, up_vector)
    proj_matrix = pybullet.computeProjectionMatrixFOV(
        fov=60,
        aspect=float(width) / height,
        nearVal=0.01,
        farVal=10.0
    )

    _, _, rgb_image, _, _ = pybullet.getCameraImage(
        width=width,
        height=height,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix)

    rgb_array = np.reshape(rgb_image, (height, width, -1))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array
