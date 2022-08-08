import math
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import gym
import numpy as np
import pybullet as p
from gym import logger

from racecar_gym.bullet import util
from racecar_gym.bullet.configs import MapConfig
from racecar_gym.bullet.positioning import AutomaticGridStrategy, RandomPositioningStrategy, \
    RandomPositioningWithinBallStrategy
from racecar_gym.core import world
from racecar_gym.core.agent import Agent
from racecar_gym.core.definitions import Pose
from racecar_gym.core.gridmaps import GridMap
from racecar_gym.core.specs import DomainRandomizationConfig


class World(world.World):
    FLOOR_ID = 0
    WALLS_ID = 1
    FINISH_ID = 2

    @dataclass
    class Config:
        name: str
        sdf: str
        map_config: MapConfig
        rendering: bool
        time_step: float
        gravity: float
        reverse: bool

    def __init__(self, config: Config, agents: List[Agent], dr_config: DomainRandomizationConfig = None):
        self._config = config
        self._dr = dr_config
        self._map_id = None
        self._time = 0.0
        self._agents = agents
        self._state = dict([(a.id, {}) for a in agents])
        self._objects = {}
        self._starting_grid = np.load(config.map_config.starting_grid)['data']
        self._maps = dict([
            (name, GridMap(
                grid_map=np.load(config.map_config.maps)[data],
                origin=self._config.map_config.origin,
                resolution=self._config.map_config.resolution
            ))
            for name, data
            in [
                ('progress', 'norm_distance_from_start'),
                ('obstacle', 'norm_distance_to_obstacle'),
                ('occupancy', 'drivable_area')
            ]
        ])
        if self._config.reverse:
            progress_map = self._maps["progress"].map
            reversed_progress = np.where(self._maps["occupancy"].map, 1 - progress_map, progress_map)
            self._maps["progress"] = GridMap(
                grid_map=reversed_progress,
                origin=self._config.map_config.origin,
                resolution=self._config.map_config.resolution
            )

        self._state['maps'] = self._maps
        self._tmp_occupancy_map = None  # used for `random_ball` sampling
        self._progress_center = None  # used for `random_ball` sampling
        self._trajectory = []

    def init(self) -> None:
        if self._config.rendering:
            id = -1  # p.connect(p.SHARED_MEMORY)
            if id < 0:
                p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self._load_scene(self._config.sdf)
        p.setTimeStep(self._config.time_step)
        p.setGravity(0, 0, self._config.gravity)

    def reset(self):
        p.setTimeStep(self._config.time_step)
        if not self.randomize_domain():
            p.setGravity(0, 0, self._config.gravity)
        p.stepSimulation()
        self._time = 0.0
        self._state = dict([(a.id, {}) for a in self._agents])

    def randomize_domain(self) -> bool:
        # if no dr config defined, return false
        if self._dr is None:
            return False
        # sample new parameters
        for parameter, domain in zip(
                ["gravity", "velocity_multiplier", "max_velocity", "motor_force", "steering_multiplier", "sensor_velocity_noise"],
                [self._dr.gravity, self._dr.velocity_multiplier, self._dr.max_velocity,
                 self._dr.motor_force, self._dr.steering_multiplier, self._dr.sensor_velocity_noise]):
            # sample value
            if domain is None:
                continue
            value = self._sample_domain(domain)
            # set value
            if parameter == "gravity":
                p.setGravity(0, 0, value)
            else:
                for agent in self._agents:
                    steering = "steering"
                    motor = "motor" if "motor" in agent.vehicle.actuators else "speed"
                    if parameter == "velocity_multiplier":
                        agent.vehicle.actuators[motor].set_velocity_multiplier(value)
                    elif parameter == "max_velocity":
                        agent.vehicle.actuators[motor].set_max_velocity(value)
                    elif parameter == "motor_force":
                        agent.vehicle.actuators[motor].set_motor_force(value)
                    elif parameter == "steering_multiplier":
                        agent.vehicle.actuators[steering].set_steering_multiplier(value)
                    elif parameter == "sensor_velocity_noise":
                        velocity_sensors = [sensor._sensor for sensor in agent.vehicle.sensors if sensor.type == "velocity"]
                        for sensor in velocity_sensors:
                            sensor.set_gaussian_noise(value)
        return True

    @staticmethod
    def _sample_domain(domain: Tuple[float, float]):
        return domain[0] + np.random.rand() * (domain[1] - domain[0])

    def _load_scene(self, sdf_file: str):
        ids = p.loadSDF(sdf_file)
        objects = dict([(p.getBodyInfo(i)[1].decode('ascii'), i) for i in ids])
        self._objects = objects

    def get_starting_position(self, agent: Agent, mode: str) -> Pose:
        start_index = list(map(lambda agent: agent.id, self._agents)).index(agent.id)
        if mode == 'grid':
            strategy = AutomaticGridStrategy(obstacle_map=self._maps['obstacle'],
                                             number_of_agents=len(self._agents),
                                             reverse=self._config.reverse)
        elif mode == 'random':
            strategy = RandomPositioningStrategy(progress_map=self._maps['progress'],
                                                 obstacle_map=self._maps['obstacle'],
                                                 alternate_direction=False)
        elif mode == 'random_bidirectional':
            strategy = RandomPositioningStrategy(progress_map=self._maps['progress'],
                                                 obstacle_map=self._maps['obstacle'],
                                                 alternate_direction=True)
        elif mode == 'random_ball':
            progress_radius = 0.10
            min_distance_to_wall = 0.5
            progress_map = self._maps['progress'].map
            obstacle_map = self._maps['obstacle'].map
            if start_index == 0:  # on first agent, compute a fixed interval for sampling and copy occupancy map
                progresses = progress_map[
                    obstacle_map > min_distance_to_wall]  # center has enough distance from the walls
                progresses = progresses[
                    (progresses > progress_radius) & (progresses < (1 - progress_radius))]  # center+-radius in [0,1]
                self._progress_center = np.random.choice(progresses)
                self._tmp_occupancy_map = self._maps['occupancy'].map.copy()
            strategy = RandomPositioningWithinBallStrategy(progress_map=self._maps['progress'],
                                                           obstacle_map=self._maps['obstacle'],
                                                           drivable_map=self._tmp_occupancy_map,
                                                           progress_center=self._progress_center,
                                                           progress_radius=progress_radius,
                                                           min_distance_to_obstacle=min_distance_to_wall)
        else:
            raise NotImplementedError(mode)
        position, orientation = strategy.get_pose(agent_index=start_index)
        if mode == 'random_ball':  # mark surrounding pixels as occupied
            px, py = self._maps['obstacle'].to_pixel(position)
            neigh_sz = int(1.0 / self._maps['obstacle'].resolution)  # mark 1 meter around the car
            self._tmp_occupancy_map[px - neigh_sz:px + neigh_sz, py - neigh_sz:py + neigh_sz] = False
        return position, orientation

    def update(self):
        p.stepSimulation()
        self._time += self._config.time_step

    def state(self) -> Dict[str, Any]:
        for agent in self._agents:
            self._update_race_info(agent=agent)

        self._update_ranks()

        return self._state

    def space(self) -> gym.Space:
        return gym.spaces.Dict({
            'time': gym.spaces.Box(low=0, high=math.inf, shape=(1,))
        })

    def _update_race_info(self, agent):
        contact_points = set([c[2] for c in p.getContactPoints(agent.vehicle_id)])
        progress_map = self._maps['progress']
        obstacle_map = self._maps['obstacle']
        pose = util.get_pose(id=agent.vehicle_id)
        if pose is None:
            logger.warn('Could not obtain pose.')
            self._state[agent.id]['pose'] = np.append((0, 0, 0), (0, 0, 0))
        else:
            self._state[agent.id]['pose'] = pose
        collision_with_wall = False
        opponent_collisions = []
        opponents = dict([(a.vehicle_id, a.id) for a in self._agents])
        for contact in contact_points:
            if self._objects['walls'] == contact:
                collision_with_wall = True
            elif contact in opponents:
                opponent_collisions.append(opponents[contact])

        self._state[agent.id]['wall_collision'] = collision_with_wall
        self._state[agent.id]['opponent_collisions'] = opponent_collisions
        velocity = util.get_velocity(id=agent.vehicle_id)

        if 'velocity' in self._state[agent.id]:
            previous_velocity = self._state[agent.id]['velocity']
            self._state[agent.id]['acceleration'] = (velocity - previous_velocity) / self._config.time_step
        else:
            self._state[agent.id]['acceleration'] = velocity / self._config.time_step

        pose = self._state[agent.id]['pose']
        progress = progress_map.get_value(position=(pose[0], pose[1], 0))
        dist_obstacle = obstacle_map.get_value(position=(pose[0], pose[1], 0))

        self._state[agent.id]['velocity'] = velocity
        self._state[agent.id]['progress'] = progress
        self._state[agent.id]['obstacle'] = dist_obstacle
        self._state[agent.id]['time'] = self._time

        progress = self._state[agent.id]['progress']
        # bug: checking checkpoints and lap when crossing exactly 1.0 may skip the lap update
        # becase the progress map jumps from 0.99 to 0.0.
        # bugfix: compute checkpoint considering a max progress of 0.99
        checkpoints = 0.99 / float(self._config.map_config.checkpoints)     # use 0.98 instead of 1.0 to avoid skip lap
        checkpoint = int(progress / checkpoints)

        if 'checkpoint' in self._state[agent.id]:
            last_checkpoint = self._state[agent.id]['checkpoint']
            if last_checkpoint + 1 == checkpoint:
                self._state[agent.id]['checkpoint'] = checkpoint
                self._state[agent.id]['wrong_way'] = False
            elif last_checkpoint - 1 == checkpoint:
                self._state[agent.id]['wrong_way'] = True
            elif last_checkpoint == self._config.map_config.checkpoints and checkpoint == 0:
                self._state[agent.id]['lap'] += 1
                self._state[agent.id]['checkpoint'] = checkpoint
                self._state[agent.id]['wrong_way'] = False
            elif last_checkpoint == 0 and checkpoint == self._config.map_config.checkpoints:
                self._state[agent.id]['wrong_way'] = True
        else:
            self._state[agent.id]['checkpoint'] = checkpoint
            self._state[agent.id]['lap'] = 1
            self._state[agent.id]['wrong_way'] = False

    def _update_ranks(self):

        agents = [
            (agent_id, self._state[agent_id]['lap'], self._state[agent_id]['progress'])
            for agent_id
            in map(lambda a: a.id, self._agents)
        ]

        ranked = [item[0] for item in sorted(agents, key=lambda item: (item[1], item[2]), reverse=True)]

        for agent in self._agents:
            rank = ranked.index(agent.id) + 1
            self._state[agent.id]['rank'] = rank

    def render(self, agent_id: str, mode: str, width=640, height=480) -> np.ndarray:
        agent = list(filter(lambda a: a.id == agent_id, self._agents))
        assert len(agent) == 1
        agent = agent[0]
        if mode == 'follow':
            return util.follow_agent(agent=agent, width=width, height=height)
        elif mode == 'birds_eye':
            return util.birds_eye(agent=agent, width=width, height=height)

    def seed(self, seed: int = None):
        if self is None:
            seed = 0
        np.random.seed(seed)
        random.seed(seed)
