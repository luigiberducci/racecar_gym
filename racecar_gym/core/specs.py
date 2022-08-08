from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from yamldataclassconfig.config import YamlDataClassConfig


@dataclass
class SimulationSpec(YamlDataClassConfig):
    time_step: float = 0.01
    rendering: bool = False
    implementation: str = None


@dataclass
class TaskSpec(YamlDataClassConfig):
    task_name: str = None
    params: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class VehicleSpec(YamlDataClassConfig):
    name: str = None
    sensors: List[str] = field(default_factory=lambda: [])
    actuators: List[str] = field(default_factory=lambda: ['steering', 'motor'])
    color: str = 'blue'  # either red, blue, green, magenta or random


@dataclass
class DomainRandomizationConfig:
    gravity: Tuple[float, float] = None
    steering_multiplier: Tuple[float, float] = None
    velocity_multiplier: Tuple[float, float] = None
    max_velocity: Tuple[float, float] = None
    motor_force: Tuple[float, float] = None
    sensor_velocity_noise: Tuple[float, float] = None


@dataclass
class WorldSpec(YamlDataClassConfig):
    name: str = None
    reverse: bool = False
    rendering: bool = False
    domain_randomization: DomainRandomizationConfig = None


@dataclass
class AgentSpec(YamlDataClassConfig):
    id: str
    vehicle: VehicleSpec = VehicleSpec()
    task: TaskSpec = TaskSpec()


@dataclass
class ScenarioSpec(YamlDataClassConfig):
    world: WorldSpec = None
    agents: List[AgentSpec] = None
