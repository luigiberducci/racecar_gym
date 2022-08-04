from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

ObservationDict = Dict[str, Any]
ActionDict = Dict[str, float]
State = Any


class Agent(ABC):

    @abstractmethod
    def __init__(self, config: Dict = {}):
        pass

    @abstractmethod
    def get_action(self, observation: ObservationDict, state: State) -> Tuple[ActionDict, State]:
        pass

    @abstractmethod
    def reset(self) -> State:
        pass
