from typing import Tuple, Dict

import numpy as np

from racecar_gym.agents.agent import ObservationDict, ActionDict, State, Agent


class RandomAgent(Agent):
    def __init__(self, config: Dict = {}):
        pass

    def get_action(self, observation: ObservationDict = None,
                   state: State = None) -> Tuple[ActionDict, State]:
        steering = -1 + 2 * np.random.rand()
        speed = -1 + 2 * np.random.rand()
        return {"steering": steering, "speed": speed}, None

    def reset(self):
        pass