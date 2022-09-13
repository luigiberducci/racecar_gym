from time import sleep

import numpy as np

from racecar_gym.agents.follow_the_gap import FollowTheGap
from racecar_gym import MultiAgentScenario
from racecar_gym.envs import gym_api

scenario = MultiAgentScenario.from_spec(
    path='../scenarios/custom.yml',
    rendering=True
)

env = gym_api.MultiAgentRaceEnv(scenario=scenario)

print(env.observation_space)
print(env.action_space)

ftg = FollowTheGap()

for episode in range(10):
    print(episode)
    done = False
    obs = env.reset(mode='grid')
    _ = ftg.reset({
        "gap_threshold": 1.5,
        "base_speed": 1.25 + np.random.rand() * 0.5,
        "variable_speed": 0.25 + np.random.rand() * 0.5,
    })

    while not done:
        agent_action, _ = ftg.get_action(observation=obs["A"])

        action = {"A": ftg.get_action(observation=obs["A"])[0],
                  "B": ftg.get_action(observation=obs["B"])[0]}

        obs, rewards, dones, states = env.step(action)
        print(states["A"]["progress"])
        done = any(dones.values())
        sleep(0.01)

env.close()
