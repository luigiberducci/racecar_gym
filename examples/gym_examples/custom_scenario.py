from time import sleep

import numpy as np

from racecar_gym import MultiAgentScenario
from racecar_gym.agents import FollowTheGap
from racecar_gym.envs import gym_api

scenario = MultiAgentScenario.from_spec(
    path='../scenarios/custom.yml',
    rendering=True
)

env = gym_api.MultiAgentRaceEnv(scenario=scenario)

print(env.observation_space)
print(env.action_space)

done = False
obs = env.reset(mode='random_ball')
ftg = FollowTheGap()
_ = ftg.reset()

maxv = 0
while not done:
    agent_action, _ = ftg.get_action(observation=obs["A"])
    action = env.action_space.sample()
    action["A"] = agent_action

    obs, rewards, dones, states = env.step(action)

    maxv = max(maxv, obs["A"]["velocity"][0])
    print(maxv)

    done = any(dones.values())
    sleep(0.01)

env.close()
