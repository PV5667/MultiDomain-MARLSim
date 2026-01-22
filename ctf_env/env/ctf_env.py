"""Capture the Flag 2.5D Multi-Agent Environment"""

from pettingzoo import ParallelEnv


class CustomEnvironment(ParallelEnv):
    def __init__(self, height, width, agents, ):
        # attributes for the environment.
        self.height = height
        self.width = width


    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]