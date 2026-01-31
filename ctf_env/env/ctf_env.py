"""Capture the Flag 2.5D Multi-Agent Environment"""

import numpy as np
from terrain import generate_heightmap


class CTFEnvironment:
    def __init__(self, height, width, n_agents, seed):
        # environment will be represented by a numpy array
        self.height = height
        self.width = width
        self.n_agents = n_agents # number of agents per swarm
        self.seed = seed
        self.swarms = []

        # initalize the terrain
        self.heightmaps = [generate_heightmap(self.height, self.width, seed) for seed in range(50)]
        self.curr_heightmap = None

        # ground-truth numpy array containing all relevant info in environment
        # localized patches directly passed to agents as part of observation
        self.channels = 4
        """
        0: heightmap
        1: agents: (1, 2), (3, 4), (5, 6) --> (Ground1, Air1), (Ground2, Air2), (cUAS1, cUAS2)
        2: health of agents
        3: flag dispositions: range between [-1, 1]. 
            - -1 --> Enemy Capture, 1 --> Friendly Capture
            - For ground truth, -1 is T1, 1 is T2, but convert when serving.
        (For now, more might be added)
        """
        self.environment = np.zeros(height, width, self.channels)

    def _agent_pos(self, n_agents, swarm_id):
        # generates list of random agent positions given heightmap and swarm id
        pass

    def _flag_pos(self, n_flags):
        # generates list of random valid flag positions given heightmap
        pass

    def init_env():
        # initializes two Swarm objects with positions and agent info
        # also initializes flag positions and creates required objects (TODO)
        # calls _agent_pos and _flag_pos
        # updates self.environment with info
        pass

    def reset(self):
        self.curr_heightmap = np.random.choice(self.heightmaps)
        self.init_swarms()

    def update(self, actions):
        # handles dynamics given actions (flag capture, movement, engagements, health updates)
        # actions is a dictionary with keys "swarm1" and "swarm2"
        # updates self.environment
        # FOR NOW: just need to implement motion
        pass

    def step(self):
        # inference + update
        # call swarm.step() --> inference with action list as output
        # swarm.step() also handles SMART + inter-agent comms
        # call self.update(actions)
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]