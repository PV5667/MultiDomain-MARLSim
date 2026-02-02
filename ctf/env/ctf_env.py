"""Capture the Flag 2.5D Multi-Agent Environment"""

import numpy as np
from terrain import generate_heightmap, compute_slope
from swarm.swarm import Swarm

class CTFEnv:
    def __init__(self, height, width, n_ground_agents, n_air_agents, n_flags, seed_range):
        # environment will be represented by a numpy array
        self.height = height
        self.width = width
        self.n_ground_agents = n_ground_agents
        self.n_air_agents = n_air_agents
        self.n_flags = n_flags
        self.swarms = []
        self.seed_range = seed_range # range of seeds to sample
        self.curr_seed = None

        # initalize the terrain
        print("Generating heightmaps...")
        self.heightmaps = [generate_heightmap(self.height, self.width, seed) for seed in self.seed_range]
        self.curr_heightmap = None
        self.gradient_map = None # for initialization purposes

        self.max_slope = 0.2

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
        self.environment = np.zeros((height, width, self.channels))

    def _agent_pos(self, swarm_id):

        # generates list of random agent positions given heightmap and swarm id
        # agent_pos: (agent_type, x, y)
        agent_pos = []
        # masking available parts of terrain (20% on left for swarm1, 20% on right for swarm2)
        x_min, x_max = 0, self.width
        if swarm_id == "swarm1":
            x_max = int(0.2 * self.width)
        elif swarm_id == "swarm2":
            x_min = int(0.8 * self.width)

        available_terrain = np.zeros((self.height, self.width))
        available_terrain[:, x_min:x_max] = 1 # boolean mask

        # Air agent initialization isn't constrained by terrain, so we simply need to randomly choose positions for them.
        min_distance_radius = 5

        for _ in range(self.n_air_agents):
            # sample random position in xs, ys
            # append to agent_pos
            # set radius around agent to 0
            ys, xs = np.where(available_terrain == 1)
            if len(xs) == 0:
                raise RuntimeError("No valid terrain left to place air agents.")

            idx = np.random.randint(len(xs))
            x = xs[idx]
            y = ys[idx]

            agent_pos.append(("air", x, y))
            
            self.invalidate_radius(x, y, min_distance_radius, available_terrain)
        """
        We want ground agents to be initialized in navigable settings. To make things more plausible, we have slope-informed probability for movement action success (medium/small slopes have p=1.0).
        
        So here we try to initialize ground agents in small/medium slope areas. In the worst case, we initialize them in high-slope areas -- they always have a chance of making it out.

        Another constraint that we try to satisfy is distance between agents. Given recommended map size and n_agents, this is fairly doable.
        """
        # using self.gradient_map, mark any idx with slope > self.max_slope as unavailable
        high_slope_idxs = np.where(self.gradient_map > self.max_slope)
        available_terrain[high_slope_idxs] = 0

        for _ in range(self.n_ground_agents):
            ys, xs = np.where(available_terrain == 1)
            if len(xs) == 0:
                raise RuntimeError("No valid terrain left to place ground agents.")

            idx = np.random.randint(len(xs))
            x = xs[idx]
            y = ys[idx]

            agent_pos.append(("ground", x, y))

            self.invalidate_radius(x, y, min_distance_radius, available_terrain)
        return agent_pos

    def _flag_pos(self):
        """
        generates list of random valid flag positions given heightmap
        valid flag positions occur within the middle 40% of the map.
        (For now) 50% of flags planted in valleys and mountains, 50% in plains
        The idea is to add Line-of-Sight and Movement constraints to the capturing process.
        """
        flag_pos = []
        available_terrain = np.zeros((self.height, self.width))
        available_terrain[:, int(0.3 * self.width): int(0.7 * self.width)] = 1
        min_distance_radius = 50
        # basically just need to factor in height
        # merge top 10% and bottom 10% in height, randomly select n_flags // 2 from them
        bottom10_height = np.percentile(self.curr_heightmap, 10)
        top10_height = np.percentile(self.curr_heightmap, 90)
        middle20_height = np.percentile(self.curr_heightmap, [40, 60])
        
        # indices of mountains
        mount_ys, mount_xs = np.where((self.curr_heightmap >= top10_height))
        # indices of valleys
        valley_ys, valley_xs = np.where((self.curr_heightmap <= bottom10_height))
        # indices of plains
        plain_ys, plain_xs = np.where(
            (self.curr_heightmap >= middle20_height[0]) & (self.curr_heightmap <= middle20_height[1])
        )

        def pick_flag(cand_ys, cand_xs):
            valid = [
                (y, x) for y, x in zip(cand_ys, cand_xs) if available_terrain[y, x] == 1
            ]
            if not valid:
                return None
            y, x = valid[np.random.randint(len(valid))]
            flag_pos.append((x, y))
            self.invalidate_radius(x, y, min_distance_radius, available_terrain)
            return (x, y)
    
        for _ in range(self.n_flags // 4):
            if pick_flag(mount_ys, mount_xs) is None:
                raise RuntimeError("No valid terrain left in mountains/valleys.")
            
        for _ in range(self.n_flags // 4):
            if pick_flag(valley_ys, valley_xs) is None:
                raise RuntimeError("No valid terrain left in mountains/valleys.")
            
        # place remaining flags in plains
        for _ in range(self.n_flags - len(flag_pos)):
            if pick_flag(plain_ys, plain_xs) is None:
                raise RuntimeError("No valid terrain left in plains.")
        
        return flag_pos

    def invalidate_radius(self, cx, cy, radius, available_terrain):
            # invalidates radius given coords
            x0 = max(0, cx - radius)
            x1 = min(self.width, cx + radius + 1)
            y0 = max(0, cy - radius)
            y1 = min(self.height, cy + radius + 1)

            xs = np.arange(x0, x1)
            ys = np.arange(y0, y1)

            xx, yy = np.meshgrid(xs, ys)

            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            available_terrain[y0:y1, x0:x1][mask] = 0

    def init_env(self):
        # initializes two Swarm objects with positions and agent info
        # also initializes flag positions and creates required objects (TODO)
        # calls _agent_pos and _flag_pos
        # updates self.environment with info
        rand_seed_idx = np.random.choice(np.arange(len(self.seed_range)))
        self.curr_seed = self.seed_range[rand_seed_idx]
        self.curr_heightmap = self.heightmaps[rand_seed_idx]
        self.gradient_map = compute_slope(self.curr_heightmap, sigma=9)

        print("Getting Swarm 1 Agent Positions")
        swarm1_agent_pos = self._agent_pos("swarm1")
        print("Getting Swarm 2 Agent Positions")
        swarm2_agent_pos = self._agent_pos("swarm2")
        print("Getting Flag Positions")
        flag_pos = self._flag_pos()
        
        # update self.environment (ground-truth array)
        self.environment[:, :, 0] = self.curr_heightmap
        for pos in swarm1_agent_pos:
            agent_type, x, y = pos
            if agent_type == "ground":
                self.environment[y, x, 1] = 1 # ground agent of swarm1
            else:
                self.environment[y, x, 1] = 2 # air agent of swarm1
        
        for pos in swarm2_agent_pos:
            agent_type, x, y = pos
            if agent_type == "ground":
                self.environment[y, x, 1] = 3 # ground agent of swarm2
            else:
                self.environment[y, x, 1] = 4 # air agent of swarm2
        
        self.environment[:, :, 3] = np.ones((self.height, self.width)) * 2
        for pos in flag_pos:
            x, y = pos
            self.environment[y, x, 3] = 0 # neutral flag

        # initialize swarm objects
        swarm1 = Swarm(swarm1_agent_pos)
        swarm2 = Swarm(swarm2_agent_pos)
        self.swarms = [swarm1, swarm2]

        # TODO create Flag objects
        return

    def reset(self):
        np.random.seed()
        self.init_env()


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
        # uses self.environment, produces a 3D image of the environment
        # keep in mind that all air agents have fixed height (maybe in future will add z-axis control)
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]