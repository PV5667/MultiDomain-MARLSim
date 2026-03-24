"""Capture the Flag 2.5D Multi-Agent Environment"""

from tqdm import tqdm
import numpy as np
from env.terrain import generate_heightmap, compute_slope
from swarm.actions import MoveAction, EngageAction, DeployAction
from swarm.core import *
from env.feedback_message import FeedbackMessage
from constants import settings
from collections import defaultdict
from env.flag import Flag

class CTFEnv:
    def __init__(self, height, width, n_ground_agents, n_air_agents, n_flags, seed_range):
        # environment will be represented by a numpy array
        self.height = height
        self.width = width
        self.n_ground_agents = n_ground_agents
        self.n_air_agents = n_air_agents
        self.n_flags = n_flags
        self.flag_pos = None
        self.flags = []
        self.seed_range = seed_range # range of seeds to sample
        self.curr_seed = None

        # initalize the terrain
        print("Generating heightmaps...")
        self.heightmaps = []
        for seed in tqdm(self.seed_range):
            self.heightmaps.append(generate_heightmap(self.height, self.width, seed))
        
        self.curr_heightmap = None
        self.gradient_map = None # for initialization purposes
        self.agent_grid = None # for agent search operations

        self.all_agents = {}
        self.agent_id_to_int = None
        self.int_to_agent_id = None

        self.swarm1_feedback = [] # cleared at each step
        self.swarm2_feedback = [] # cleared at each step

        # used for flag capture reward calc
        self.swarm_1_prev_full_capture = 0
        self.swarm_2_prev_full_capture = 0

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
        
        self.damage_events = []
        self.damage_received = {}
        self.eliminated_agents = set()

        self.flag_events = []

        self.history = []

        # damage kernels
        self.ground_damage_kernel = self._generate_gaussian_kernel(n=3, sigma=1)
        self.air_damage_kernel = self._generate_gaussian_kernel(n=5, sigma=1)

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
        high_slope_idxs = np.where(self.gradient_map > settings.MAX_SLOPE)
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

    def init_env_terrain(self):
        rand_seed_idx = np.random.choice(np.arange(len(self.seed_range)))
        self.curr_seed = self.seed_range[rand_seed_idx]
        self.curr_heightmap = self.heightmaps[rand_seed_idx]
        self.gradient_map = compute_slope(self.curr_heightmap, sigma=9)

        # update self.environment (ground-truth array)
        self.environment[:, :, 0] = self.curr_heightmap
        
        print("Getting Flag Positions")
        self.flag_pos = self._flag_pos()
        return
    
    def init_env_entities(self, swarm1_agent_pos, swarm2_agent_pos, swarm1, swarm2):
        for pos in swarm1_agent_pos:
            agent_type, x, y = pos
            if agent_type == "ground":
                self.environment[y, x, 1] = 1 # ground agent of swarm1
                self.environment[y, x, 2] = settings.GROUND_HEALTH
            else:
                self.environment[y, x, 1] = 2 # air agent of swarm1
                self.environment[y, x, 2] = settings.AIR_HEALTH
        
        for pos in swarm2_agent_pos:
            agent_type, x, y = pos
            if agent_type == "ground":
                self.environment[y, x, 1] = 3 # ground agent of swarm2
                self.environment[y, x, 2] = settings.GROUND_HEALTH
            else:
                self.environment[y, x, 1] = 4 # air agent of swarm2
                self.environment[y, x, 2] = settings.AIR_HEALTH
        
        self.environment[:, :, 3] = np.ones((self.height, self.width)) * 2
        for pos in self.flag_pos:
            x, y = pos
            self.environment[y, x, 3] = 0.0 # neutral flag
            self.flags.append(Flag(x, y, 0.0))

        # initialize agent grid (has agent id's in a spatial manner)
        self.all_agents = swarm1.agents | swarm2.agents
        all_agent_ids = list(self.all_agents.keys())

        self.agent_id_to_int = {agent_id: i for i, agent_id in enumerate(all_agent_ids)}
        self.int_to_agent_id = {i: agent_id for agent_id, i in self.agent_id_to_int.items()}

        self.agent_grid = np.full((self.height, self.width), -1, dtype=np.int32)
        agent_positions = [(agent.status.x, agent.status.y, agent.status.id) for agent in self.all_agents.values()]

        for pos in agent_positions:
            x, y, id = pos
            agent_int = self.agent_id_to_int[id]
            self.agent_grid[y, x] = agent_int
        # add the initial environment to the history (for rendering later)
        self.history.append((self.environment.copy(), []))

    def reset(self):
        np.random.seed()
        self.history = []
        self.flags = []
        self.init_env_terrain()
        self.swarm_1_prev_full_capture = 0
        self.swarm_2_prev_full_capture = 0

    def _execute_move(self, swarm, action: MoveAction):
        curr_x = action.agent_status.x
        curr_y = action.agent_status.y
        direction = action.direction
        magnitude = action.magnitude

        # swarm1 uses standard directions, swarm2 uses reversed directions
        x_mult = 1 if swarm == "swarm1" else -1
        y_mult = 1 if swarm == "swarm1" else -1

        direction_map = {
            Direction.NORTH:     (1, 0),
            Direction.NORTHEAST: (1, 1),
            Direction.EAST:      (0, 1),
            Direction.SOUTHEAST: (-1, 1),
            Direction.SOUTH:     (-1, 0),
            Direction.SOUTHWEST: (-1, -1),
            Direction.WEST:      (0, -1),
            Direction.NORTHWEST: (1, -1),
        }

        dx, dy = direction_map[direction]
        dx *= magnitude * x_mult
        dy *= magnitude * y_mult

        new_x = curr_x + dx
        new_y = curr_y + dy

        # clamp new_x and new_y to bounds of grid
        max_y, max_x = self.height, self.width
        new_x = max(0, min(new_x, max_x - 1))
        new_y = max(0, min(new_y, max_y - 1))

        # If target cell occupied, do not move
        if self.agent_grid[new_y, new_x] != -1:
            new_x = curr_x
            new_y = curr_y

        move_reward_modifier = 0
        alpha = 8.0
        if action.agent_status.agent_type == "ground":
            curr_h = self.curr_heightmap[curr_y, curr_x]
            new_h = self.curr_heightmap[new_y, new_x]

            dz = new_h - curr_h
            slope = self.gradient_map[new_y, new_x]
            
            # fails if slope is too much
            if slope > settings.MAX_SLOPE:
                new_x, new_y = curr_x, curr_y

            # movement cost based on slope
            #move_reward_modifier -= min(abs(dz) * alpha, 0.5) 

        if (new_x, new_y) != (curr_x, curr_y):
            action.agent_status.x = new_x
            action.agent_status.y = new_y

            prev_val = self.environment[curr_y, curr_x, 1]
            self.environment[curr_y, curr_x, 1] = 0
            self.environment[new_y, new_x, 1] = prev_val

            prev_val_health = self.environment[curr_y, curr_x, 2]
            self.environment[curr_y, curr_x, 2] = 0
            self.environment[new_y, new_x, 2] = prev_val_health

            self.agent_grid[curr_y, curr_x] = -1
            self.agent_grid[new_y, new_x] = self.agent_id_to_int[action.agent_status.id]

        # computing reward and sending message
        move_reward = self._calc_move_reward(curr_x, curr_y, new_x, new_y, swarm, action.agent_status.agent_type)
        move_reward += move_reward_modifier
        details = {"new_x": new_x, "new_y": new_y, "reward": move_reward}
        msg = FeedbackMessage(action.agent_status.id, "move", details)

        if swarm == "swarm1":
            self.swarm1_feedback.append(msg)
        else:
            self.swarm2_feedback.append(msg)
        return
    def _execute_engage(self, action: EngageAction):
        # basically damage gets dealt to the specified target position
        # we attribute damage, also calculate total damage done
        # damage update function takes damage attributions and calculates rewards/health updates
        agent_status = action.agent_status
        agent_type = agent_status.agent_type
        nominal_damage = 0
        damage_kernel = None
        if agent_type == "ground":
            nominal_damage = settings.NOMINAL_GROUND_DAMAGE
            damage_kernel = self.ground_damage_kernel
        elif agent_type == "air":
            nominal_damage = settings.NOMINAL_AIR_DAMAGE
            damage_kernel = self.air_damage_kernel

        success = np.random.rand() < settings.PROB_ENGAGE_SUCCESS
        if success:
            damage_kernel = damage_kernel * nominal_damage
            # add damage kernel to damage map at target position
            aimed_x = action.target_x
            aimed_y = action.target_y
            target_int = self.prev_agent_grid[aimed_y, aimed_x] # int id of targetted agent
            if target_int in self.eliminated_agents:
                return
            target_agent_id = self.int_to_agent_id[target_int]
            target_status = self.all_agents[target_agent_id].status
            """
            k_size = damage_kernel.shape[0]
            limit = k_size // 2
            x0 = max(0, target_x - limit)
            x1 = min(self.width, target_x + limit + 1)
            y0 = max(0, target_y - limit)
            y1 = min(self.height, target_y + limit + 1)
            kernel_x0 = limit - (target_x - x0)
            kernel_y0 = limit - (target_y - y0)
            kernel_x1 = kernel_x0 + (x1 - x0)
            kernel_y1 = kernel_y0 + (y1 - y0)
            # find if there are any agents in the damage kernel, search the agent grid (friendly fire on)
            agent_grid_slice = self.agent_grid[y0:y1, x0:x1]
            damage_kernel_slice = damage_kernel[kernel_y0:kernel_y1, kernel_x0:kernel_x1]

            ys, xs = np.nonzero(agent_grid_slice != -1)
            for ay, ax in zip(ys, xs):
                target_agent_int = agent_grid_slice[ay, ax]
                target_agent_id = self.int_to_agent_id[target_agent_int]
                damage = damage_kernel_slice[ay, ax]
                self.damage_events.append({"engager_id": agent_status.id, "target_id": target_agent_id, "damage": damage, "target_x": target_x, "target_y": target_y})
                self.damage_received[target_agent_id] = self.damage_received.get(target_agent_id, 0) + damage
            """
            self.damage_events.append({
                "engager_id": agent_status.id,
                "target_id": target_agent_id,
                "damage": nominal_damage,
                "target_x": target_status.x,
                "target_y": target_status.y
            })
            self.damage_received[target_agent_id] = self.damage_received.get(target_agent_id, 0) + nominal_damage
        return 
    
    def _damage_update(self):
        # iterate through damage_received and damage_events
        # update agent statuses + calc reward
        for target_agent_id in self.damage_received:
            status = self.all_agents[target_agent_id].status
            status.health -= self.damage_received[target_agent_id]
            if status.health <= 0:
                # add to elimination list
                self.eliminated_agents.add(target_agent_id)
            # update health in self.environment
            self.environment[status.y, status.x, 2] -= self.damage_received[target_agent_id]
        # reward calculation
        rewards = defaultdict(float)
        for event in self.damage_events:
            engager_id = event["engager_id"]
            target_id = event["target_id"]
            damage = event["damage"]
            #print(f"{engager_id} dealt {damage:.4f} damage to {target_id}")
            target_status = self.all_agents[target_id].status
            target_type = target_status.agent_type
            target_x, target_y = target_status.x, target_status.y
            max_health = settings.GROUND_HEALTH if target_type == "ground" else settings.AIR_HEALTH
            engager_swarm = int(engager_id[0])
            target_swarm = int(target_id[0])
            if engager_swarm == target_swarm:
                rewards[engager_id] -= damage / max_health * 0.5 # penalty for friendly fire, if too much it discourages attacking at all
            else:
                rewards[engager_id] += damage / max_health * settings.ENGAGE_REWARD
            #rewards[target_id] -= damage / max_health * 0.5 # commented out to reduce fear of damage
            msg = FeedbackMessage(None, action_type="engage", details={"engager_id": engager_id, "target_id": target_id, "damage": damage, "target_x": target_x, "target_y": target_y})
            self.swarm1_feedback.append(msg)
            self.swarm2_feedback.append(msg)

        for target_id in self.eliminated_agents:
            # elimination reward is done as fraction of damage done in current step
            # find all agents who dealt damage to target
            #print(f"{target_id} eliminated.")
            total_damage = sum(
                event["damage"] for event in self.damage_events if event["target_id"] == target_id
            )
            if total_damage == 0:
                continue  # for safety shouldn't happen
            for event in self.damage_events:
                if event["target_id"] != target_id:
                    continue
                engager_id = event["engager_id"]
                fraction = event["damage"] / total_damage
                engager_swarm = int(engager_id[0])
                target_swarm = int(target_id[0])
                if engager_swarm == target_swarm:
                    rewards[engager_id] -= fraction * settings.ELIMINATE_REWARD
                else:
                    rewards[engager_id] += fraction * settings.ELIMINATE_REWARD

            # full penalty to eliminated agent
            rewards[target_id] -= settings.ELIMINATE_REWARD * 0.2

            # update overall env state, delete agent
            agent_status = self.all_agents[target_id].status
            y, x = agent_status.y, agent_status.x
            self.agent_grid[y, x] = -1
            self.environment[y, x, 1] = 0 # position 
            self.environment[y, x, 2] = 0 # health -- corrects for "negative health" from above

            # message swarms about eliminations
            swarm = int(target_id[0])
            msg = FeedbackMessage(target_id, "eliminate", {"target_x": x, "target_y": y})
            if swarm == 1:
                self.swarm1_feedback.append(msg)
            else:
                self.swarm2_feedback.append(msg)
        
        for id in rewards:
            swarm = int(id[0])
            msg = FeedbackMessage(id, action_type="engage_reward", details={"reward": rewards[id]})
            if swarm == 1:
                self.swarm1_feedback.append(msg)
            else:
                self.swarm2_feedback.append(msg)
        return rewards

    def _execute_deploy(self, agent_status, action: DeployAction):
        pass

    def _generate_gaussian_kernel(self, n, sigma):
        # Gaussian kernel for damage calculation
        limit = n // 2
        ax = np.linspace(-limit, limit, n)
        xx, yy = np.meshgrid(ax, ax)
        
        # getting squared distance from center
        dist_sq = xx**2 + yy**2
        
        kernel = np.exp(-dist_sq / (2 * sigma**2))
        return kernel
    
    def _flag_capture_calc(self):
        # iterate through each of the flags, record all agents in proximity
        # then calculate net movement and rewards
        capture_radius = settings.FLAG_CAPTURE_RADIUS
        for flag_idx, flag in enumerate(self.flags):
            fx, fy = flag.x, flag.y
            
            slice_x0 = max(0, fx - capture_radius)
            slice_x1 = min(self.width, fx + capture_radius + 1)
            slice_y0 = max(0, fy - capture_radius)
            slice_y1 = min(self.height, fy + capture_radius + 1)

            agent_grid_slice = self.agent_grid[slice_y0:slice_y1, slice_x0:slice_x1]
            
            ys, xs = np.nonzero(agent_grid_slice != -1)

            for ay, ax in zip(ys, xs):
                agent_int = agent_grid_slice[ay, ax]
                agent_id = self.int_to_agent_id[agent_int]
                agent_type = self.all_agents[agent_id].status.agent_type
                # circular radius check
                wx = slice_x0 + ax
                wy = slice_y0 + ay
                dist_sq = (wx - fx)**2 + (wy - fy)**2
                if dist_sq > capture_radius**2:
                    continue
                # figure out capture increment from agent type
                capture_inc = settings.GROUND_FLAG_CAPTURE if agent_type == "ground" else settings.AIR_FLAG_CAPTURE
                self.flag_events.append({"agent_id": agent_id, "capture_inc": capture_inc, "flag_idx": flag_idx})
    
    def _capture_power(self):
        power = defaultdict(lambda: defaultdict(float))
        for event in self.flag_events:
            flag_idx = event["flag_idx"]
            swarm_id = int(event["agent_id"][0])
            power[flag_idx][swarm_id] += event["capture_inc"]
        return power
    
    def _flag_capture_update(self):
        # iterate through flag capture events. calculate rewards for each agent
        # also update dispositions of flags
        rewards = defaultdict(float)
        power_dict = self._capture_power()
        n_agents = self.n_ground_agents + self.n_air_agents
        pop_scale = 1.0 / max(1, n_agents) # for scaling by number of agents in swarm
        for event in self.flag_events:
            agent_id = event["agent_id"]
            swarm = int(agent_id[0])
            flag_idx = event["flag_idx"]
            flag = self.flags[flag_idx]
            # dont give capture reward if flag already captured by team
            friendly_power = power_dict[flag_idx][swarm]
            raw_inc = event["capture_inc"]
            already_captured = (swarm == 1 and flag.disposition <= -1) or (swarm == 2 and flag.disposition >= 1)
            # reward scaled by friendly power already at the flag
            if not already_captured:
                rewards[agent_id] += settings.FLAG_CAPTURE_REWARD * (raw_inc / friendly_power)
            else:
                rewards[agent_id] += settings.FLAG_HOLD_REWARD * 0.5 / friendly_power

            # update disposition of flag
            disp_inc = raw_inc if swarm == 2 else -raw_inc
            flag.disposition = np.clip(flag.disposition + disp_inc, -1.0, 1.0)
            self.environment[flag.y, flag.x, 3] += disp_inc
            msg = FeedbackMessage(agent_id, action_type="capture", details={"flag_idx": event["flag_idx"], "capture_inc": raw_inc})
            if swarm == 1:
                self.swarm1_feedback.append(msg)
            else:
                self.swarm2_feedback.append(msg)

        swarm_1_full_capture = 0
        swarm_2_full_capture = 0

        for flag in self.flags:
            if flag.disposition <= -1:
                flag.disposition = -1
                self.environment[flag.y, flag.x, 3] = -1
                swarm_1_full_capture += 1
            elif flag.disposition >= 1:
                flag.disposition = 1
                self.environment[flag.y, flag.x, 3] = 1
                swarm_2_full_capture += 1
        
        # attempt to encourage capturing all flags
        if swarm_1_full_capture == len(self.flags) and self.swarm_1_prev_full_capture < len(self.flags):
            for agent_id in self.all_agents:
                if int(agent_id[0]) == 1:
                    rewards[agent_id] += (settings.FLAG_CAPTURE_REWARD * 5.0) * pop_scale
        
        if swarm_2_full_capture == len(self.flags) and self.swarm_2_prev_full_capture < len(self.flags):
            for agent_id in self.all_agents:
                if int(agent_id[0]) == 2:
                    rewards[agent_id] += (settings.FLAG_CAPTURE_REWARD * 5.0) * pop_scale

        delta_swarm_1 = swarm_1_full_capture - self.swarm_1_prev_full_capture
        delta_swarm_2 = swarm_2_full_capture - self.swarm_2_prev_full_capture
        flag_advantage = swarm_1_full_capture - swarm_2_full_capture
        for agent_id in self.all_agents:
            swarm = int(agent_id[0])
            capture_val = (delta_swarm_1 - delta_swarm_2) if swarm == 1 else (delta_swarm_2 - delta_swarm_1)
            rewards[agent_id] += capture_val * settings.FLAG_CAPTURE_REWARD * pop_scale * 3
            if swarm == 1:
                rewards[agent_id] += flag_advantage * 0.2
            else:
                rewards[agent_id] -= flag_advantage * 0.2
        self.swarm_1_prev_full_capture = swarm_1_full_capture
        self.swarm_2_prev_full_capture = swarm_2_full_capture
        
        for flag_idx, flag in enumerate(self.flags):
            holding_swarm = 0
            if flag.disposition <= -0.5:
                holding_swarm = 1
            elif flag.disposition >= 0.5:
                holding_swarm = 2        
            enemy_power = power_dict[flag_idx][1] if holding_swarm == 2 else power_dict[flag_idx][2]
            if enemy_power > 0:
                for event in self.flag_events:
                    if event["flag_idx"] != flag_idx:
                        continue
                    agent_id = event["agent_id"]
                    if int(agent_id[0]) != holding_swarm:
                        continue
                    rewards[agent_id] -= enemy_power * settings.FLAG_CAPTURE_REWARD * 0.25
        for id in rewards:
            swarm = int(id[0])
            if rewards[id] != 0:
                msg = FeedbackMessage(id, action_type="capture_reward", details={"reward": rewards[id]})
                if swarm == 1:
                    self.swarm1_feedback.append(msg)
                else:
                    self.swarm2_feedback.append(msg)

    def update(self, actions):
        # handles dynamics given actions (flag capture, movement, engagements, health updates)
        # actions is a dictionary with keys "swarm1" and "swarm2"
        # updates self.environment
        actions_swarm1 = actions["swarm1"]
        actions_swarm2 = actions["swarm2"]
        engage_actions = []
        # execute all move and deploy actions first
        for action in actions_swarm1:
            # swarm 1 is always on the left side, facing right.
            if isinstance(action, MoveAction):
                self._execute_move("swarm1", action)
            elif isinstance(action, DeployAction):
                self._execute_deploy()
            elif isinstance(action, EngageAction):
                engage_actions.append(action)

        for action in actions_swarm2:
            # swarm 2 is always on the right side, facing left.
            if isinstance(action, MoveAction):
                self._execute_move("swarm2", action)
            elif isinstance(action, DeployAction):
                self._execute_deploy()
            elif isinstance(action, EngageAction):
                engage_actions.append(action)

        for act in engage_actions:
            self._execute_engage(act)

        self._damage_update()

        # updates for flag capture
        self._flag_capture_calc()
        self._flag_capture_update()
        return
    
    def step(self, actions):
        # inference + update
        # call swarm.step() --> inference with action list as output
        # swarm.step() also handles SMART + inter-agent comms
        # call self.update(actions)

        # each "action" in the list is a tuple with agent and action
        # for now this seems fine
        self.damage_events = []
        self.damage_received = {}
        self.eliminated_agents = set()
        self.flag_events = []
        self.swarm1_feedback = []
        self.swarm2_feedback = []
        self.prev_agent_grid = self.agent_grid.copy()
        self.update(actions)

        # add updated environment to history!
        self.history.append((self.environment.copy(), self.damage_events))
        return

    def render(self):
        # uses self.environment, produces a 3D image of the environment
        # keep in mind that all air agents have fixed height (maybe in future will add z-axis control)
        pass

    def _calc_move_reward(self, old_x, old_y, new_x, new_y, swarm, agent_type):
        reward = -0.001  # small step penalty

        enemy_flags = []
        for (fx, fy), flag in zip(self.flag_pos, self.flags):
            if (swarm == "swarm1" and flag.disposition >= 0) or (swarm == "swarm2" and flag.disposition <= 0):
                enemy_flags.append((fx, fy))

        if not enemy_flags:
            return reward

        old_d = min(np.hypot(old_x - fx, old_y - fy) for fx, fy in enemy_flags)
        new_d = min(np.hypot(new_x - fx, new_y - fy) for fx, fy in enemy_flags)

        reward += (old_d - new_d) * 0.05
        return reward