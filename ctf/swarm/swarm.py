import numpy as np
from swarm.smart import SMART, Entity, FlagEntity, Disposition, Event, EventType
from swarm.agent import GroundAgent, AirAgent, DeterministicGroundAgent, DeterministicAirAgent
from swarm.core import *
from swarm.actions import *
from env.feedback_message import FeedbackMessage
from collections import defaultdict
from constants import settings
from swarm.policy import CentralizedCritic, FFAgent, RecurrrentAgent

class Swarm:
    """
    Wrapper class for the swarm of agents.
    This is mainly to facilitate inter-agent and SMART communication.

    Also makes it easier for the env to step forward.
    """
    def __init__(self, height, width, n_ground, n_air, swarm_id: int, device="cpu", policy_type="ff"):
        self.agents = {}
        self.height = height
        self.width = width
        self.rewards = defaultdict(float)
        self.current_tick = 0
        self.swarm_id = swarm_id
        self.smart = SMART(height, width, settings.TTL)
        self.critic = CentralizedCritic().to(device)
        self.policy_type = policy_type
        if self.policy_type == "ff":
            self.ground_policy = FFAgent(4, 9 + settings.N_FLAGS, 9 + settings.N_FLAGS, 256, 3, 8, settings.GROUND_SPEED).to(device)
            self.air_policy = FFAgent(4, 9 + settings.N_FLAGS, 9 + settings.N_FLAGS, 256, 3, 8, settings.AIR_SPEED).to(device)
        else:
            self.ground_policy = RecurrrentAgent(4, 9 + settings.N_FLAGS, 9 + settings.N_FLAGS, 256, 3, 8, settings.GROUND_SPEED).to(device)
            self.air_policy = RecurrrentAgent(4, 9 + settings.N_FLAGS, 9 + settings.N_FLAGS, 256, 3, 8, settings.AIR_SPEED).to(device)
        self.n_ground_agents = n_ground
        self.n_air_agents = n_air

        # init each of the agents, assign unique id to each agent (unique on global level)
        for i in range(self.n_ground_agents):
            agt_id = f"{self.swarm_id}_ground_{i + 1}"
            agent_type = "ground"
            status = AgentStatus(agt_id, agent_type, 0, 0, 0, settings.GROUND_HEALTH) # dummy coords
            self.agents[agt_id] = GroundAgent(self.swarm_id, status, self.ground_policy, self.smart, device=device, policy_type=self.policy_type)

        for i in range(self.n_air_agents):
            agt_id = f"{self.swarm_id}_air_{i + 1}"
            agent_type = "air"
            status = AgentStatus(agt_id, agent_type, 0, 0, 0, settings.AIR_HEALTH) # dummy coords
            self.agents[agt_id] = AirAgent(self.swarm_id, status, self.air_policy, self.smart, device=device, policy_type=self.policy_type)

    def reset(self, flag_pos, agent_pos):
        self.dones = {agent_id: False for agent_id in self.agents}
        self.comms = []
        self.comms_in = {agent_id: [] for agent_id in self.agents}
        self.active_agents = set()

        self.smart.reset()
        # purely for agt id construction
        n_ground_count = 0
        n_air_count = 0
        for pos in agent_pos:
            agent_type, x, y = pos
            if agent_type == "ground":
                agt_id = f"{self.swarm_id}_ground_{n_ground_count + 1}"
                agent_type = "ground"
                status = AgentStatus(agt_id, agent_type, x, y, 0, settings.GROUND_HEALTH)
                self.agents[agt_id].reset(status)
                self.smart.known_entities[agt_id] = Entity(agt_id, "ground", Disposition.FRIENDLY, status.health, status.x, status.y, status.z)
                n_ground_count += 1
                self.active_agents.add(agt_id)
            else:
                agt_id = f"{self.swarm_id}_air_{n_air_count + 1}"
                agent_type = "air"
                status = AgentStatus(agt_id, agent_type, x, y, 0, settings.AIR_HEALTH)
                self.agents[agt_id].reset(status)
                self.smart.known_entities[agt_id] = Entity(agt_id, "air", Disposition.FRIENDLY, status.health, status.x, status.y, status.z)
                n_air_count += 1
                self.active_agents.add(agt_id)

        for i, pos in enumerate(flag_pos):
            x, y = pos
            self.smart.known_entities[f"flag_{i}"] = FlagEntity(f"flag_{i}", x, y)
        
    def step(self, environment):
        # environment is the ground truth array -- it's passed in at every swarm step since it keeps updating
        self.rewards = defaultdict(float)
        self.obs = {}
        self.raw_preds = {}
        self.latents = {}

        actions = []
        for agent_id, agent in self.agents.items():
            # gather comms from previous step, excluding self
            self.comms_in[agent_id] = [c for other_id, c in self.comms if other_id != agent_id]
        next_comms = []
        # iterate through all of the agents and get their actions
        for id in sorted(self.active_agents):
            agent = self.agents[id]
            # get env_patch to pass to each agent
            obs_radius = agent.obs_radius
            # calculating indices and slicing -- should be centered at agent pos
            env_patch = self.get_env_patch(environment, agent, obs_radius)
            # getting info from SMART
            out_actions, comm_out, latent, hidden_state = agent.step(env_patch, self.comms_in[id])
            self.obs[id] = agent.obs
            self.raw_preds[id] = agent.raw_preds
            self.latents[id] = latent
            for i in out_actions:
                if i is not None:
                    actions.append(i)

            if comm_out is not None:
                next_comms.append((id, comm_out))

        self.comms = next_comms
        self.current_tick += 1
        self.smart.step()
        return actions
    
    def get_env_patch(self, environment, agent, obs_radius):
        # environment shape is (H, W, C)
        H, W, C = environment.shape
        y, x = agent.status.y, agent.status.x
        
        # raw bounds
        y_min, y_max = y - obs_radius, y + obs_radius + 1
        x_min, x_max = x - obs_radius, x + obs_radius + 1

        # clamp bounds to slice actual array
        slice_y_min = max(0, y_min)
        slice_y_max = min(H, y_max)
        slice_x_min = max(0, x_min)
        slice_x_max = min(W, x_max)

        patch = environment[slice_y_min:slice_y_max, slice_x_min:slice_x_max, :]

        # how much to pad by on all sides
        # basically if a val is out of bounds it should result in being more than 0
        pad_y_top = max(0, -y_min)
        pad_y_bottom = max(0, y_max - H)
        pad_x_left = max(0, -x_min)
        pad_x_right = max(0, x_max - W)

        patch = np.pad(
            patch,
            ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right), (0, 0)),
            mode='constant',
            constant_values=0
        )
        return patch
    
    def receive_feedback(self, feedback):
        # feedback is a list of FeedbackMessages
        # send to SMART when necessary
        for msg in feedback:
            msg_type = msg.action_type
            if msg_type == "move":
                self.rewards[msg.agent_id] += msg.details["reward"]
                # update agent position in SMART
                self.smart.update_known_entity_pos(msg.agent_id, msg.details["new_x"], msg.details["new_y"])
            elif msg_type == "engage":
                # either register damage dealt or received
                engager_id = msg.details["engager_id"]
                target_id = msg.details["target_id"]
                target_x, target_y = msg.details["target_x"], msg.details["target_y"]
                damage = msg.details["damage"]
                if self.swarm_id == int(engager_id[0]):
                    # means damage was dealt
                    event = Event(
                        tick=self.current_tick,
                        type=EventType.FRIENDLY_ATTACK,
                        source_id=engager_id,
                        x=target_x,
                        y=target_y,
                        metadata={"damage": damage}
                    )
                    self.smart.add_event(event)
                if self.swarm_id == int(target_id[0]):
                    # means damage was received
                    event = Event(
                        tick=self.current_tick,
                        type=EventType.ENEMY_ATTACK,
                        target_id = target_id,
                        x=target_x,
                        y=target_y,
                        metadata={"damage": damage}
                    )
                    self.smart.add_event(event)
            elif msg_type == "engage_reward":
                self.rewards[msg.agent_id] += msg.details["reward"]
            elif msg_type == "deploy":
                pass
            elif msg_type == "capture":
                # add capture info to SMART? too much?
                agent_id = msg.agent_id
                inc = msg.details["capture_inc"]
                flag_id = f"flag_{msg.details['flag_idx']}"
                event = Event(
                        tick=self.current_tick,
                        type=EventType.FLAG_CAPTURE,
                        source_id = agent_id,
                        target_id = flag_id, # global flag id matches with smart id
                        metadata={"inc": inc}
                    )
                self.smart.add_event(event)
            elif msg_type == "capture_reward":
                self.rewards[msg.agent_id] += msg.details["reward"]
            elif msg_type == "eliminate":
                # remove agent
                self.active_agents.remove(msg.agent_id)
                self.dones[msg.agent_id] = True
                event = Event(
                    tick=self.current_tick,
                    type=EventType.FRIENDLY_ELIMINATE,
                    target_id = msg.agent_id,
                    x = msg.details["target_x"],
                    y = msg.details["target_y"]
                )



class DeterministicSwarm(Swarm):
    def __init__(self, strategy, height, width, n_ground, n_air, swarm_id: int, device="cpu"):
        super().__init__(height, width, n_ground, n_air, swarm_id, device)
        self.strategy = strategy
        del self.critic
        del self.ground_policy
        del self.air_policy
        self.agents = {}
        for i in range(self.n_ground_agents):
            agt_id = f"{self.swarm_id}_ground_{i + 1}"
            status = AgentStatus(agt_id, "ground", 0, 0, 0, settings.GROUND_HEALTH)
            self.agents[agt_id] = DeterministicGroundAgent(self.swarm_id, status, self.smart, device=device)

        for i in range(self.n_air_agents):
            agt_id = f"{self.swarm_id}_air_{i + 1}"
            status = AgentStatus(agt_id, "air", 0, 0, 0, settings.AIR_HEALTH)
            self.agents[agt_id] = DeterministicAirAgent(self.swarm_id, status, self.smart, device=device)

    def reset(self, flag_pos, agent_pos):
        super().reset(flag_pos, agent_pos)
        flag_ids = [f"flag_{i}" for i in range(len(flag_pos))]
        active_list = sorted(self.active_agents)
        n_agents = len(active_list)

        if self.strategy == "distributed":
            # assign agents to flags as evenly as possible
            self.agent_flag_assignment = {}
            for i, agt_id in enumerate(active_list):
                assigned_flag = flag_ids[i % len(flag_ids)]
                self.agent_flag_assignment[agt_id] = assigned_flag
        elif self.strategy == "liquid":
            # all agents start assigned to first flag
            self.flag_order = flag_ids # will remove flags as they're captured
            self.current_target_flag = self.flag_order[0]
            self.agent_flag_assignment = {agt_id: self.current_target_flag for agt_id in active_list}
            self.movable = self.active_agents.copy()
            self.captured_flags = set()
            self.contingent_size = max(1, n_agents // len(flag_ids))
        self.patrolling = set()

    def step(self, environment):
        self.rewards = defaultdict(float)
        actions = []

        if self.strategy == "liquid":
            self._update_liquid_assignments()

        for agt_id in sorted(self.active_agents):
            agent = self.agents[agt_id]
            obs_radius = agent.obs_radius
            env_patch = self.get_env_patch(environment, agent, obs_radius)
            smart_tensors, smart_entities = agent.step(env_patch)
            hostile_in_range = smart_tensors["hostile_in_range"].squeeze()
            agent_actions = []

            if hostile_in_range.any():
                tgt_idx = hostile_in_range.nonzero(as_tuple=True)[0][0].item()
                target_entity = smart_entities[tgt_idx]
                agent_actions.append(EngageAction(
                    agent.status,
                    target_x=target_entity.x,
                    target_y=target_entity.y
                ))

            assigned_flag_id = self.agent_flag_assignment.get(agt_id)
            flag = self.smart.known_entities[assigned_flag_id]
            flag_x, flag_y = flag.x, flag.y
            ax, ay = agent.status.x, agent.status.y
            dist = ((ax - flag_x) ** 2 + (ay - flag_y) ** 2) ** 0.5
            capture_radius = settings.FLAG_CAPTURE_RADIUS

            if agt_id in self.patrolling or dist <= capture_radius:
                self.patrolling.add(agt_id)
                angle = np.random.uniform(0, 2 * np.pi)
                patrol_dx = int(round(np.cos(angle) * (capture_radius * 0.5)))
                patrol_dy = int(round(np.sin(angle) * (capture_radius * 0.5)))
                tx = np.clip(flag_x + patrol_dx, 0, self.width - 1)
                ty = np.clip(flag_y + patrol_dy, 0, self.height - 1)
                direction, magnitude = self._dir_mag_to(ax, ay, tx, ty, agent)
            else:
                direction, magnitude = self._dir_mag_to(ax, ay, flag_x, flag_y, agent)

            agent_actions.append(MoveAction(agent.status, direction=direction, magnitude=magnitude))
            actions.extend(agent_actions)
        self.current_tick += 1
        self.smart.step()
        return actions

    def _update_liquid_assignments(self):
        # check if current target flag is captured. if so, leave a contigent and move on
        if not self.flag_order:
            return
        current_flag_id = self.flag_order[0]
        flag = self.smart.known_entities[current_flag_id]
        flag_disp = flag.disposition
        owned = flag_disp > 0.5 # flag disp is normalized at time of adding to smart
        if owned and current_flag_id not in self.captured_flags:
            self.captured_flags.add(current_flag_id)
            self.flag_order.pop(0)
            if not self.flag_order: # no next flag
                return
            next_flag = self.flag_order[0]
            def dist_to_flag(agt_id):
                s = self.agents[agt_id].status
                return (s.x - flag.x) ** 2 + (s.y - flag.y) ** 2
            moving_agts = sorted(self.movable, key=dist_to_flag)
            staying = set(moving_agts[:self.contingent_size])
            moving = moving_agts[self.contingent_size:]
            for agt_id in moving:
                self.agent_flag_assignment[agt_id] = next_flag
                self.patrolling.discard(agt_id)
            for agt_id in staying:
                self.agent_flag_assignment[agt_id] = current_flag_id
                self.movable.discard(agt_id)
    def _dir_mag_to(self, ax, ay, tx, ty, agent):
        dx = tx - ax
        dy = ty - ay
        if self.swarm_id == 1:
            dx = -dx
            dy = -dy
        if dx == 0 and dy == 0:
            return Direction.NORTH, 0
        angle = np.arctan2(dy, dx)
        idx = int((angle + np.pi) / (2 * np.pi) * 8 + 0.5) % 8
        direction = Direction(idx + 1)
        speed = settings.AIR_SPEED if isinstance(agent, AirAgent) else settings.GROUND_SPEED
        magnitude = min(int((dx**2 + dy**2)**0.5), speed)
        return direction, magnitude