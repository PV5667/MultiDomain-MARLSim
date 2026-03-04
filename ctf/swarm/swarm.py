import numpy as np
from swarm.smart import SMART, Entity, FlagEntity, Disposition, Event, EventType
from swarm.agent import GroundAgent, AirAgent
from swarm.core import *
from env.feedback_message import FeedbackMessage
from collections import defaultdict
from constants import settings
from swarm.policy import CentralizedCritic

class Swarm:
    """
    Wrapper class for the swarm of agents.
    This is mainly to facilitate inter-agent and SMART communication.

    Also makes it easier for the env to step forward.
    """
    def __init__(self, height, width, n_ground, n_air, swarm_id: int):
        self.agents = {}
        self.height = height
        self.width = width
        self.smart = SMART(height, width, settings.TTL)
        self.rewards = defaultdict(float)
        self.current_tick = 0
        self.swarm_id = swarm_id
        self.critic = CentralizedCritic()

        self.n_ground_agents = n_ground
        self.n_air_agents = n_air

        # init each of the agents, assign unique id to each agent (unique on global level)
        for i in range(self.n_ground_agents):
            agt_id = f"{self.swarm_id}_ground_{i + 1}"
            agent_type = "ground"
            status = AgentStatus(agt_id, agent_type, 0, 0, 0, settings.GROUND_HEALTH) # dummy coords
            self.agents[agt_id] = GroundAgent(self.swarm_id, status, self.smart)

        for i in range(self.n_air_agents):
            agt_id = f"{self.swarm_id}_air_{i + 1}"
            agent_type = "air"
            status = AgentStatus(agt_id, agent_type, 0, 0, 0, settings.AIR_HEALTH) # dummy coords
            self.agents[agt_id] = AirAgent(self.swarm_id, status, self.smart)

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
        for id in self.active_agents:
            agent = self.agents[id]
            # get env_patch to pass to each agent
            obs_radius = agent.obs_radius
            # calculating indices and slicing -- should be centered at agent pos
            env_patch = self.get_env_patch(environment, agent, obs_radius)
            # getting info from SMART
            smart_obs = self.smart.publish(agent)
            out_actions, comm_out, latent = agent.step(env_patch, smart_obs, self.comms_in[id])
            self.obs[id] = agent.obs
            self.raw_preds[id] = agent.raw_preds
            self.latents[id] = latent
            for i in out_actions:
                if i is not None:
                    actions.append(i)

            if comm_out is not None:
                next_comms.append((agent_id, comm_out))

        self.comms = next_comms
        self.current_tick += 1
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
                if self.swarm_id == engager_id[0]:
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
                if self.swarm_id == target_id[0]:
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
                agent_id = msg.details["agent_id"]
                inc = msg.details["inc"]
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