"""Defining the agent types, along with their action and observation spaces."""
import torch
import numpy as np
from swarm.actions import Action, MoveAction, EngageAction, DeployAction, Direction
from swarm.smart import SMART, Entity, FlagEntity, EntityObservation, EventType
from policy import ActorAgent
from constants import settings
from dataclasses import dataclass

@dataclass
class AgentStatus:
    """
    Internal State of the Agent
    Allows for lightweight replay + communication.
    """
    id: str
    agent_type: str
    x: int
    y: int
    z: int
    health: int
    # will probably add more (e.g. scout mode, engage mode when working with rules-based behavior)

class Agent:
    def __init__(self):
        self.swarm_id = None
        self.policy = None
        self.status = None
        self.smart = None
        self.obs = {} # actual tensors
        self.obs_radius = None
        self.smart_entities = []

    def process_env_patch(self, env_patch):
        # basically go through patch, make entity observations for everything
        # resolution is done on the SMART level
        patch = torch.tensor(env_patch, dtype=torch.float32)
        patch = patch.permute(2, 0, 1)  # (C, H, W)
        return patch
    
    def report_entity_obs(self, env_patch):
        entity_obs = []
        type_channel = env_patch[:, :, 1]
        health_channel = env_patch[:, :, 2]
        if self.swarm_id == 1:
            hostile_vals = [3, 4]
        else:
            hostile_vals = [1, 2]
        
        mask = np.isin(type_channel, hostile_vals)
        ys, xs = np.where(mask)

        for y_patch, x_patch in zip(ys, xs):
            entity_type_val = int(type_channel[y_patch, x_patch])
            health = float(health_channel[y_patch, x_patch])

            world_x = self.status.x - self.obs_radius + x_patch
            world_y = self.status.y - self.obs_radius + y_patch

            if entity_type_val in [1, 3]:
                ent_type = "ground"
            else:
                ent_type = "air"

            entity_obs.append(
                EntityObservation(
                    observer_id=self.status.id,
                    entity_type=ent_type,
                    x=world_x,
                    y=world_y,
                    health=health
                )
            )
        return entity_obs
    
    def process_smart_obs(self, smart_obs):
        agent_x = self.status.x
        agent_y = self.status.y

        entity_vectors = []
        entity_mask = []
        event_vectors = []
        hostile_in_range = []

        all_entities = []
        all_entities.extend(smart_obs["foreign_entities"])
        all_entities.extend(smart_obs["flags"])
        all_entities.extend(smart_obs["known_entities"])
        self.smart_entities = all_entities # for targetting info
        # now iterate through each entity and encode type, disposition, health, rel. pos
        for ent in all_entities:
            type_vec = [0, 0, 0]  # ground, air, flag
            if ent.type == "ground":
                type_vec[0] = 1
            elif ent.type == "air":
                type_vec[1] = 1
            else:
                type_vec[2] = 1

            disp_vec = [0, 0]  # friendly, enemy
            if isinstance(ent, Entity):
                if ent.disposition.name == "FRIENDLY":
                    disp_vec[0] = 1
                else:
                    disp_vec[1] = 1

            health = getattr(ent, "health", 0.0) / 100.0 # normalized

            rel_x = (ent.x - agent_x) / self.obs_radius
            rel_y = (ent.y - agent_y) / self.obs_radius

            flag_idx_vec = [0.0] * settings["N_FLAGS"]
            if isinstance(ent, FlagEntity):
                idx = int(ent.id[-1])
                flag_idx_vec[idx] = 1.0

            vec = type_vec + disp_vec + [health, rel_x, rel_y] + flag_idx_vec
            entity_vectors.append(vec)
            entity_mask.append(False)

            if isinstance(ent, Entity):
                is_enemy = ent.disposition.name != "FRIENDLY"
                dist_sq = (ent.x - agent_x) ** 2 + (ent.y - agent_y) ** 2
                in_range = is_enemy and dist_sq <= self.obs_radius ** 2
                hostile_in_range.append(in_range)
            else:
                hostile_in_range.append(False)
            
        # creating padding
        MAX_ENTITIES = settings["N_GROUND_AGENTS"] + settings["N_AIR_AGENTS"] + settings["N_FLAGS"]
        PAD_LEN = 3 + 2 + 1 + 2 + settings["N_FLAGS"]
        while len(entity_vectors) < MAX_ENTITIES:
            entity_vectors.append([0.0] * PAD_LEN)
            entity_mask.append(True)
            hostile_in_range.append(False)

        entity_tensor = torch.tensor(entity_vectors, dtype=torch.float32)
        entity_mask = torch.tensor(entity_mask, dtype=torch.bool)
        hostile_in_range = torch.tensor(hostile_in_range, dtype=torch.bool)

        # iterate through events and encode as vecs
        for event in smart_obs["relevant_events"]:
            type_vec = [0, 0, 0, 0, 0]
            type_vec[event.type.value] = 1
            flag_id_vec = [0] * settings["N_FLAGS"]
            
            if event.type == EventType.FLAG_CAPTURE:
                idx = int(event.target_id[-1])
                flag_id_vec[idx] = 1
            
            rel_x = (event.x - agent_x) / self.obs_radius if event.x is not None else 0.0
            rel_y = (event.y - agent_y) / self.obs_radius if event.y is not None else 0.0

            damage = event.metadata.get("damage", 0) / max(settings["NOMINAL_GROUND_DAMAGE"], settings["NOMINAL_AIR_DAMAGE"])
            time_delta = (self.smart.current_tick - event.tick) / self.smart.ttl

            vec = type_vec + [rel_x, rel_y, damage, time_delta] + flag_id_vec
            event_vectors.append(vec)
        
        event_mask = []
        MAX_EVENTS = settings["MAX_EVENTS"]
        EVENT_PAD_LEN = 5 + 4 + settings["N_FLAGS"]
        while len(event_vectors) < MAX_EVENTS:
            event_vectors.append([0.0] * EVENT_PAD_LEN)
            event_mask.append(True)

        event_vectors = event_vectors[:MAX_EVENTS]
        event_mask = [False] * min(len(smart_obs["relevant_events"]), MAX_EVENTS) + event_mask
        event_mask = event_mask[:MAX_EVENTS]

        event_tensor = torch.tensor(event_vectors, dtype=torch.float32)
        event_mask = torch.tensor(event_mask, dtype=torch.bool)

        return {
            "entities": entity_tensor,
            "entity_mask": entity_mask,
            "events": event_tensor,
            "hostile_in_range": hostile_in_range,
            "event_mask": event_mask,
        }
    
    def _internal_state_vec(self):
        # include health, agent type
        # will be used in future for recurrence
        health = self.status.health
        agent_type = self.status.agent_type
        vec = [0, 0, 0]
        if agent_type == "ground":
            vec[0] = health / settings["GROUND_HEALTH"]
            vec[1] = 1
        else:
            vec[0] = health / settings["AIR_HEALTH"]
            vec[2] = 1
        
        return torch.tensor(vec, dtype=torch.float32)
    
    def process_action_dists(self, action_dists):
        # goes over different decision heads. returns actual action
        agent_status = self.status  # current agent status
        move_dir_logits = action_dists["move_dir"]
        move_mag_logits = action_dists["move_mag"]

        dir_idx = torch.argmax(move_dir_logits).item()
        if dir_idx == 0: # NOOP
            # dir doesn't matter if magnitude = 0
            move_action = MoveAction(agent_status, direction=Direction.NORTH, magnitude=0)
        else:
            direction = Direction(dir_idx)
            mag_idx = torch.argmax(move_mag_logits).item() + 1  # 1..motion_range
            move_action = MoveAction(agent_status, direction=direction, magnitude=mag_idx)

        engage_logits = action_dists["engage_bin"]
        engage_prob = torch.softmax(engage_logits, dim=-1)
        engage_decision = torch.argmax(engage_prob).item()

        engage_action = None
        if engage_decision == 1:
            # choose target based on entity scores
            target_scores = action_dists["engage_tgt"]
            tgt_idx = torch.argmax(target_scores).item()
            target_entity = self.smart_entities[tgt_idx]
            engage_action = EngageAction(
                agent_status,
                target_x=target_entity.x,
                target_y=target_entity.y
            )
        return move_action, engage_action
    def step(self, env_patch, smart_obs, comms_in):
        self.obs = {}
        self.smart_entities = []
        # env_patch is localized to the fixed area around the agent (patching done during swarm.step())
        patch = self.process_env_patch(env_patch)
        smart_tensors = self.process_smart_obs(smart_obs)
        internal_state = self._internal_state_vec()
        if len(comms_in) > 0:
            comms_tensor = torch.stack(comms_in, dim=0).mean(dim=0)
        else:
            comms_tensor = torch.zeros(256)
        self.obs["patch"] = patch
        self.obs["entities"] = smart_tensors["entities"]
        self.obs["entity_mask"] = smart_tensors["entity_mask"]
        self.obs["events"] = smart_tensors["events"]
        self.obs["event_mask"] = smart_tensors["event_mask"]
        self.obs["comms_in"] = comms_tensor
        self.obs["internal_state"] = internal_state

        action_dists, comms_out = self.policy(self.obs)
        action_dists["engage_tgt"] = action_dists["engage_tgt"].masked_fill(~smart_obs["hostile_in_range"], -1e9)
        # process the action_dists and output the final action
        action = self.process_action_dists(action_dists)
        return action, comms_out
    
class GroundAgent(Agent):
    def __init__(self, swarm_id, status: AgentStatus, smart: SMART):
        super().__init__()
        self.swarm_id = swarm_id
        self.status = status
        self.smart = smart
        self.obs_radius = settings["GROUND_OBS_RADIUS"]
        self.policy = ActorAgent(4, 8, 9, 256, 2, 8, settings["GROUND_SPEED"])

class AirAgent(Agent):
    def __init__(self, swarm_id, status: AgentStatus, smart: SMART):
        super().__init__()
        self.swarm_id = swarm_id
        self.status = status
        self.smart = smart
        self.obs_radius = settings["AIR_OBS_RADIUS"]
        self.policy = ActorAgent(4, 8, 9, 256, 2, 8, settings["AIR_SPEED"])