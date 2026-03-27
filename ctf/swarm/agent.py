"""Defining the agent types, along with their action and observation spaces."""
import torch
import numpy as np
from swarm.actions import MoveAction, EngageAction, DeployAction, Direction
from constants import settings
from swarm.core import *

class Agent:
    def __init__(self):
        self.swarm_id = None
        self.policy = None
        self.status = None
        self.smart = None
        self.obs = {} # actual tensors
        self.obs_radius = None
        self.recurrent_state = None
        self.smart_entities = []
    
    def process_env_patch(self, env_patch):
        env_patch = env_patch.copy()
        type_channel = env_patch[:, :, 1]
        flag_channel = env_patch[:, :, 3]
        # only flipping actual flag cells
        ax, ay = self.status.x, self.status.y
        for entity in self.smart.known_entities.values():
            if not isinstance(entity, FlagEntity):
                continue
            dx = entity.x - ax
            dy = entity.y - ay
            if abs(dx) > self.obs_radius or abs(dy) > self.obs_radius:
                continue
            px = dx + self.obs_radius
            py = dy + self.obs_radius
            raw_disp = float(flag_channel[py, px])
            disp = -raw_disp if self.swarm_id == 1 else raw_disp
            env_patch[py, px, 3] = disp
            self.smart.update_flag_disp(entity.x, entity.y, disp)  
                
        if self.swarm_id == 2:
            remap = {1: 3, 2: 4, 3: 1, 4: 2}
            new_type_channel = type_channel.copy()
            for old_val, new_val in remap.items():
                new_type_channel[type_channel == old_val] = new_val
            env_patch[:, :, 1] = new_type_channel
        
        # flips to make obs symmetric
        if self.swarm_id == 2:
            env_patch = env_patch[::-1, ::-1, :].copy()
        return env_patch
        
    def to_tensor(self, env_patch):
        patch = torch.tensor(env_patch, dtype=torch.float32)
        patch = patch.permute(2, 0, 1)
        patch = patch.unsqueeze(0)
        return patch

    def report_entity_obs(self, env_patch):
        # finds entities to report and also flips values as necessary
        type_channel = env_patch[:, :, 1]
        health_channel = env_patch[:, :, 2]

        entity_obs = []
        mask = np.isin(type_channel, [3, 4])
        ys, xs = np.where(mask)

        for y_patch, x_patch in zip(ys, xs):
            entity_type_val = int(type_channel[y_patch, x_patch])
            health = float(health_channel[y_patch, x_patch])

            if self.swarm_id == 2:
                # unflip patch coordinates back to world coordinates
                world_x = self.status.x + self.obs_radius - x_patch
                world_y = self.status.y + self.obs_radius - y_patch
            else:
                world_x = self.status.x - self.obs_radius + x_patch
                world_y = self.status.y - self.obs_radius + y_patch

            ent_type = "ground" if entity_type_val == 3 else "air"

            entity_obs.append(
                EntityObservation(
                    tick=self.smart.current_tick,
                    type=ent_type,
                    disposition=Disposition.ENEMY,
                    x=world_x,
                    y=world_y,
                    z=0,
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
        x_sign = 1 if self.swarm_id == 1 else -1
        # now iterate through each entity and encode type, disposition, health, rel. pos
        for ent in all_entities:
            type_vec = [0, 0, 0]  # ground, air, flag
            if isinstance(ent, Entity):
                if ent.type == "ground":
                    type_vec[0] = 1
                elif ent.type == "air":
                    type_vec[1] = 1
            elif isinstance(ent, FlagEntity):
                type_vec[2] = 1
            disp_vec = [0, 0]  # friendly, enemy
            if isinstance(ent, Entity):
                if ent.disposition.name == "FRIENDLY":
                    disp_vec[0] = 1
                else:
                    disp_vec[1] = 1

            health = getattr(ent, "health", 0.0) / 100.0 # normalized

            rel_x = x_sign * (ent.x - agent_x) / self.obs_radius
            rel_y = x_sign * (ent.y - agent_y) / self.obs_radius

            flag_idx_vec = [0.0] * settings.N_FLAGS
            if isinstance(ent, FlagEntity):
                idx = int(ent.id.split("_")[1])
                flag_idx_vec[idx] = 1.0

            flag_disp = 0.0
            if isinstance(ent, FlagEntity):
                flag_disp = getattr(ent, 'disposition', 0.0)

            vec = type_vec + disp_vec + [health, rel_x, rel_y, flag_disp] + flag_idx_vec
            entity_vectors.append(vec)
            entity_mask.append(False)

            if isinstance(ent, Entity):
                is_enemy = ent.disposition.name != "FRIENDLY"
                dist_sq = (ent.x - agent_x) ** 2 + (ent.y - agent_y) ** 2
                in_range = is_enemy and dist_sq <= self.obs_radius ** 2 and ent.last_seen_tick == self.smart.current_tick
                hostile_in_range.append(in_range)
            else:
                hostile_in_range.append(False)
            
        # creating padding
        # this is just an approximation
        MAX_ENTITIES = (settings.N_GROUND_AGENTS + settings.N_AIR_AGENTS)*2 + settings.N_FLAGS
        PAD_LEN = 3 + 2 + 1 + 2 + 1 + settings.N_FLAGS
        
        entity_vectors = entity_vectors[:MAX_ENTITIES]
        entity_mask = entity_mask[:MAX_ENTITIES]
        hostile_in_range = hostile_in_range[:MAX_ENTITIES]
        self.smart_entities = all_entities[:MAX_ENTITIES]

        while len(entity_vectors) < MAX_ENTITIES:
            entity_vectors.append([0.0] * PAD_LEN)
            entity_mask.append(True)
            hostile_in_range.append(False)
            self.smart_entities.append(None)

        entity_tensor = torch.tensor(entity_vectors, dtype=torch.float32)
        entity_mask = torch.tensor(entity_mask, dtype=torch.bool)
        hostile_in_range = torch.tensor(hostile_in_range, dtype=torch.bool)
        
        # iterate through events and encode as vecs
        event_vectors = []
        event_mask = []
        for event in smart_obs["relevant_events"]:
            type_vec = [0, 0, 0, 0, 0]
            type_vec[event.type.value] = 1
            flag_id_vec = [0] * settings.N_FLAGS
            
            if event.type == EventType.FLAG_CAPTURE:
                idx = int(event.target_id.split("_")[1])
                flag_id_vec[idx] = 1

            rel_x = x_sign * (event.x - agent_x)/self.obs_radius if event.x is not None else 0.0            
            rel_y = x_sign * (event.y - agent_y) / self.obs_radius if event.y is not None else 0.0

            if event.metadata is None:
                event.metadata = {}
            damage = event.metadata.get("damage", 0) / max(settings.NOMINAL_GROUND_DAMAGE, settings.NOMINAL_AIR_DAMAGE)
            time_delta = (self.smart.current_tick - event.tick) / self.smart.ttl

            vec = type_vec + [rel_x, rel_y, damage, time_delta] + flag_id_vec
            event_vectors.append(vec)
            event_mask.append(False)
        
        MAX_EVENTS = settings.MAX_EVENTS
        event_vectors = event_vectors[:MAX_EVENTS]
        event_mask = event_mask[:MAX_EVENTS]
        EVENT_PAD_LEN = 5 + 4 + settings.N_FLAGS
        while len(event_vectors) < MAX_EVENTS:
            event_vectors.append([0.0] * EVENT_PAD_LEN)
            event_mask.append(True)

        event_tensor = torch.tensor(event_vectors, dtype=torch.float32)
        event_mask = torch.tensor(event_mask, dtype=torch.bool)

        return {
            "entities": entity_tensor.unsqueeze(0),
            "entity_mask": entity_mask.unsqueeze(0),
            "events": event_tensor.unsqueeze(0),
            "hostile_in_range": hostile_in_range.unsqueeze(0).to(self.device),
            "event_mask": event_mask.unsqueeze(0),
        }
    
    def _internal_state_vec(self):
        # include health, agent type
        # will be used in future for recurrence
        health = self.status.health
        agent_type = self.status.agent_type
        vec = [0, 0, 0]
        if agent_type == "ground":
            vec[0] = health / settings.GROUND_HEALTH
            vec[1] = 1
        else:
            vec[0] = health / settings.AIR_HEALTH
            vec[2] = 1
        return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
    
    def process_action_dists(self, action_dists, in_range_mask):
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
        tgt_idx = None
        if engage_decision == 1:
            # choose target based on entity scores
            if in_range_mask.any():
                target_scores = action_dists["engage_tgt"]
                tgt_idx = torch.argmax(target_scores).item()
                target_entity = self.smart_entities[tgt_idx]
                engage_action = EngageAction(
                    agent_status,
                    target_x=target_entity.x,
                    target_y=target_entity.y
                )
        return move_action, engage_action, tgt_idx
    
    def step(self, env_patch, comms_in):
        self.obs = {}
        self.raw_preds = {}
        self.smart_entities = []
        # env_patch is localized to the fixed area around the agent (patching done during swarm.step())
        env_patch = self.process_env_patch(env_patch)
        entity_obs_list = self.report_entity_obs(env_patch)
        patch = self.to_tensor(env_patch)
        if len(entity_obs_list) > 0:
            self.smart.add_entity_observations(entity_obs_list)
        self.smart.update_known_entity_pos(self.status.id, self.status.x, self.status.y)
        smart_obs = self.smart.publish(self.status)
        smart_tensors = self.process_smart_obs(smart_obs)
        internal_state = self._internal_state_vec()
        if len(comms_in) > 0:
            comms_tensor = torch.stack(comms_in, dim=0).mean(dim=0).to(self.device)
        else:
            comms_tensor = torch.zeros((1, 256)).to(self.device)

        self.obs["patch"] = patch.to(self.device)
        self.obs["entities"] = smart_tensors["entities"].to(self.device)
        self.obs["entity_mask"] = smart_tensors["entity_mask"].to(self.device)
        self.obs["events"] = smart_tensors["events"].to(self.device)
        self.obs["event_mask"] = smart_tensors["event_mask"].to(self.device)
        self.obs["hostile_in_range"] = smart_tensors["hostile_in_range"].to(self.device)
        self.obs["comms_in"] = comms_tensor.to(self.device)
        self.obs["internal_state"] = internal_state.to(self.device)
        if self.recurrent_state is not None:
            self.obs["h"] = self.recurrent_state[0].to(self.device)
            self.obs["c"] = self.recurrent_state[1].to(self.device)
        else:
            self.obs["h"] = torch.zeros(comms_tensor.size(0), 256, device=comms_tensor.device)
            self.obs["c"] = torch.zeros(comms_tensor.size(0), 256, device=comms_tensor.device)

        if self.policy_type == "ff":
            action_dists, comms_out, latent = self.policy(self.obs)
        else:
            action_dists, comms_out, latent, recurrent_state = self.policy(self.obs, self.recurrent_state)
            h, c = recurrent_state
            self.recurrent_state = (h.detach(), c.detach())
        action_dists["engage_tgt"] = action_dists["engage_tgt"].masked_fill(~smart_tensors["hostile_in_range"], -1e9)
        self.raw_preds["actions"] = action_dists
        self.raw_preds["comms_out"] = comms_out
        # process the action_dists and output the final action
        move, engage, _ = self.process_action_dists(action_dists, smart_tensors["hostile_in_range"])
        return (move, engage), comms_out, latent, self.recurrent_state
    
    def evaluate(self, stored_obs, stored_action_idx):
        action_dists, _, latent = self.policy(stored_obs)
        action_dists["engage_tgt"] = action_dists["engage_tgt"].masked_fill(
            ~stored_obs["hostile_in_range"], -1e9
        )
        dir_dist = torch.distributions.Categorical(logits=action_dists["move_dir"].squeeze(0))
        mag_dist = torch.distributions.Categorical(logits=action_dists["move_mag"].squeeze(0))
        engage_dist = torch.distributions.Categorical(logits=action_dists["engage_bin"].squeeze(0))
        tgt_dist = torch.distributions.Categorical(logits=action_dists["engage_tgt"].squeeze(0))
        log_prob = dir_dist.log_prob(stored_action_idx["dir_idx"]) + mag_dist.log_prob(stored_action_idx["mag_idx"]) + engage_dist.log_prob(stored_action_idx["engage_idx"])
        if stored_action_idx["engage_idx"] == 1:
            log_prob += tgt_dist.log_prob(stored_action_idx["tgt_idx"])
        entropy = (dir_dist.entropy() + mag_dist.entropy() + engage_dist.entropy())
        if stored_action_idx["engage_idx"] == 1:
            entropy += tgt_dist.entropy()
        return log_prob, latent, entropy
    
    def reset(self, new_status):
        self.status = new_status

class GroundAgent(Agent):
    def __init__(self, swarm_id, status: AgentStatus, policy, smart, device="cpu", policy_type="ff"):
        super().__init__()
        self.swarm_id = swarm_id
        self.status = status
        self.smart = smart
        self.obs_radius = settings.GROUND_OBS_RADIUS
        self.policy = policy
        self.device = device
        self.policy_type = policy_type

class AirAgent(Agent):
    def __init__(self, swarm_id, status: AgentStatus, policy, smart, device="cpu", policy_type="ff"):
        super().__init__()
        self.swarm_id = swarm_id
        self.status = status
        self.smart = smart
        self.obs_radius = settings.AIR_OBS_RADIUS
        self.policy = policy
        self.device = device
        self.policy_type = policy_type

class DeterministicAgent(Agent):
    def __init__(self):
        super().__init__()
        # used mainly for obs processing (control rests at swarm level)
    def step(self, env_patch):
        # swarm-level controller just needs to know hostile_in_range and position of agent
        self.obs = {}
        self.raw_preds = {}
        self.smart_entities = []
        # env_patch is localized to the fixed area around the agent (patching done during swarm.step())
        env_patch = self.process_env_patch(env_patch)
        entity_obs_list = self.report_entity_obs(env_patch)
        if len(entity_obs_list) > 0:
            self.smart.add_entity_observations(entity_obs_list)
        self.smart.update_known_entity_pos(self.status.id, self.status.x, self.status.y)
        smart_obs = self.smart.publish(self.status)
        smart_tensors = self.process_smart_obs(smart_obs)
        return smart_tensors, self.smart_entities
    
class DeterministicGroundAgent(DeterministicAgent):
    def __init__(self, swarm_id, status: AgentStatus, smart, device="cpu"):
        self.swarm_id = swarm_id
        self.status = status
        self.smart = smart
        self.obs_radius = settings.GROUND_OBS_RADIUS
        self.device = device

class DeterministicAirAgent(DeterministicAgent):
    def __init__(self, swarm_id, status: AgentStatus, smart, device="cpu"):
        self.swarm_id = swarm_id
        self.status = status
        self.smart = smart
        self.obs_radius = settings.AIR_OBS_RADIUS
        self.device = device
    