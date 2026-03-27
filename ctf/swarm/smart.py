import numpy as np
from swarm.agent import Agent
from env.flag import Flag
from constants import settings
from swarm.core import *
from scipy.optimize import linear_sum_assignment


class SMART:
    """
    SMART (Swarm-based Multi-agent Awareness & Real-time Tracking)
    
    Entities: Dictionary based on agent id
    Events: Dictionary based on timestamp
    Events PQ: TODO priority queue for events
    Spatial Index: TODO some kind of spatial index, has both events and entities.

    Note: SMART is a *swarm-specific* model that acts as a global information repository. It is updated as agents observe entities and events.


    """
    def __init__(self, height, width, ttl):
        self.ttl = ttl # time-to-live for entities
        self.height = height
        self.width = width
        self.relevance_radius = 100

        self.reset()

    def reset(self):
        self.current_tick = 0
        self.known_entities = {} # initialized with friendly agents + flags
        self.foreign_entities = {}
        self.events = []
        self.foreign_grid = np.full((self.height, self.width), -1)
        self.event_grid = np.full((self.height, self.width), -1)
        
        self.next_foreign_air = 0
        self.next_foreign_ground = 0
        self.next_int = 0

        self.foreign_id_to_int = {}
        self.foreign_int_to_id = {}

    def publish(self, agent_status: AgentStatus):
        # basically look at the position of the agent, along with tick, and output info accordingly
        obs = {} # consists of events and entities
        x, y = agent_status.x, agent_status.y
        # spatial index: all entities within 200x200 block
        x0 = max(0, x - self.relevance_radius)
        x1 = min(self.width, x + self.relevance_radius + 1)
        y0 = max(0, y - self.relevance_radius)
        y1 = min(self.height, y + self.relevance_radius + 1)
        foreign_patch = self.foreign_grid[y0:y1, x0:x1]
        foreign_in_patch = np.unique(foreign_patch)
        foreign_in_patch = [self.foreign_int_to_id[i] for i in foreign_in_patch if i != -1]
        foreign_entities = [self.foreign_entities[id] for id in foreign_in_patch]
        obs["foreign_entities"] = foreign_entities
        # getting relevant known entities[]
        flags = []
        known_entities = []
        for entity in self.known_entities.values():
            if isinstance(entity, FlagEntity):
                flags.append(entity)
                continue
            if entity.id == agent_status.id:  # skipping self
                continue
            dx = entity.x - x
            dy = entity.y - y
            if dx*dx + dy*dy <= self.relevance_radius*self.relevance_radius:
                known_entities.append(entity)
        obs["flags"] = flags
        obs["known_entities"] = known_entities
        # getting relevant events
        relevant_events = self._get_relevant_events(x, y)
        obs["relevant_events"] = relevant_events
        return obs
    
    def _get_relevant_events(self, x, y):
        relevant = []
        for event in self.events:
            # these are distance-agnostic
            if event.type in [EventType.FLAG_CAPTURE, EventType.FRIENDLY_ELIMINATE]:
                relevant.append(event)
            else:
                dx = event.x - x
                dy = event.y - y
                if dx*dx + dy*dy <= self.relevance_radius**2:
                    relevant.append(event)
        return relevant
    
    def add_entity_observations(self, observations: list[EntityObservation]):
        matches = self._find_matching_entities(observations)
        for obs, entity in matches:
            if entity is not None:
                self._update_foreign_entity(entity, obs)
            else:
                self._add_foreign_entity(obs)

    def _find_matching_entities(self, observations: list[EntityObservation]):
        # Hungarian algorithm + entities last seen this tick can't be match candidates
        results = []
        all_entities = list(self.foreign_entities.values())
        for obs_type in ("ground", "air"):
            obs_group = [o for o in observations if o.type == obs_type]
            ent_group = [e for e in all_entities if e.type == obs_type]

            if not obs_group:
                continue
            if not ent_group:
                results.extend((o, None) for o in obs_group)
                continue

            match_radius = settings.GROUND_SPEED if obs_type == "ground" else settings.AIR_SPEED

            obs_coords = np.array([[o.x, o.y] for o in obs_group])
            ent_coords = np.array([[e.x, e.y] for e in ent_group])
            # numpy broadcasting to create (n, m, 2) array from (n, 1, 2) and (m, 1, 2) arrays
            diff = obs_coords[:, np.newaxis, :] - ent_coords[np.newaxis, :, :]
            # then taking norm along axis 2, creating (n,m) matrix of dists
            dist_matrix = np.linalg.norm(diff, axis=2)

            seen_this_tick = np.array([
                e.last_seen_tick == self.current_tick for e in ent_group
            ], dtype=bool)
            within_radius = dist_matrix <= match_radius
            exact_match   = dist_matrix < 1e-6

            # if seen this tick, only True if exact match. else only True if within radius
            within_radius = np.where(seen_this_tick[np.newaxis, :], exact_match, within_radius)
            # 1e6 is enough of a "large distance" for our map scale
            cost_for_solver = np.where(within_radius, dist_matrix, 1e6)
            
            row_indices, col_indices = linear_sum_assignment(cost_for_solver)
            matched_obs_idx = set()
            for r, c in zip(row_indices, col_indices):
                if within_radius[r, c]:
                    results.append((obs_group[r], ent_group[c]))
                    matched_obs_idx.add(r)
            for i, obs in enumerate(obs_group):
                if i not in matched_obs_idx:
                    results.append((obs, None))
        return results
    
    def update_known_entity_pos(self, entity_id, new_x, new_y):
        entity = self.known_entities[entity_id]
        entity.x = new_x
        entity.y = new_y

    def update_flag_disp(self, flag_x, flag_y, disp):
        for entity in self.known_entities.values():
            if isinstance(entity, FlagEntity):
                if entity.x == flag_x and entity.y == flag_y:
                    entity.disposition = disp
                    return

    def _update_foreign_entity(self, entity: Entity, entity_obs: EntityObservation):
        old_x, old_y = entity.x, entity.y
        entity.last_seen_tick = self.current_tick
        entity.x = entity_obs.x
        entity.y = entity_obs.y
        entity.z = entity_obs.z
        entity.health = entity_obs.health
        id_int = self.foreign_id_to_int[entity.id]
        self.foreign_grid[old_y, old_x] = -1
        self.foreign_grid[entity.y, entity.x] = id_int 
    
    def _add_foreign_entity(self, entity_obs: EntityObservation):
        new_id = ""
        if entity_obs.type == "ground":
            new_id = f"foreign_ground_{self.next_foreign_ground}"
            self.next_foreign_ground += 1
        else:
            new_id = f"foreign_air_{self.next_foreign_air}"
            self.next_foreign_air += 1
        entity = Entity(new_id, entity_obs.type, entity_obs.disposition, entity_obs.health, entity_obs.x, entity_obs.y, entity_obs.z, self.current_tick)
        self.foreign_entities[new_id] = entity
        self.foreign_id_to_int[new_id] = self.next_int
        self.foreign_int_to_id[self.next_int] = new_id
        self.foreign_grid[entity.y, entity.x] = self.next_int
        self.next_int += 1
        # add enemy discovery event
        event = Event(
            tick=self.current_tick,
            type=EventType.ENEMY_DISCOVERY,
            source_id=None,
            target_id=new_id,
            x=entity_obs.x,
            y=entity_obs.y
        )
        self.events.append(event)

    def _expire_foreign_entities(self):
        to_delete = []
        for entity_id, entity in self.foreign_entities.items():
            if self.current_tick - entity.last_seen_tick > self.ttl:
                to_delete.append(entity_id)
        for entity_id in to_delete:
            entity = self.foreign_entities[entity_id]
            id_int = self.foreign_id_to_int[entity_id]
            del self.foreign_int_to_id[id_int]
            del self.foreign_id_to_int[entity_id]
            # remove from grid
            self.foreign_grid[entity.y, entity.x] = -1
            del self.foreign_entities[entity_id]

    def add_event(self, event: Event):
        if event.type == EventType.FRIENDLY_ATTACK:
            # if friendly attack, try to resolve w.r.t. (existing) entities
            engager_type = self.known_entities[event.source_id].type
            engage_radius = settings.GROUND_OBS_RADIUS if engager_type == "ground" else settings.AIR_OBS_RADIUS
            closest_entity = None
            closest_dist_sq = engage_radius * engage_radius
            for entity in self.foreign_entities.values():
                # basically checking the distance from entities to targetted pos
                dx = entity.x - event.x
                dy = entity.y - event.y
                dist_sq = dx * dx + dy * dy

                if dist_sq < closest_dist_sq:
                    closest_dist_sq = dist_sq
                    closest_entity = entity
            event.target_id = closest_entity.id if closest_entity is not None else None
        elif event.type == EventType.ENEMY_ATTACK:
            # update the agent's health
            target_id = event.target_id
            target_agent = self.known_entities[target_id]
            damage = event.metadata["damage"]
            if target_agent.health > 0:
                target_agent.health -= damage
            # clamp health to 0
            if target_agent.health < 0:
                target_agent.health = 0
        elif event.type == EventType.FRIENDLY_ELIMINATE:
            # set the agent's health to 0
            target_id = event.target_id
            target_agent = self.known_entities[target_id]
            target_agent.health = 0
        self.events.append(event)

    def _expire_events(self):
        self.events = [e for e in self.events if self.current_tick - e.tick <= self.ttl]

    def step(self):
        self._expire_foreign_entities()
        self._expire_events()
        self.current_tick += 1
