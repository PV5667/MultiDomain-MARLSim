import numpy as np
from swarm.agent import Agent, AgentStatus, GroundAgent, AirAgent
from env.flag import Flag
from enum import Enum
from dataclasses import dataclass
from constants import settings
from typing import Optional, Dict, Any

class Disposition(Enum):
    """Currently supports Friendly, Enemy. Future iterations may support Unknown."""
    FRIENDLY = 1
    ENEMY = 2

class EventType(Enum):
    """TODO: Think more about the event types."""
    FLAG_CAPTURE = 1
    ENEMY_DISCOVERY = 2
    FRIENDLY_ATTACK = 3 # Friendly attacks enemy (including self)
    ENEMY_ATTACK = 4 # Enemy attacks friendly (this indicates damage dealt, not enemy id etc.)
    FRIENDLY_ELIMINATE = 5 # friendly agent eliminated


@dataclass
class FlagEntity:
    # swarms should have their own representation of flags
    id: str
    x: int
    y: int
    disposition: float # negative if hostile, positive if friendly


@dataclass
class Entity:
    id: int
    type: str
    disposition: Disposition
    health: int
    x: int
    y: int
    z: int
    last_seen_tick: int = 0

@dataclass
class EntityObservation:
    tick: int
    type: str
    disposition: Disposition
    health: int
    x: int
    y: int
    z: int

@dataclass
class Event:
    """Represents various events that occur, for agents and humans alike to gain context on the battlespace."""
    tick: int
    type: EventType
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

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
        self.current_tick = 0
        self.ttl = ttl # time-to-live for entities
        self.known_entities = {} # initialized with friendly agents + flags
        self.foreign_entities = {}
        self.events = []
        
        self.foreign_grid = np.full((height, width), -1)
        self.relevance_radius = 50
        
        self.event_grid = np.full((height, width), -1)
        
        self.next_foreign_air = 0
        self.next_foreign_ground = 0
        self.next_int = 0

        self.foreign_id_to_int = {}
        self.foreign_int_to_id = {}

    def publish(self, agent: Agent):
        # basically look at the position of the agent, along with tick, and output info accordingly
        obs = {} # consists of events and entities
        agent_status = agent.status
        x, y = agent_status.x, agent_status.y
        # spatial index: all entities within 100x100 block
        x0 = max(0, x - self.foreign_grid_radius)
        x1 = min(self.width, x + self.foreign_grid_radius + 1)
        y0 = max(0, y - self.foreign_grid_radius)
        y1 = min(self.height, y + self.foreign_grid_radius + 1)
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
            dx = entity.x - x
            dy = entity.y - y
            if dx*dx + dy*dy <= self.relevance_radius*self.relevance_radius:
                known_entities.append(entity)
        obs["flags"] = flags
        obs["known_entities"] = known_entities
        # getting relevant events
        relevant_events = self._get_relevant_events(x, y)
        obs["relevant_events"] = relevant_events
    
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
    
    def add_entity_observation(self, observation: EntityObservation):
        candidate = self._find_matching_entity(observation)
        if candidate:
            # update candidates position
            self._update_foreign_entity(candidate, observation)
        else:
            self._add_foreign_entity(observation)

    def _find_matching_entity(self, observation: EntityObservation):
        match_radius = settings["GROUND_SPEED"] if observation.type == "ground" else settings["AIR_SPEED"]
        closest_entity = None
        closest_dist_sq = match_radius * match_radius

        for entity in self.foreign_entities.values():
            if entity.type != observation.type:
                continue
            dx = entity.x - observation.x
            dy = entity.y - observation.y
            dist_sq = dx * dx + dy * dy

            if dist_sq < closest_dist_sq:
                closest_dist_sq = dist_sq
                closest_entity = entity
        return closest_entity
    
    def update_known_entity_pos(self, entity_id, new_x, new_y):
        entity = self.known_entities[entity_id]
        entity.x = new_x
        entity.y = new_y

    def _update_foreign_entity(self, entity: Entity, entity_obs: EntityObservation):
        old_x, old_y = entity.x, entity.y
        entity.last_seen_tick = self.current_tick
        entity.x = entity_obs.x
        entity.y = entity_obs.y
        entity.z = entity_obs.z
        prev_val = self.foreign_grid[old_y, old_x]
        self.foreign_grid[entity.y, entity.x] = prev_val
        self.foreign_grid[old_y, old_x] = -1
    
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
            entity = self.foreign_grid[entity_id]
            self.foreign_grid[entity.y, entity.x] = -1
            del self.foreign_entities[entity_id]

    def add_event(self, event: Event):
        if event.type == EventType.FRIENDLY_ATTACK:
            # if friendly attack, try to resolve w.r.t. (existing) entities
            engager_type = self.known_entities[event.source_id].type
            engage_radius = settings["GROUND_OBS_RADIUS"] if engager_type == "ground" else settings["AIR_OBS_RADIUS"]

            closest_entity = None
            closest_dist_sq = engage_radius * engage_radius
            for entity in self.foreign_entities.values():
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
            target_agent = self.known_entities[target_agent]
            target_agent.health = 0
        self.events.append(event)

    def _expire_events(self):
        self.events = [e for e in self.events if self.current_tick - e.tick <= self.ttl]

    def step(self):
        self._expire_foreign_entities()
        self._expire_events()
        self.current_tick += 1
