import numpy as np
from swarm.agent import Agent, AgentStatus, GroundAgent, AirAgent
from enum import Enum

class Disposition(Enum):
    """Currently supports Friendly, Enemy. Future iterations may support Unknown."""
    FRIENDLY = 1
    ENEMY = 2


class EventType(Enum):
    """TODO: Think more about the event types."""
    FLAG_CAPTURE = 1
    ENEMY_DISCOVERY = 2
    FRIENDLY_ATTACK = 3 # Friendly attacks enemy (including self)
    ENEMY_ATTACK = 4 # Enemy attacks friendly (this indicates damage dealt, not enemy id etc.) -- includes self

class Entity:
    def __init__(self, id: str, type: Agent, disp: Disposition, status: AgentStatus):
        """Denotes active entity in environment."""
        self.id = id
        self.type = type
        self.disp = disp
        self.health = status.health
        self.x = status.x
        self.y = status.y
        self.z = status.z

class Event:
    def __init__(self, tick: int, type: EventType, agt_id: str, priority, **kwargs):
        """Represents various events that occur, for agents and humans alike to gain context on the battlespace."""
        self.tick = tick
        self.priority = priority # priority is defined by a combo of event type AND tick
        

class SMART:
    """
    SMART (Swarm-based Multi-agent Awareness & Real-time Tracking)
    
    Entities: Dictionary based on agent id
    Events: Dictionary based on timestamp
    Events PQ: TODO priority queue for events
    Spatial Index: TODO some kind of spatial index, has both events and entities.

    Note: SMART is a *swarm-specific* model that acts as a global information repository. It is updated as agents observe entities and events.
    """
    def __init__(self):
        self.entities = {}
        self.events = None
        self.spatial_index = None
        pass

    def publish(self, agt: Agent, tick: int):
        # basically look at the position of the agent, along with tick, and output info accordingly
        pass

    def add_entity(self, entity: Entity):
        # observation-oriented
        pass

    def add_event(self, event: Event):
        # action-oriented
        pass

