from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class Direction(Enum):
    """Relative to each swarm's overall direction. North = Forward."""
    NORTH=1
    NORTHEAST=2
    EAST=3
    SOUTHEAST=4
    SOUTH=5
    SOUTHWEST=6
    WEST=7
    NORTHWEST=8

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
    disposition: float = 0.0

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
