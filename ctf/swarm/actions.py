"""
Defining the actions that agents can take
There are multiple kinds of actions, e.g. engage, move, deploy.
"""
from enum import Enum
from dataclasses import dataclass
from agent import AgentStatus

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
class MoveAction:
    agent_status: AgentStatus
    direction: Direction
    magnitude: int

@dataclass
class EngageAction:
    agent_status: AgentStatus
    target_x: int
    target_y: int


@dataclass
class DeployAction:
    agent_status: AgentStatus
    deploy_type: str = "cUAS" # for now, maybe have a counter-ground in the future?
    # deploys one tile right in front of the agent