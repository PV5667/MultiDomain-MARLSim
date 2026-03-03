"""
Defining the actions that agents can take
There are multiple kinds of actions, e.g. engage, move, deploy.
"""
from enum import Enum
from dataclasses import dataclass
from swarm.core import *

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