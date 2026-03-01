"""
Defining the actions that agents can take
There are multiple kinds of actions, e.g. engage, move, deploy.
"""
from enum import Enum

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


class Action:
    def __init__(self, params):
        self.params = params  # params specific to action type

    # probably doing action execution at the environment level
    def execute(self, agent, environment):
        # Logic to execute the action with the given agent within the environment
        # Update agent status
        # Update ground truth environment
        # Update SMART on result of action
        pass

class MoveAction(Action):
    def __init__(self, agent_status, direction: Direction, magnitude: int):
        self.agent_status = agent_status
        self.direction = direction
        self.magnitude = magnitude
        
class EngageAction(Action):
    def __init__(self, agent_status, target_x, target_y):
        self.agent_status = agent_status
        self.target_x = target_x
        self.target_y = target_y

class DeployAction(Action):
    def __init__(self):
        self.deploy_type = "cUAS" # for now, maybe have a counter-ground in the future?
        # deploys one tile right in front of the agent