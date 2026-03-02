"""Defining the agent types, along with their action and observation spaces."""
from swarm.actions import Action, MoveAction, EngageAction, DeployAction, Direction
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
        self.observation_space = None
        self.action_space = None
        self.model = None
        self.status = None
    def process_env_patch(self, env_patch):

        pass
    def process_smart_obs(self, smart_obs):
        # convert to GNN inputs
        pass
    def get_action(self, env_patch, smart_obs):
        # env_patch is localized to the fixed area around the agent (patching done during swarm.step())
        # call preprocess_obs
        # call to the model
        pass
    
class GroundAgent(Agent):
    def __init__(self, status: AgentStatus):
        super().__init__()
        self.status = status
    def get_action(self, env_patch):
        return MoveAction(self.status, Direction.NORTH, settings.GROUND_SPEED)

class AirAgent(Agent):
    def __init__(self, status: AgentStatus):
        super().__init__()
        self.status = status
    def get_action(self, env_patch):
        return MoveAction(self.status, Direction.NORTH, settings.AIR_SPEED)