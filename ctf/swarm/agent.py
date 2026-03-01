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
    def calculate_observation(self, env_patch):
        # calculates the env. ground truth observation for the agent
        pass
    def preprocess_obs(self, env_patch):
        # basically packages inputs for the policy network
        # grid inputs: calculate_observation
        # smart inputs: query_smart
        pass
    def get_action(self, env_patch):
        # env_patch is localized to the fixed area around the agent (patching done during swarm.step())
        # call preprocess_obs
        # call to the model
        pass
    def query_smart(self):
        # call to SMART to get relevant info
        """Query SMART given state"""
        pass
    def publish_smart(self):
        """Report to SMART"""
        pass
    def update_health(increment):
        # add validity checks
        pass
    
class GroundAgent(Agent):
    def __init__(self, status: AgentStatus):
        super().__init__()
        self.status = status
    def calculate_observation(self, env_patch):
        # this is ground, so need to do LoS calculations to determine what can/cannot be seen.
        pass
    def preprocess_obs(self, env_patch):
        pass
    def get_action(self, env_patch):
        return MoveAction(self.status, Direction.NORTH, settings.GROUND_SPEED)

class AirAgent(Agent):
    def __init__(self, status: AgentStatus):
        super().__init__()
        self.status = status
    def calculate_observation(self, env_patch):
        # this is air, so just return the fixed NxN patch around the air agent.
        return env_patch
    def preprocess_obs(self, env_patch):
        pass
    def get_action(self, env_patch):
        return MoveAction(self.status, Direction.NORTH, settings.AIR_SPEED)