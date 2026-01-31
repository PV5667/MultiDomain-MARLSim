"""Defining the agent types, along with their action and observation spaces."""


class AgentState:
    """
    Internal State of the Agent
    Allows for lightweight replay + communication.
    """
    def __init__(self, id, x, y, z, health):
        self.id = id # this id is specific to the swarm
        self.x = x
        self.y = y
        self.z = z
        self.health = health
        # will probably add more (e.g. scout mode, engage mode when working with rules-based behavior)

class Agent:
    def __init__(self, status: AgentState):
        self.observation_space = None
        self.action_space = None
        self.model = None
        self.status = status
    def preprocess_obs(self):
        pass
    def get_action(self):
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
    def __init__(self):
        super.__init__()
    def preprocess_obs(self):
        pass
    def get_action(self):
        pass
    def query_smart(self):
        """Query SMART given state"""
        pass
    def publish_smart(self):
        """Report to SMART"""
        pass

class AirAgent(Agent):
    def __init__(self):
        super.__init__()
    def preprocess_obs(self):
        pass
    def get_action(self):
        pass
    def query_smart(self):
        """Query SMART given state"""
        pass
    def publish_smart(self):
        """Report to SMART"""
        pass


