"""Defining the agent types, along with their action and observation spaces."""


class AgentStatus:
    """Internal State of the Agent"""
    def __init__(self, x, y, z, health):
        self.x = x
        self.y = y
        self.z = z
        self.health = health

class Agent:
    def __init__(self):
        self.observation_space = None
        self.action_space = None
        self.model = None
        self.status = None
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


    
class GroundAgent(Agent):
    def __init__(self, status: AgentStatus):
        super.__init__()
        self.status = status
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


