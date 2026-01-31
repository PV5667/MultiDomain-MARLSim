from smart import SMART, Entity
from agents import GroundAgent, AirAgent

class Swarm:
    """
    Wrapper class for the swarm of agents.
    This is mainly to facilitate inter-agent and SMART communication.

    Also makes it easier for the env to step forward.
    """
    def __init__(self, agent_pos):
        self.agents = [] # list of Agent()
        self.smart = SMART()

        # init each of the n_agents

    def step(self):
        pass
