from swarm.smart import SMART, Entity
from swarm.agents import GroundAgent, AirAgent

class Swarm:
    """
    Wrapper class for the swarm of agents.
    This is mainly to facilitate inter-agent and SMART communication.

    Also makes it easier for the env to step forward.
    """
    def __init__(self, agent_pos):
        self.agents = [] # list of Agent()
        self.smart = SMART()

        # init each of the agents in agent_pos
        # assign unique id to each agent
        self.n_ground_agents = 0
        self.n_air_agents = 0
        for pos in agent_pos:
            agent_type, x, y = pos
            if agent_type == "ground":
                agt_id = f"ground_{self.n_ground_agents}"
                agent = GroundAgent(agt_id, x, y)
                self.n_ground_agents += 1
            else:
                agt_id = f"air_{self.n_air_agents}"
                agent = AirAgent(agt_id, x, y)
                self.n_air_agents += 1
            self.agents.append(agent)

    def step(self):
        pass
