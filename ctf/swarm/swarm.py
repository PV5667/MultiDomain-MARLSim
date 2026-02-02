from swarm.smart import SMART, Entity
from swarm.agent import GroundAgent, AirAgent, AgentStatus

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
                agt_id = f"ground_{self.n_ground_agents + 1}"
                agent_type = "ground"
                status = AgentStatus(agt_id, agent_type, x, y, 0, 100) # no z at the moment...
                agent = GroundAgent(status)
                self.n_ground_agents += 1
            else:
                agt_id = f"air_{self.n_air_agents + 1}"
                agent_type = "air"
                status = AgentStatus(agt_id, agent_type, x, y, 0, 100) # no z at the moment...
                agent = AirAgent(status)
                self.n_air_agents += 1
            self.agents.append(agent)

    def step(self, environment):
        # environment is the ground truth array -- it's passed in at every swarm step since it keeps updating
        actions = []
        # iterate through all of the agents and get their actions
        for i in range(len(self.agents)):
            agent = self.agents[i]
            # TODO get env_patch to pass to each agent
            action = agent.get_action(None)
            actions.append(action)
        
        return actions