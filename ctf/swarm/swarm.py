from swarm.smart import SMART, Entity, FlagEntity, Disposition, Event, EventType
from swarm.agent import GroundAgent, AirAgent, AgentStatus
from env.feedback_message import FeedbackMessage
from collections import defaultdict

class Swarm:
    """
    Wrapper class for the swarm of agents.
    This is mainly to facilitate inter-agent and SMART communication.

    Also makes it easier for the env to step forward.
    """
    def __init__(self, flag_pos, agent_pos, swarm_id: int):
        self.agents = [] # list of Agent()
        self.smart = SMART()
        self.rewards = defaultdict(float)
        self.current_tick = 0
        # init each of the agents in agent_pos
        # assign unique id to each agent (unique on global level)
        self.n_ground_agents = 0
        self.n_air_agents = 0
        for pos in agent_pos:
            agent_type, x, y = pos
            if agent_type == "ground":
                agt_id = f"{swarm_id}_ground_{self.n_ground_agents + 1}"
                agent_type = "ground"
                status = AgentStatus(agt_id, agent_type, x, y, 0, 100) # no z at the moment...
                agent = GroundAgent(status)
                self.smart.known_entities[agt_id] = Entity(agt_id, "ground", Disposition.FRIENDLY, status.health, status.x, status.y, status.z)
                self.n_ground_agents += 1
            else:
                agt_id = f"{swarm_id}_air_{self.n_air_agents + 1}"
                agent_type = "air"
                status = AgentStatus(agt_id, agent_type, x, y, 0, 100) # no z at the moment...
                agent = AirAgent(status)
                self.smart.known_entities[agt_id] = Entity(agt_id, "air", Disposition.FRIENDLY, status.health, status.x, status.y, status.z)
                self.n_air_agents += 1
            self.agents.append(agent)
        
        for i, pos in enumerate(flag_pos):
            x, y = flag_pos
            self.smart.known_entities[f"flag_{i}"] = FlagEntity(f"flag_{i}", x, y, 0.0)

    def step(self, environment):
        # environment is the ground truth array -- it's passed in at every swarm step since it keeps updating
        actions = []
        # iterate through all of the agents and get their actions
        for i in range(len(self.agents)):
            agent = self.agents[i]
            # TODO get env_patch to pass to each agent -- base this off observation
            action = agent.get_action(None)
            actions.append(action)
        self.current_tick += 1
        return actions
    
    def receive_feedback(self, feedback):
        # feedback is a list of FeedbackMessages
        # send to SMART when necessary

        for msg in feedback:
            msg_type = msg.action_type
            if msg_type == "move":
                self.rewards[msg.agent_id] += msg.details["reward"]
                # update agent position in SMART
                self.smart.update_known_entity_pos(msg.agent_id, msg.details["new_x"], msg.details["new_y"])
            elif msg_type == "engage":
                # either register damage dealt or received
                engager_id = msg.details["engager_id"]
                target_id = msg.details[target_id]
                target_x, target_y = msg.details["target_x"], msg.details["target_y"]
                damage = msg.details["damage"]
                if self.swarm_id == engager_id[0]:
                    # means damage was dealt
                    event = Event(
                        tick=self.current_tick,
                        type=EventType.FRIENDLY_ATTACK,
                        source_id=engager_id,
                        x=target_x,
                        y=target_y,
                        metadata={"damage": damage}
                    )
                    self.smart.add_event(event)
                if self.swarm_id == target_id[0]:
                    # means damage was received
                    event = Event(
                        tick=self.current_tick,
                        type=EventType.ENEMY_ATTACK,
                        target_id = target_id,
                        x=target_x,
                        y=target_y,
                        metadata={"damage": damage}
                    )
                    self.smart.add_event(event)
            elif msg_type == "engage_reward":
                self.rewards[msg.agent_id] += msg.details["reward"]
            elif msg_type == "deploy":
                pass
            elif msg_type == "capture":
                # add capture info to SMART? too much?
                agent_id = msg.details["agent_id"]
                inc = msg.details["inc"]
                flag_id = f"flag_{msg.details['flag_idx']}"
                event = Event(
                        tick=self.current_tick,
                        type=EventType.FLAG_CAPTURE,
                        source_id = agent_id,
                        target_id = flag_id, # global flag id matches with smart id
                        metadata={"inc": inc}
                    )
                self.smart.add_event(event)
            elif msg_type == "capture_reward":
                self.rewards[msg.agent_id] += msg.details["reward"]

        
