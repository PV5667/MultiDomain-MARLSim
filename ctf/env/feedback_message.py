# Messages sent from the environment class back to the swarm after actions executed.

"""
Different types of feedback corresponding to the type of action

Move: info about updated position
Engage: info about engagement
Engage Reward: reward assoc. with engage action
Deploy: success/not (for now always successful deployment)

"""
class FeedbackMessage:
    def __init__(self, agent_id, action_type, details):
        self.agent_id = agent_id
        self.action_type = action_type # "move", "engage", "engage_reward", "deploy"
        self.details = details # parse based on action type
    