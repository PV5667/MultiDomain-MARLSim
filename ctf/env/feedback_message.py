# Messages sent from the environment class back to the swarm after actions executed.

from dataclasses import dataclass

"""
Different types of feedback corresponding to the type of action

Move: info about updated position
Engage: info about engagement
Engage Reward: reward assoc. with engage action
Deploy: success/not (for now always successful deployment)

"""

@dataclass
class FeedbackMessage:
    agent_id: str
    action_type: str # "move", "engage", "engage_reward", "deploy"
    details: dict # parse based on action type
    