"""
Defining the actions that agents can take
There are multiple kinds of actions, e.g. engage, move, deploy.
"""


class Action:
    def __init__(self, action_type, params):
        self.action_type = action_type  # e.g., "move", "engage", "deploy"
        self.params = params  # params specific to action type

    # probably doing action execution at the environment level
    def execute(self, agent, environment):
        # Logic to execute the action with the given agent within the environment
        # Update agent status
        # Update ground truth environment
        # Update SMART on result of action
        pass



class MoveAction(Action):
    def __init__(self, direction, magnitude):
        self.direction = direction
        self.magnitude = magnitude
        
class EngageAction(Action):
    def __init__(self, target_agent_id):
        self.target_agent_id = target_agent_id

class DeployAction(Action):
    def __init__(self):
        self.deploy_type = "cUAS" # for now, maybe have a counter-ground in the future?
        # deploys one tile right in front of the agent