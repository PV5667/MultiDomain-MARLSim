"""
Defining the actions that agents can take
There are multiple kinds of actions, e.g. engage, move, deploy.
"""


class Action:
    def __init__(self, action_type, params):
        self.action_type = action_type  # e.g., "move", "engage", "deploy"
        self.params = params  # params specific to action type

    def execute(self, agent, environment):
        # Logic to execute the action with the given agent within the environment
        # Update agent status
        # Update ground truth environment
        # Update SMART on result of action
        pass



class MoveAction(Action):
    def __init__(self, target_x, target_y):
        super().__init__("move", {"target_x": target_x, "target_y": target_y})

    def execute(self, agent, environment):
        # Logic to move the agent to (target_x, target_y)
        pass

class EngageAction(Action):
    def __init__(self, target_agent_id):
        super().__init__("engage", {"target_agent_id": target_agent_id})

    def execute(self, agent, environment):
        # Logic to engage with the target agent
        pass

class DeployAction(Action):
    def __init__(self, deploy_type, location):
        super().__init__("deploy", {"deploy_type": deploy_type, "location": location})

    def execute(self, agent, environment):
        # Logic to deploy cUAS at specified location
        pass