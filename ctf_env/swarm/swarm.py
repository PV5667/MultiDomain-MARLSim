from smart import SMART

class Swarm:
    """Wrapper class for the swarm of agents"""
    def __init__(self):
        self.agents = []
        self.smart = SMART()
    