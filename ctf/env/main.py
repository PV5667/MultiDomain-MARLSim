from ctf_env import CTFEnv
from swarm.swarm import Swarm
from constants import settings

height = 1024
width = 2048

env = CTFEnv(height=height, width=width, n_ground_agents=settings["N_GROUND_AGENTS"], n_air_agents=settings["N_AIR_AGENTS"], n_flags=settings["N_FLAGS"], seed_range=[4])
env.reset()

print("Getting Swarm 1 Agent Positions")
swarm1_agent_pos = env._agent_pos("swarm1")
print("Getting Swarm 2 Agent Positions")
swarm2_agent_pos = env._agent_pos("swarm2")

env.init_env_entities()

swarm1 = Swarm(height, width, env.flag_pos, swarm1_agent_pos, 1)
swarm2 = Swarm(height, width, env.flag_pos, swarm2_agent_pos, 2)

for _ in range(150):
    actions_swarm1 = swarm1.step(env)
    actions_swarm2 = swarm2.step(env)
    actions = {"swarm1": actions_swarm1, "swarm2": actions_swarm2}
    env.step(actions)
    swarm1.receive_feedback(env.swarm1_feedback)
    swarm2.receive_feedback(env.swarm2_feedback)
    
