# MultiDomain-MARLSim
Experimenting with MARL techniques and multi-domain autonomy


This is going to be a project exploring swarm autonomy. The initial idea I have is to implement a capture-the-flag type of scenario, with two teams working against each other. Each team will consist of multiple agents, each communicating with each other. These agents will be heterogeneous -- at the moment, I am aiming to have two kinds of agents, air and land. The following spec details the expected deliverables/capabilities of this project.


### Environment
- For feasibility (my current compute resources are an M3 Mac and Colab Pro subscription), I will implement a grid-based 2D environment with an attached height map (this will be randomly generated).
- 2D environment size will be fairly big (4096 x 4096 to start)


### Observation Space


### Action Space



### Agent Specs:
- Ground agent: An autonomous ground vehicle, with fast flag capture and cUAS deployment capability.
    - Mobility per sec: 3 tiles
    - Observation Capability: 20 tiles in all directions, or as determined by height map.
    - Health: 200 points
    - Attack: 50 points, ground-only, attempted once per second. Attempt success determined by P(success), and damage is scaled based on stochastic damage distribution (between [0, 1], skewed to 1).
    - Flag capture speed: 20% per second
    - Communication: 
    - Can deploy counter-UAS systems (25 health) (takes 10 secs). These will be able to take out air assets within a certain radius with 75% probability.
- Air agent: Drone with fast mobility and heightened visibility. Slower flag capture and lower health.
    - Mobility per sec: 10 tiles
    - Observation Capability: 100 tiles in all directions
    - Health: 100 points
    - Attack: 25 points, attempted once per second. Attempt success determined by P(success), and damage is scaled based on stochastic damage distribution (between [0, 1], skewed to 1).
    - Flag capture speed: 10% per second

**Communication capabilities detailed below.**



### Communication Model:
- The core idea of this project is to construct resilient networks of communicating agents, allowing for robust situational awareness and decision-making.
- There are two main types of communication in this project: inter-agent and global.
    - Inter-agent communication is learned and meant to aid in collaboration between agents.
    - Global communication: these are deterministic reports that are published by agents to SMART, which is a global situational awareness layer that compiles all info collected by the swarm.


### Swarm-based Multi-agent Awareness & Real-time Tracking (SMART):
- This is a situational awareness layer that is contributed to and updated by the individual agents.
- Essentially this acts as a global tracker of state, and also could be visualized for human operators.
- Each swarm gets its own SMART, it will contain info like friendly asset locations & health, enemy asset locations & health, flag capture statuses, etc.


### Policy network design
- Good mental model for the observation space of each agent:
    - 2D grid inputs: terrain -- handled by CNN
    - Entity inputs: entities (e.g. flags, agents) -- handled by GNN
    - Internal State Vectors: Goal, Health -- handled by MLP
    - Inter-agent Vectors: Learned communication between agents, handled by MLP
- Sourcing of info for different inputs
    - 2D grid inputs: from environment (ground-truth)
    - Entity inputs: **within range** from env (ground-truth), **outside range** from SMART
    - Internal State Vectors: **agent health** from env (ground-truth), **goal** from SMART
    - Inter-agent Vectors: Communicated between agents
- Then combine the resulting embeddings and input to MLP, which ouptuts comms and actions.
- There are two kinds of comms:
    - Inter-agent comms (learned)
    - SMART comms (deterministic) -- these are structured reports published to SMART


Three different statuses for the swarm
- Green: move forward and capture the points (assumed safe env.)
- Yellow: scouting, move forward cautiously. Purpose is to gain situational awareness. If engaged, transition to red/green (dependent on specification from human operator).
- Red: capture points, destroy all enemy assets
- **Training Note:** Encode different reward schemes + capability restrictions for each method.

Improvements for future iterations:
- Having different dispositions for entities, e.g. red (enemy), yellow (unknown), or green (friendly)
    - Determining these dispositions gets into keeping records for past actions done by entities (e.g. movements or engagements with friendly assets) to determine intent!
- (Pretty feasible) Making a UI interface for SMART:
    - Goal-setting capabilities (setting flags/capture points on a map)
    - Status-setting for the swarm (red, yellow, green)
    - Real-time alerts bar denoting: flag captures, engagements, friendly entity statuses, etc.

Overall Autonomy Stack:
- SMART is the high-level mission planner which also serves as the human interface.
- SMART provides objectives/statuses to Behavior Trees, which act as control flow and safety guards (e.g. limiting certain actions, forced retreat if health low, setting agent objectives/modes, etc.)
- Learned Policies are conditioned on observations, comms, and objectives, mainly focused on movement, engagement tactics, inter-agent comms, etc.