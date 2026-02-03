# MultiDomain-MARLSim
Experimenting with MARL techniques and multi-domain autonomy


This is going to be a project exploring swarm autonomy. The initial idea I have is to implement a capture-the-flag type of scenario, with two teams working against each other. Each team will consist of multiple agents, each communicating with each other. These agents will be heterogeneous -- at the moment, I am aiming to have two kinds of agents, air and land. The following spec details the expected deliverables/capabilities of this project.


### Environment
- For feasibility (my current compute resources are an M3 Mac and Colab Pro subscription), I will implement a grid-based 2D environment with an attached height map (this will be randomly generated).
- 2D environment size will be fairly big (4096 x 4096 to start)

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
    - Event inputs: Key temporal events, handled by GNN
- Sourcing of info for different inputs
    - 2D grid inputs: from environment (ground-truth)
    - Entity inputs: **within range** from env (ground-truth), **outside range** from SMART
    - Event inputs: Handled from SMART, fixed number provided based on prioritization scores from distance and severity.
    - Internal State Vectors: **agent health** from env (ground-truth), **goal** from SMART
    - Inter-agent Vectors: Communicated between agents
- Then combine the resulting embeddings and input to MLP, which ouptuts comms and actions.
- Actions will be done in a factored manner, there will be multiple heads for each kind of action
    - Movement: This will be 2 heads. One for direction (e.g. 8 directions), and another for magnitude (this is also discrete, from 1...max_range). In the direction head there will also be a NOOP category (e.g. if the agent is capturing a flag).
    - Deploy: At the moment this is for ground vehicles, and only cUAS can be deployed. This is a simple binary head.
    - Engage: This is slightly more complicated, since we need to support dynamic numbers of targets. So for this class of action, we will also have 2 heads. One is focused on whether to engage or not (a binary decision), and another is focused on which entity to engage. This second head will be based on the entity GNN. Each of the outputted vectors from the GNN will be scored (only if they are enemy and within range), and then we run softmax to determine which entity to engage.
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


Rough sketch of the overall flow:
1. Environment object is initialized, and generates N environments with the seeds provided by the user.
2. Then run env.reset() to trigger agent and flag initialization. These are randomly initialized each time.
3. Initialize the Swarm objects with the agent positions. TODO: initializing Flag objects.
4. Swarm objects create individual agent instances and assign unique id's. They also initialize a SMART instance (these are representative of the situational awareness of the swarm as a whole).
5. Call env.step(), this results in (1) observations being calculated and sent to each agent, (2) agents updating SMART if necessary, (3) agents computing decision (e.g. running the policy network), (4) actions being sent back to the environment level for updates.
6. Basically any read operations (e.g. computing observations) can be done by the Agents, but actual updates to the environment ground truth array need to be done by the environment itself.


Actions Overall Structure:
- There are three kinds of actions so far:
    - Movement: Two aspects to this. One is direction, another is magnitude. Also needs to be done within physical constraints. So if slope is too big, the action results in nothing (in the future, can maybe look at reduced movement?)
    - Engage: Get the entity id and run the engagement. Done and scaled based on a stochastic distribution. Make update to the "damage map" stored by the environment. Once all actions have updated environment (and the damage map), the damage calculation and updates will be done by the environment in a separate pass.
    - Deploy: Doesn't make sense to create another object for this. Add an entity to the environment ground-truth array. For simplicity assuming immediate deployment. Then, in the next step/so on, after all actions have been enacted, if any air agent is within range of the cUAS, deal damage or take it out.


Relaying Engage Action Feedback:
- It doesn't make sense that an agent engages a target and doesn't get any feedback on if it succeeded or not. In the future this will also be important for reward function calculations.
- Right now I calculate damage by adding it to a global "damage map". And then after all actions have been executed (motion, deployment included), my plan is to iterate through all agents and make the necessary updates to their health.
- I need to have a way to attribute damage to certain agents though, and then pass information on success/not as part of observations in the next step.
- Possibly have the damage map be multichannel. Each channel corresponds to an agent id (agentid). Each swarm will have its own damage map
- Iterating through agents, updating health, and recording obs for next step should be done at the swarm level. Call swarm.damage_calc(damage_map)
- swarm.damage_calc does multiple things. Calculates damage-based reward
    - Reward positive if hit enemy. Reward negative if hit friendly.
    - On a separate tangent, should I calculate rewards at the swarm-level? To my understanding, there are really two policies I'm training, a ground policy and an air policy...

