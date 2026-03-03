import torch

class MAPPOBuffer:
    def __init__(self, rollout_len, n_envs, n_agents, device):
        self.T = rollout_len
        self.B = n_envs
        self.A = n_agents
        self.device = device
        
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.action_idx_list = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, obs, actions, action_idx, log_probs, values, rewards, dones):
        self.obs.append(obs)
        self.actions.append(actions)
        self.action_idx_list(action_idx)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.rewards.append(rewards)
        self.dones.append(dones)

    def stack(self):
        self.log_probs = torch.stack(self.log_probs, dim=1)  # (B, T, A)
        self.values = torch.stack(self.values, dim=1)
        self.rewards = torch.stack(self.rewards, dim=1)
        self.dones = torch.stack(self.dones, dim=1)

def compute_gae(buffer, last_values, gamma=0.99, lam=0.95):
    B, T, A = buffer.values.shape
    advantages = torch.zeros((B, T, A), device=buffer.values.device)
    next_adv = torch.zeros((B, A), device=buffer.values.device)
    next_value = last_values

    for t in reversed(range(T)):
        mask = 1.0 - buffer.dones[:, t]  # (B, A)
        delta = buffer.rewards[:, t] + gamma * next_value * mask - buffer.values[:, t]
        next_adv = delta + gamma * lam * mask * next_adv
        advantages[:, t] = next_adv * mask
        next_value = buffer.values[:, t]

    returns = advantages + buffer.values
    return advantages, returns

def normalize_advantages(advantages):
    mean = advantages.mean()
    std = advantages.std()
    normalized_adv = (advantages - mean) / (std + 1e-8)
    return normalized_adv