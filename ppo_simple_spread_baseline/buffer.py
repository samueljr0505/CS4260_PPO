import torch


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, logprob, reward, done, value):
        self.obs.append(obs.detach())
        self.actions.append(action.detach())
        self.logprobs.append(logprob.detach())
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.detach())

    def compute_gae(self, next_value, gamma=0.99, lam=0.95):

        rewards = torch.stack(self.rewards)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        values = torch.stack(self.values + [next_value])

        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages.insert(0, gae)

        advantages = torch.stack(advantages)
        returns = advantages + values[:-1]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns.detach(), advantages.detach()

    def clear(self):
        self.__init__()