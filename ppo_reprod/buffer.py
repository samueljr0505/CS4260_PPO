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
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns(self, gamma=0.99, lam=0.95):
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values + [0.0])
        dones = torch.tensor(self.dones, dtype=torch.float32)

        gae = 0
        returns = []

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            returns.insert(0, gae + values[t])

        return torch.tensor(returns)

    def clear(self):
        self.__init__()