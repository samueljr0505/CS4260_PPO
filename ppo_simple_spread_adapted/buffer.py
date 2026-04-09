import torch


class MultiAgentRolloutBuffer:
    def __init__(self):
        self.obs = []
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, state, action, logprob, reward, done, value):
        self.obs.append(obs)
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(value if isinstance(value, float) else float(value))

    def as_tensors(self):
        return (
            torch.stack(self.obs),
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.logprobs),
        )

    def compute_returns_and_advantages(self, next_value=0.0, gamma=0.99, lam=0.95):
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        values = torch.tensor(self.values + [float(next_value)], dtype=torch.float32)

        gae = 0.0
        advantages = []
        returns = []

        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
            gae = delta + gamma * lam * non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        returns_t = torch.tensor(returns, dtype=torch.float32)
        advantages_t = torch.tensor(advantages, dtype=torch.float32)

        # Normalize returns for critic stability, but keep raw for logging
        returns_normalized = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        return returns_normalized, advantages_t

    def clear(self):
        self.__init__()