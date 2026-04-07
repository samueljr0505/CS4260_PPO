import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )

        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def get_action(self, obs):
        x = self.shared(obs)

        mean = self.actor(x)
        std = torch.exp(self.log_std)

        dist = torch.distributions.Normal(mean, std)

        action = dist.sample()

        # FIX: keep bounded for MPE env
        action = torch.sigmoid(action)

        logprob = dist.log_prob(action).sum(-1)

        return action, logprob

    def evaluate(self, obs, action):
        x = self.shared(obs)

        mean = self.actor(x)
        std = torch.exp(self.log_std)

        dist = torch.distributions.Normal(mean, std)

        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(x).squeeze(-1)

        return logprob, value, entropy