import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def get_dist(self, obs):
        logits = self.actor(obs)
        return Categorical(logits=logits)

    def get_action(self, obs):
        dist = self.get_dist(obs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def value(self, obs):
        return self.critic(obs).squeeze(-1)