import torch
import torch.nn as nn
from torch.distributions import Categorical


class MultiAgentActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, state_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def policy(self, obs):
        logits = self.actor(obs)
        return Categorical(logits=logits)

    def act(self, obs):
        dist = self.policy(obs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def value(self, state):
        return self.critic(state).squeeze(-1)

    def evaluate(self, obs, state, action):
        dist = self.policy(obs)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value(state)
        return logprob, value, entropy
