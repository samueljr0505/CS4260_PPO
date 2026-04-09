import torch
import torch.nn as nn
from torch.distributions import Categorical

class MultiAgentActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, state_dim, centralized_critic=True):
        super().__init__()

        self.centralized_critic = centralized_critic

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
        )

        # If centralized: critic sees global state (state_dim)
        # If ablated:     critic sees local obs only (obs_dim) — same as basic PPO
        critic_input_dim = state_dim if centralized_critic else obs_dim

        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, 256),
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

    def value(self, state_or_obs):
        return self.critic(state_or_obs).squeeze(-1)

    def evaluate(self, obs, state, action):
        dist = self.policy(obs)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        # Use state if centralized, local obs if ablated
        critic_input = state if self.centralized_critic else obs
        value = self.value(critic_input)
        return logprob, value, entropy