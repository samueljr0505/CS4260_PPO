import torch
import torch.nn as nn

class PPO:
    def __init__(self, model, lr=3e-4, clip=0.2):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip = clip

    def update(self, buffer, epochs=10):
        obs = torch.stack(buffer.obs)
        actions = torch.stack(buffer.actions)
        old_logprobs = torch.stack(buffer.logprobs)
        returns = buffer.compute_returns()
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(epochs):
            logprobs, values, entropy = self.model.evaluate(obs, actions)

            ratio = torch.exp(logprobs - old_logprobs)

            adv = returns - values.detach()

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = ((returns - values) ** 2).mean()
            entropy_loss = -entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()