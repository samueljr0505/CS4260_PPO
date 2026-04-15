import torch
class PPO:
    def __init__(
        self,
        model,
        lr=3e-4,
        gamma=0.99,
        clip=0.2,
        epochs=10,
        batch_size=256,
        value_coef=0.5,
        entropy_coef=0.01
    ):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)

        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def update(self, buffer, next_value):

        obs = torch.stack(buffer.obs)
        actions = torch.stack(buffer.actions)
        old_logprobs = torch.stack(buffer.logprobs)

        returns, advantages = buffer.compute_gae(next_value)

        n = len(obs)

        for _ in range(self.epochs):
            idx = torch.randperm(n)

            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]

                b_obs = obs[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logp = old_logprobs[batch_idx]
                b_adv = advantages[batch_idx]
                b_ret = returns[batch_idx]

                dist = self.model.get_dist(b_obs)
                logp = dist.log_prob(b_actions)
                entropy = dist.entropy()

                ratio = torch.exp(logp - b_old_logp)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * b_adv

                actor_loss = -torch.min(surr1, surr2).mean()

                value = self.model.value(b_obs)
                value_loss = ((b_ret - value) ** 2).mean()

                loss = (
                    actor_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()