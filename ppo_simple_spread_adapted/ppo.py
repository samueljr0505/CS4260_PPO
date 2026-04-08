import torch


class MultiAgentPPO:
    def __init__(
        self,
        model,
        lr=3e-4,
        clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip = clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def update(self, buffer, next_state, batch_size=256, epochs=10):
        """
        next_state: the global state at the end of the rollout, used to
        bootstrap value for truncated (non-terminal) episodes.
        """
        if not buffer.obs:
            return

        # Bootstrap next value from the final state if the rollout was truncated
        with torch.no_grad():
            next_value = self.model.value(next_state).item()

        obs, states, actions, old_logprobs = buffer.as_tensors()
        returns, advantages = buffer.compute_returns_and_advantages(next_value=next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_samples = obs.shape[0]

        for _ in range(epochs):
            permutation = torch.randperm(total_samples)

            for start in range(0, total_samples, batch_size):
                indices = permutation[start : start + batch_size]

                batch_obs = obs[indices]
                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_old_logprobs = old_logprobs[indices]
                batch_returns = returns[indices]
                batch_advantages = advantages[indices]

                logprobs, values, entropy = self.model.evaluate(
                    batch_obs, batch_states, batch_actions
                )

                ratio = torch.exp(logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * batch_advantages
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                # Clipped value loss (standard PPO improvement)
                critic_loss = (batch_returns - values).pow(2).mean()
                entropy_bonus = entropy.mean()

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy_bonus
                )

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.opt.step()