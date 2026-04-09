import torch
class MultiAgentPPO:
    def __init__(
            self,
            model,
            actor_lr=3e-4,
            critic_lr=1e-3,
            clip=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            entropy_final=0.001,
            total_updates=500,
            max_grad_norm=0.5,
    ):
        self.model = model
        self.actor_opt = torch.optim.Adam(model.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(model.critic.parameters(), lr=critic_lr)
        self.clip = clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.entropy_final = entropy_final
        self.total_updates = total_updates
        self.max_grad_norm = max_grad_norm
        self._update = 0  # internal counter

    def _current_entropy_coef(self):
        # Linear decay from entropy_coef -> entropy_final over training
        frac = min(self._update / self.total_updates, 1.0)
        coef = self.entropy_coef + frac * (self.entropy_final - self.entropy_coef)
        return max(coef, 0.005)  # floor — never let entropy collapse completely

    def update(self, buffer, next_state, batch_size=256, epochs=10):
        if not buffer.obs:
            return

        entropy_coef = self._current_entropy_coef()
        self._update += 1

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

                # Actor loss
                ratio = torch.exp(logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                # Critic loss (clipped value loss)
                logprobs, values, entropy = self.model.evaluate(
                    batch_obs, batch_states, batch_actions
                )
                values_old = values.detach()
                values_clipped = values_old + torch.clamp(values - values_old, -self.clip, self.clip)
                critic_loss = torch.max(
                    (batch_returns - values).pow(2),
                    (batch_returns - values_clipped).pow(2),
                ).mean()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()