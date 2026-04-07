import numpy as np
import torch
import gymnasium as gym
from pettingzoo.mpe import simple_spread_v3
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ActorCritic
from ppo_reprod.ppo import PPO
from ppo_reprod.buffer import RolloutBuffer
from utils import coordination_metric


# -----------------------------
# ROLLOUT
# -----------------------------
def run_episode(env, model, buffer, max_steps=25):
    obs_dict, _ = env.reset()
    agents = env.agents

    ep_reward = 0

    for _ in range(max_steps):

        actions = {}
        logprobs = {}
        values = {}

        # ❗ SINGLE OBS (intentionally weak baseline)
        obs = torch.tensor(obs_dict[agents[0]], dtype=torch.float32)

        action, logprob = model.get_action(obs)
        value = model.critic(model.shared(obs)).squeeze(-1)

        # clip actions into valid space
        action = torch.sigmoid(action)

        for a in agents:
            actions[a] = action.detach().numpy()
            logprobs[a] = logprob.detach()
            values[a] = value.detach()

        next_obs, rewards, terminations, truncations, _ = env.step(actions)

        done = any(terminations.values()) or any(truncations.values())

        reward = sum(rewards.values())

        buffer.add(
            obs.detach(),
            action.detach(),
            logprob.detach(),
            torch.tensor(reward, dtype=torch.float32),
            done,
            value.detach()
        )

        obs_dict = next_obs
        ep_reward += reward

        if done:
            break

    return ep_reward


# -----------------------------
# TRAIN
# -----------------------------
def train(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = simple_spread_v3.parallel_env(
        N=3,
        max_cycles=25,
        continuous_actions=True
    )

    obs_dict, _ = env.reset()
    agents = env.agents

    obs_dim = obs_dict[agents[0]].shape[0]
    act_dim = env.action_space(agents[0]).shape[0]

    model = ActorCritic(obs_dim, act_dim)
    ppo = PPO(model)
    buffer = RolloutBuffer()

    rewards_log = []
    coord_log = []

    for episode in range(1000):

        ep_reward = run_episode(env, model, buffer)

        ppo.update(buffer)
        buffer.clear()

        rewards_log.append(ep_reward)
        coord_log.append(coordination_metric(env))

        if episode % 20 == 0:
            print(
                f"[Seed {seed}] Ep {episode} | "
                f"Reward: {ep_reward:.2f} | "
                f"Coord: {coord_log[-1]:.3f}"
            )

    return rewards_log, coord_log


# -----------------------------
# RUN 3 SEEDS
# -----------------------------
all_rewards = []
all_coords = []

for seed in [0, 1, 2]:
    r, c = train(seed)
    all_rewards.append(r)
    all_coords.append(c)

os.makedirs("runs", exist_ok=True)

np.save("runs/simple_spread_rewards.npy", np.array(all_rewards))
np.save("runs/simple_spread_coord.npy", np.array(all_coords))

print("DONE")