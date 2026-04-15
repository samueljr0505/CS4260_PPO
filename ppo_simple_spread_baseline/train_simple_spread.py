import numpy as np
import torch
import os

from pettingzoo.mpe import simple_spread_v3

from model import ActorCritic
from ppo import PPO
from buffer import RolloutBuffer
from utils import success_rate


def run_episode(env, model, buffer, max_steps=25):

    obs_dict, _ = env.reset()
    agents = env.agents

    ep_reward = 0

    for _ in range(max_steps):

        actions = {}

        obs_list = []
        action_list = []
        logprob_list = []
        value_list = []

        for a in agents:
            obs = torch.tensor(obs_dict[a], dtype=torch.float32)

            action, logprob = model.get_action(obs)
            value = model.value(obs)

            actions[a] = int(action.item())

            obs_list.append(obs)
            action_list.append(action)
            logprob_list.append(logprob)
            value_list.append(value)

        next_obs, rewards, terminations, truncations, _ = env.step(actions)

        done = any(terminations.values()) or any(truncations.values())
        reward = torch.tensor(sum(rewards.values()), dtype=torch.float32)

        for i in range(len(agents)):
            buffer.add(
                obs_list[i],
                action_list[i],
                logprob_list[i],
                reward,
                done,
                value_list[i]
            )

        obs_dict = next_obs
        ep_reward += reward.item()

        if done:
            break

    # bootstrap value
    next_value = torch.tensor(0.0)

    if not done:
        first_agent = agents[0]
        obs = torch.tensor(obs_dict[first_agent], dtype=torch.float32)
        next_value = model.value(obs)

    return ep_reward, obs_dict, next_value


def train(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = simple_spread_v3.parallel_env(
        N=3,
        max_cycles=25,
        continuous_actions=False
    )

    obs_dict, _ = env.reset()
    agents = env.agents

    obs_dim = obs_dict[agents[0]].shape[0]
    act_dim = env.action_space(agents[0]).n

    model = ActorCritic(obs_dim, act_dim)
    ppo = PPO(model)
    buffer = RolloutBuffer()

    rewards_log = []
    coord_log = []

    for episode in range(500):

        ep_reward, obs_dict, next_value = run_episode(env, model, buffer)

        ppo.update(buffer, next_value)
        buffer.clear()

        rewards_log.append(ep_reward)
        coord_log.append(success_rate(obs_dict))

        if episode % 20 == 0:
            print(f"[Seed {seed}] Episode {episode} | Reward {ep_reward:.2f}")

    return rewards_log, coord_log


if __name__ == "__main__":

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