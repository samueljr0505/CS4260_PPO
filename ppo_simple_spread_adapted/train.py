import os

import numpy as np
import torch

try:
    from mpe2 import simple_spread_v3
except ImportError:
    from pettingzoo.mpe import simple_spread_v3

from buffer import MultiAgentRolloutBuffer
from model import MultiAgentActorCritic
from ppo import MultiAgentPPO
from utils import coordination_metric


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_rollout(env, model, buffer, rollout_steps):
    obs_dict, _ = env.reset()
    episode_reward = 0.0
    episode_coord = []
    steps_collected = 0

    while steps_collected < rollout_steps:
        if not env.agents:
            obs_dict, _ = env.reset()
            continue

        state = torch.tensor(env.state(), dtype=torch.float32)

        with torch.no_grad():
            value = model.value(state)

        action_dict = {}
        cached_obs = []
        cached_actions = []
        cached_logprobs = []

        for agent in env.agents:
            obs = torch.tensor(obs_dict[agent], dtype=torch.float32)

            with torch.no_grad():
                action, logprob = model.act(obs)

            action_dict[agent] = int(action.item())
            cached_obs.append(obs)
            cached_actions.append(action)
            cached_logprobs.append(logprob)

        next_obs, rewards, terminations, truncations, _ = env.step(action_dict)
        done = any(terminations.values()) or any(truncations.values())
        team_reward = float(sum(rewards.values()))

        for obs, action, logprob in zip(cached_obs, cached_actions, cached_logprobs):
            buffer.add(
                obs=obs.detach(),
                state=state.detach(),
                action=action.detach(),
                logprob=logprob.detach(),
                reward=team_reward,
                done=done,
                value=value.detach(),
            )

        obs_dict = next_obs
        steps_collected += 1
        episode_reward += team_reward
        episode_coord.append(coordination_metric(obs_dict))

        if done:
            obs_dict, _ = env.reset()

    mean_coord = float(np.mean(episode_coord)) if episode_coord else 0.0
    return episode_reward, mean_coord


def train(
    seed,
    num_agents=3,
    updates=400,
    rollout_steps=1024,
    max_cycles=25,
    ppo_epochs=10,
    batch_size=256,
):
    set_seed(seed)

    env = simple_spread_v3.parallel_env(
        N=num_agents,
        max_cycles=max_cycles,
        continuous_actions=False,
        local_ratio=0.0,
    )

    obs_dict, _ = env.reset(seed=seed)
    first_agent = env.agents[0]

    obs_dim = obs_dict[first_agent].shape[0]
    state_dim = np.asarray(env.state()).shape[0]
    act_dim = int(env.action_space(first_agent).n)

    model = MultiAgentActorCritic(obs_dim=obs_dim, act_dim=act_dim, state_dim=state_dim)
    ppo = MultiAgentPPO(model)
    buffer = MultiAgentRolloutBuffer()

    rewards_log = []
    coordination_log = []

    for update_idx in range(updates):
        reward_sum, coord = collect_rollout(env, model, buffer, rollout_steps=rollout_steps)
        ppo.update(buffer, batch_size=batch_size, epochs=ppo_epochs)
        buffer.clear()

        avg_reward = reward_sum / rollout_steps
        rewards_log.append(avg_reward)
        coordination_log.append(coord)

        if update_idx % 10 == 0:
            trailing = rewards_log[-10:]
            print(
                f"[Seed {seed}] "
                f"Update {update_idx:04d} | "
                f"AvgStepReward: {avg_reward:.3f} | "
                f"Avg10: {np.mean(trailing):.3f} | "
                f"Coord: {coord:.3f}"
            )

    env.close()
    return rewards_log, coordination_log


def main():
    all_rewards = []
    all_coords = []

    for seed in [0, 1, 2]:
        rewards, coords = train(seed=seed)
        all_rewards.append(rewards)
        all_coords.append(coords)

    os.makedirs("runs", exist_ok=True)
    np.save("runs/simple_spread_rewards.npy", np.asarray(all_rewards, dtype=np.float32))
    np.save("runs/simple_spread_coord.npy", np.asarray(all_coords, dtype=np.float32))
    print("DONE")


if __name__ == "__main__":
    main()
