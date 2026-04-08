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
from utils import success_rate


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_rollout(env, model, buffer, rollout_steps, num_landmarks=3):
    """
    Collect exactly `rollout_steps` *environment steps* (not agent steps).
    Returns:
        episode_rewards  - list of completed episode total rewards
        mean_coord       - mean coordination metric over the rollout
        last_state       - final global state (for value bootstrapping)
        last_truncated   - whether the rollout ended mid-episode (for bootstrapping)
    """
    obs_dict, _ = env.reset()
    episode_reward = 0.0
    completed_rewards = []
    coord_log = []
    steps_collected = 0

    while steps_collected < rollout_steps:
        if not env.agents:
            completed_rewards.append(episode_reward)
            episode_reward = 0.0
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

        # Add one buffer entry per agent, sharing state/value/reward/done
        # (CTDE: centralised critic, decentralised actors)
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
        # Count one step per *environment* step, not per agent
        steps_collected += 1
        episode_reward += team_reward
        coord_log.append(success_rate(obs_dict, num_landmarks=num_landmarks))

        if done:
            completed_rewards.append(episode_reward)
            episode_reward = 0.0
            obs_dict, _ = env.reset()

    # Capture the last state for value bootstrapping (needed when rollout ends mid-episode)
    last_state = torch.tensor(env.state(), dtype=torch.float32)
    last_truncated = not done  # if we exited loop without a done, episode was cut short

    mean_coord = float(np.mean(coord_log)) if coord_log else 0.0
    return completed_rewards, mean_coord, last_state, last_truncated


def train(
    seed,
    num_agents=3,
    num_landmarks=3,
    updates=500,
    rollout_steps=2048,
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

    episode_rewards_log = []  # mean reward per completed episode per update
    coordination_log = []

    for update_idx in range(updates):
        completed_rewards, coord, last_state, last_truncated = collect_rollout(
            env, model, buffer,
            rollout_steps=rollout_steps,
            num_landmarks=num_landmarks,
        )

        # Pass next_state so PPO can bootstrap if rollout ended mid-episode
        ppo.update(
            buffer,
            next_state=last_state if last_truncated else torch.zeros_like(last_state),
            batch_size=batch_size,
            epochs=ppo_epochs,
        )
        buffer.clear()

        mean_ep_reward = float(np.mean(completed_rewards)) if completed_rewards else 0.0
        episode_rewards_log.append(mean_ep_reward)
        coordination_log.append(coord)

        if update_idx % 10 == 0:
            trailing = episode_rewards_log[-10:]
            print(
                f"[Seed {seed}] "
                f"Update {update_idx:04d} | "
                f"MeanEpReward: {mean_ep_reward:.3f} | "
                f"Avg10: {np.mean(trailing):.3f} | "
                f"Success Rate: {coord:.3f}"
            )

    env.close()
    return episode_rewards_log, coordination_log


def main():
    all_rewards = []
    all_coords = []

    for seed in [0, 1, 2]:
        rewards, coords = train(seed=seed)
        all_rewards.append(rewards)
        all_coords.append(coords)

    os.makedirs("runs", exist_ok=True)
    np.save("runs/simple_spread_rewards1.npy", np.asarray(all_rewards, dtype=np.float32))
    np.save("runs/simple_spread_coord1.npy", np.asarray(all_coords, dtype=np.float32))
    print("DONE")


if __name__ == "__main__":
    main()