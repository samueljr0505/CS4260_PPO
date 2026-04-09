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


def _coverage_bonus(obs_dict, agents, threshold=0.15, bonus=1.0):
    """
    Computes a shaped reward bonus that encourages each agent to get close
    to at least one landmark. Added on top of the team reward during training
    only — never used for logging or plots.

    Args:
        obs_dict:  dictionary of {agent_name: obs_array} from the environment
        agents:    list of active agent names (from env.agents)
        threshold: distance below which an agent is considered "covering" a landmark
                   (landmark radius in simple_spread is ~0.1–0.15 world units)
        bonus:     reward added per agent that is close enough to any landmark

    Returns:
        total: float, sum of bonuses across all agents this timestep
    """
    total = 0.0
    num_landmarks = 3

    for agent in agents:
        obs = obs_dict[agent]
        # Each agent's observation is structured as:
        # [vel(2), pos(2), landmark_rel_pos(num_landmarks * 2), other_agents_rel_pos(...)]
        # Landmark positions start at index 4, stored as (x, y) pairs relative to this agent.
        # Relative position means the vector FROM the agent TO the landmark,
        # so its magnitude is directly the distance — no coordinate conversion needed.

        for i in range(num_landmarks):
            # Extract the relative x, y to landmark i
            lx = obs[4 + i * 2]
            ly = obs[4 + i * 2 + 1]

            # Euclidean distance from this agent to landmark i
            dist = (lx ** 2 + ly ** 2) ** 0.5

            if dist < threshold:
                # This agent is close enough to landmark i — give the bonus
                total += bonus

                # Break immediately: only reward once per agent even if it's
                # near multiple landmarks. Without this, an agent sitting between
                # two landmarks would get double credit, incentivizing it to hover
                # at intersections rather than commit to covering one landmark.
                break

    return total

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_rollout(env, model, buffer, rollout_steps, num_landmarks=3):
    """
    Collect exactly `rollout_steps` environment steps.
    Supports both centralized critic (global state) and
    ablation mode (local obs) via model.centralized flag.
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

        # Global state always computed for logging/bootstrapping
        state = torch.tensor(env.state(), dtype=torch.float32)

        # Critic input depends on centralized flag
        if model.centralized_critic:
            critic_input = state
        else:
            # Local critic uses first agent's obs as representative
            critic_input = torch.tensor(
                obs_dict[env.agents[0]], dtype=torch.float32
            )

        with torch.no_grad():
            value = model.value(critic_input)

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
        shaped_reward = team_reward + _coverage_bonus(next_obs, env.agents)

        # Buffer stores critic_input (state or obs) not always global state
        for obs, action, logprob in zip(cached_obs, cached_actions, cached_logprobs):
            buffer.add(
                obs=obs.detach(),
                state=critic_input.detach(),  # global state or local obs
                action=action.detach(),
                logprob=logprob.detach(),
                reward=shaped_reward,
                done=done,
                value=value.detach(),
            )

        obs_dict = next_obs
        steps_collected += 1
        episode_reward += team_reward  # raw reward for logging
        coord_log.append(success_rate(next_obs, num_landmarks=num_landmarks))

        if done:
            completed_rewards.append(episode_reward)
            episode_reward = 0.0
            obs_dict, _ = env.reset()

    # Bootstrap value from final critic input
    if model.centralized_critic:
        last_critic_input = torch.tensor(env.state(), dtype=torch.float32)
    else:
        last_critic_input = torch.tensor(
            obs_dict[env.agents[0]] if env.agents else
            cached_obs[0].numpy(),
            dtype=torch.float32
        )

    last_truncated = not done
    mean_coord = float(np.mean(coord_log)) if coord_log else 0.0

    return completed_rewards, mean_coord, last_critic_input, last_truncated


def train(
    seed,
    num_agents=3,
    num_landmarks=3,
    updates=300,
    rollout_steps=4096,
    max_cycles=25,
    ppo_epochs=15,
    batch_size=256,
    return_model = False,
    ablation=False,
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

    model = MultiAgentActorCritic(obs_dim=obs_dim, act_dim=act_dim, state_dim=state_dim, centralized_critic=not ablation)
    ppo = MultiAgentPPO(model, actor_lr=3e-4, critic_lr=1e-3, total_updates=updates, entropy_final=0.003)
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

    os.makedirs("pt_files", exist_ok=True)
    torch.save(model.state_dict(), f"pt_files/model_seed{seed}.pt")
    print(f"Saved model to pt_files/model_seed{seed}.pt")

    if return_model:
        return episode_rewards_log, coordination_log, model
    return episode_rewards_log, coordination_log


def main():
    all_rewards = []
    all_coords = []

    for seed in [0, 1, 2]:
        rewards, coords = train(seed=seed)
        all_rewards.append(rewards)
        all_coords.append(coords)

    os.makedirs("pt_files", exist_ok=True)
    np.save("pt_files/simple_spread_rewards.npy", np.asarray(all_rewards, dtype=np.float32))
    np.save("pt_files/simple_spread_success.npy", np.asarray(all_coords, dtype=np.float32))
    print("DONE")


if __name__ == "__main__":
    main()