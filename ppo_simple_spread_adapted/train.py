import os
import sys
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


def collect_rollout(env, model, buffer, rollout_steps, num_landmarks=3,
                    use_shaping=True):
    """
    Collect exactly `rollout_steps` environment steps.
    Supports both centralized critic (global state) and
    ablation mode (local obs) via model.centralized_critic flag.

    Args:
        use_shaping: if False, disables reward shaping (shaping ablation)
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
        team_reward = float(next(iter(rewards.values())))
        # Apply shaping only if enabled — disabled for shaping ablation
        if use_shaping:
            shaped_reward = team_reward + _coverage_bonus(next_obs, env.agents)
        else:
            shaped_reward = team_reward

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
        episode_reward += team_reward  # always log raw reward, never shaped
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


def get_run_label(ablation_critic, ablation_shaping):
    """
    Returns (human_label, file_suffix) for the current run mode.

    Modes:
        full                 — centralized critic + reward shaping (default)
        ablation_critic      — local critic only, shaping on
        ablation_shaping     — centralized critic, shaping off
        ablation_both        — both ablated
    """
    if not ablation_critic and not ablation_shaping:
        return "Full MAPPO", "mappo_full"
    elif ablation_critic and not ablation_shaping:
        return "Ablation: Local Critic", "ablation_critic"
    elif not ablation_critic and ablation_shaping:
        return "Ablation: No Shaping", "ablation_shaping"
    else:
        return "Ablation: No Critic + No Shaping", "ablation_both"


def train(
    seed,
    num_agents=3,
    num_landmarks=3,
    updates=300,
    rollout_steps=4096,
    max_cycles=25,
    ppo_epochs=15,
    batch_size=256,
    return_model=False,
    ablation_critic=False,   # True = local obs for critic (no global state)
    ablation_shaping=False,  # True = no coverage bonus
):
    set_seed(seed)

    label, suffix = get_run_label(ablation_critic, ablation_shaping)

    env = simple_spread_v3.parallel_env(
        N=num_agents,
        max_cycles=max_cycles,
        continuous_actions=False,
        local_ratio=0.0,
    )

    obs_dict, _ = env.reset(seed=seed)
    first_agent = env.agents[0]

    obs_dim   = obs_dict[first_agent].shape[0]
    state_dim = np.asarray(env.state()).shape[0]
    act_dim   = int(env.action_space(first_agent).n)

    model = MultiAgentActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        state_dim=state_dim,
        centralized_critic=not ablation_critic,
    )
    ppo    = MultiAgentPPO(model, actor_lr=3e-4, critic_lr=1e-3,
                           total_updates=updates, entropy_final=0.003)
    buffer = MultiAgentRolloutBuffer()

    episode_rewards_log = []
    coordination_log    = []

    print(f"\n{'='*60}")
    print(f"Mode:               {label}")
    print(f"Seed:               {seed}  |  Updates: {updates}")
    print(f"Centralized critic: {not ablation_critic}  |  "
          f"Reward shaping: {not ablation_shaping}")
    print(f"{'='*60}")

    for update_idx in range(updates):
        completed_rewards, coord, last_state, last_truncated = collect_rollout(
            env, model, buffer,
            rollout_steps=rollout_steps,
            num_landmarks=num_landmarks,
            use_shaping=not ablation_shaping,
        )

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
                f"[{suffix}][Seed {seed}] "
                f"Update {update_idx:04d} | "
                f"MeanEpReward: {mean_ep_reward:.3f} | "
                f"Avg10: {np.mean(trailing):.3f} | "
                f"Success Rate: {coord:.3f}"
            )

    env.close()

    # Save model checkpoint with mode suffix so files never overwrite each other
    os.makedirs("pt_files", exist_ok=True)
    model_path = f"pt_files/model_{suffix}_seed{seed}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model → {model_path}")

    if return_model:
        return episode_rewards_log, coordination_log, model
    return episode_rewards_log, coordination_log


def save_results(all_rewards, all_coords, suffix):
    """Saves reward and success rate arrays to runs/ with the mode suffix."""
    os.makedirs("runs", exist_ok=True)
    r_path = f"runs/simple_spread_rewards_{suffix}.npy"
    s_path = f"runs/simple_spread_success_{suffix}.npy"
    np.save(r_path, np.asarray(all_rewards, dtype=np.float32))
    np.save(s_path, np.asarray(all_coords,  dtype=np.float32))
    print(f"Saved → {r_path}")
    print(f"Saved → {s_path}")


def main():
    """
    Selects run mode from command line argument.

    Usage:
        python train.py                   →  full MAPPO (default)
        python train.py full              →  full MAPPO
        python train.py ablation_critic   →  local critic, shaping on
        python train.py ablation_shaping  →  centralized critic, shaping off
        python train.py ablation_both     →  both ablated

    Output files are named automatically based on mode:
        runs/simple_spread_rewards_mappo_full.npy
        runs/simple_spread_rewards_ablation_critic.npy
        runs/simple_spread_rewards_ablation_shaping.npy
        runs/simple_spread_rewards_ablation_both.npy
    """
    seeds = [0, 1, 2]

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    mode_configs = {
        "full": {
            "ablation_critic":  False,
            "ablation_shaping": False,
        },
        "ablation_critic": {
            "ablation_critic":  True,
            "ablation_shaping": False,
        },
        "ablation_shaping": {
            "ablation_critic":  False,
            "ablation_shaping": True,
        },
        "ablation_both": {
            "ablation_critic":  True,
            "ablation_shaping": True,
        },
    }

    if mode not in mode_configs:
        print(f"Unknown mode '{mode}'.")
        print(f"Choose from: {list(mode_configs.keys())}")
        sys.exit(1)

    cfg = mode_configs[mode]
    _, suffix = get_run_label(cfg["ablation_critic"], cfg["ablation_shaping"])

    print(f"\nRunning: {mode}  →  suffix = '{suffix}'")
    print(f"Seeds: {seeds}\n")

    all_rewards = []
    all_coords  = []

    for seed in seeds:
        rewards, coords = train(seed=seed, **cfg)
        all_rewards.append(rewards)
        all_coords.append(coords)

    save_results(all_rewards, all_coords, suffix)
    print("\nDONE")


if __name__ == "__main__":
    main()