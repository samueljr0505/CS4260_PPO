import os
import numpy as np
import torch

try:
    from mpe2 import simple_spread_v3
except ImportError:
    from pettingzoo.mpe import simple_spread_v3

from model import MultiAgentActorCritic
from utils import success_rate


def evaluate_model(model, num_episodes=100, seed=999, num_agents=3,
                   num_landmarks=3, max_cycles=25):
    """
    Runs the trained policy on held-out episodes with NO gradient updates.
    Uses greedy action selection (no exploration) to measure true policy quality.

    Args:
        model:        trained MultiAgentActorCritic
        num_episodes: how many full episodes to evaluate over
        seed:         held-out seed never seen during training

    Returns:
        dict of mean/std reward and success rate across all eval episodes
    """
    env = simple_spread_v3.parallel_env(
        N=num_agents,
        max_cycles=max_cycles,
        continuous_actions=False,
        local_ratio=0.0,
    )

    obs_dict, _ = env.reset(seed=seed)

    episode_rewards = []
    episode_success = []
    episode_reward = 0.0
    step_success = []
    episodes_done = 0

    model.eval()  # disable dropout/batchnorm if any

    while episodes_done < num_episodes:
        if not env.agents:
            episode_rewards.append(episode_reward)
            episode_success.append(float(np.mean(step_success)) if step_success else 0.0)
            episode_reward = 0.0
            step_success = []
            episodes_done += 1
            if episodes_done < num_episodes:
                obs_dict, _ = env.reset()
            continue

        action_dict = {}
        for agent in env.agents:
            obs = torch.tensor(obs_dict[agent], dtype=torch.float32)
            with torch.no_grad():
                # Greedy — take the most probable action, no sampling
                dist = model.policy(obs)
                action = dist.probs.argmax()
            action_dict[agent] = int(action.item())

        next_obs, rewards, terminations, truncations, _ = env.step(action_dict)
        done = any(terminations.values()) or any(truncations.values())

        episode_reward += float(sum(rewards.values()))
        step_success.append(success_rate(next_obs, num_landmarks=num_landmarks))

        if done:
            episode_rewards.append(episode_reward)
            episode_success.append(float(np.mean(step_success)))
            episode_reward = 0.0
            step_success = []
            episodes_done += 1
            if episodes_done < num_episodes:
                obs_dict, _ = env.reset()
        else:
            obs_dict = next_obs

    env.close()
    model.train()  # restore training mode

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_success": float(np.mean(episode_success)),
        "std_success": float(np.std(episode_success)),
        "all_rewards": episode_rewards,
        "all_success": episode_success,
    }


def run_validation(seeds=(0, 1, 2), num_agents=3, num_landmarks=3,
                   updates=300, rollout_steps=4096, max_cycles=25,
                   ppo_epochs=15, batch_size=256, eval_episodes=100):
    """
    Retrain each seed from scratch then immediately evaluate on held-out episodes.
    Saves validation results to runs/validation_results.npy
    """
    # Import here to avoid circular imports
    from train import train

    all_eval = []

    for seed in seeds:
        print(f"\n{'=' * 50}")
        print(f"Training seed {seed} for validation...")
        print(f"{'=' * 50}")

        rewards_log, coord_log, model = train(
            seed=seed,
            num_agents=num_agents,
            num_landmarks=num_landmarks,
            updates=updates,
            rollout_steps=rollout_steps,
            max_cycles=max_cycles,
            ppo_epochs=ppo_epochs,
            batch_size=batch_size,
            return_model=True,  # we add this flag below
        )

        # Evaluate on a held-out seed (training seeds + 999 offset)
        eval_seed = 999 + seed
        print(f"Evaluating seed {seed} on held-out env seed {eval_seed}...")
        results = evaluate_model(
            model,
            num_episodes=eval_episodes,
            seed=eval_seed,
            num_agents=num_agents,
            num_landmarks=num_landmarks,
            max_cycles=max_cycles,
        )

        print(
            f"[Seed {seed}] VALIDATION | "
            f"Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f} | "
            f"Success: {results['mean_success']:.3f} ± {results['std_success']:.3f}"
        )
        all_eval.append(results)

    # Summary across seeds
    print(f"\n{'=' * 50}")
    print("VALIDATION SUMMARY (across all seeds)")
    print(f"{'=' * 50}")
    rewards = [e["mean_reward"] for e in all_eval]
    success = [e["mean_success"] for e in all_eval]
    print(f"Reward:       {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Success Rate: {np.mean(success):.3f} ± {np.std(success):.3f}")

    os.makedirs("runs", exist_ok=True)
    np.save("runs/validation_results.npy", np.array(all_eval, dtype=object), allow_pickle=True)
    print("\nSaved to runs/validation_results.npy")

    return all_eval


if __name__ == "__main__":
    run_validation()