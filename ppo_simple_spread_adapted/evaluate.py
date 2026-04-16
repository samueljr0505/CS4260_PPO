import os
import numpy as np
import torch

try:
    from mpe2 import simple_spread_v3
except ImportError:
    from pettingzoo.mpe import simple_spread_v3

RUNS_DIR = "pt_files"
SUFFIX = "mappo_full"

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

        episode_reward += float(next(iter(rewards.values())))
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


def run_validation(
    seeds=(0, 1, 2),
    num_agents=3,
    num_landmarks=3,
    eval_episodes=100,
    max_cycles=25
):

    all_eval = []

    # -----------------------------
    # CREATE ENV TO GET DIMS
    # -----------------------------
    env = simple_spread_v3.parallel_env(
        N=num_agents,
        max_cycles=max_cycles,
        continuous_actions=False,
        local_ratio=0.0,
    )

    obs_dict, _ = env.reset()

    obs_dim = list(obs_dict.values())[0].shape[0]
    act_dim = env.action_space(env.possible_agents[0]).n
    state_dim = obs_dim * num_agents

    env.close()

    for seed in seeds:
        model_path = os.path.join(RUNS_DIR, f"model_{SUFFIX}_seed{seed}.pt")

        print(f"\n{'=' * 50}")
        print(f"Loading model: {model_path}")
        print(f"{'=' * 50}")

        # -----------------------------
        # BUILD MODEL CORRECTLY
        # -----------------------------
        model = MultiAgentActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            state_dim=state_dim,
            centralized_critic=True
        )

        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()

        # -----------------------------
        # EVALUATE
        # -----------------------------
        eval_seed = 999 + seed

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

    print("\nVALIDATION SUMMARY")
    rewards = [e["mean_reward"] for e in all_eval]
    success = [e["mean_success"] for e in all_eval]

    print(f"Reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Success Rate: {np.mean(success):.3f} ± {np.std(success):.3f}")

    np.save(
        "runs/validation_results_adapted.npy",
        np.array(all_eval, dtype=object),
        allow_pickle=True
    )

    return all_eval


if __name__ == "__main__":
    run_validation()