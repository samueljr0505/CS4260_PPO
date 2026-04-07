"""
random_baseline.py

Runs a purely random policy on Simple Spread for 3 seeds,
then plots it alongside your saved PPO baseline results.

Place this file next to your existing code and run:
    python random_baseline.py

Expects:
    runs/simple_spread_rewards.npy   (already saved by your train script)
    runs/simple_spread_coord.npy     (already saved by your train script)

Outputs:
    runs/random_baseline_rewards.npy
    runs/comparison_reward.png
    runs/comparison_coord.png
"""

import numpy as np
import os
from pettingzoo.mpe import simple_spread_v3
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# -----------------------------
# RANDOM POLICY ROLLOUT
# -----------------------------
def run_random_episode(env, max_steps=25):
    obs_dict, _ = env.reset()
    agents = env.agents

    ep_reward = 0
    step_count = 0  # track steps

    for _ in range(max_steps):
        actions = {a: env.action_space(a).sample() for a in agents}
        _, rewards, terminations, truncations, _ = env.step(actions)

        reward = np.mean(list(rewards.values()))

        ep_reward += reward
        step_count += 1

        if any(terminations.values()) or any(truncations.values()):
            break

    # convert to average step reward
    avg_reward = ep_reward / step_count
    return avg_reward


def run_random_baseline(n_episodes=500, seeds=[0, 1, 2]):
    all_rewards = []

    for seed in seeds:
        np.random.seed(seed)
        env = simple_spread_v3.parallel_env(
            N=3,
            max_cycles=25,
            continuous_actions=True
        )
        env.reset(seed=seed)

        rewards_log = []
        for ep in range(n_episodes):
            r = run_random_episode(env)
            rewards_log.append(r)
            if ep % 50 == 0:
                print(f"[Random Seed {seed}] Ep {ep} | Reward: {r:.2f}")

        all_rewards.append(rewards_log)
        env.close()

    return np.array(all_rewards)


# -----------------------------
# PLOTTING
# -----------------------------
def smooth(arr, window=10):
    """Simple moving average for readability."""
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def plot_comparison(ppo_rewards, random_rewards, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    # PPO baseline
    ppo_mean = np.mean(ppo_rewards, axis=0)
    ppo_std  = np.std(ppo_rewards, axis=0)
    eps = np.arange(len(ppo_mean))
    ax.plot(eps, ppo_mean, color='steelblue', label='PPO Baseline (single-obs)')
    ax.fill_between(eps, ppo_mean - ppo_std, ppo_mean + ppo_std,
                    alpha=0.25, color='steelblue')

    # Random baseline
    rand_mean = np.mean(random_rewards, axis=0)
    rand_std  = np.std(random_rewards, axis=0)
    eps_r = np.arange(len(rand_mean))
    ax.plot(eps_r, rand_mean, color='tomato', linestyle='--', label='Random Policy')
    ax.fill_between(eps_r, rand_mean - rand_std, rand_mean + rand_std,
                    alpha=0.20, color='tomato')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Step Reward')
    ax.set_title('PPO Baseline vs Random Policy — Simple Spread')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    os.makedirs("runs", exist_ok=True)

    # ---- Run random baseline ----
    print("=== Running random policy ===")
    random_rewards = run_random_baseline(n_episodes=500, seeds=[0, 1, 2])
    np.save("runs/random_baseline_rewards.npy", random_rewards)
    print(f"Random policy mean final reward: {np.mean(random_rewards[:, -50:]):.2f}")

    # ---- Load your existing PPO results ----
    if not os.path.exists("runs/simple_spread_rewards.npy"):
        print("ERROR: runs/simple_spread_rewards.npy not found.")
        print("Run your train script first, then re-run this script.")
        sys.exit(1)

    ppo_rewards = np.load("runs/simple_spread_rewards.npy")  # shape (3, N_episodes)

    # Align episode counts (in case they differ)
    n_eps = min(ppo_rewards.shape[1], random_rewards.shape[1])
    ppo_rewards   = ppo_rewards[:, :n_eps]
    random_rewards = random_rewards[:, :n_eps]

    # ---- Plot ----
    plot_comparison(ppo_rewards, random_rewards, "plots/comparison_reward.png")

    # ---- Print summary table ----
    print("\n=== Summary (mean ± std over last 50 episodes, 3 seeds) ===")
    ppo_final  = ppo_rewards[:, -50:]
    rand_final = random_rewards[:, -50:]
    print(f"  PPO Baseline : {np.mean(ppo_final):.2f} ± {np.std(ppo_final):.2f}")
    print(f"  Random Policy: {np.mean(rand_final):.2f} ± {np.std(rand_final):.2f}")
    print("\nIf PPO ≈ Random, that's your failure evidence.")