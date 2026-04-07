import os

import matplotlib.pyplot as plt
import numpy as np


RUNS_DIR = "runs"
PLOTS_DIR = "plots"


def load_runs(path):
    data = np.load(path, allow_pickle=True)
    runs = [np.asarray(run, dtype=np.float32) for run in data]

    if not runs:
        raise ValueError(f"No runs found in {path}")

    min_len = min(len(run) for run in runs)
    if min_len == 0:
        raise ValueError(f"Empty run found in {path}")

    return np.stack([run[:min_len] for run in runs], axis=0)


def plot_with_band(values, title, xlabel, ylabel, output_path):
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    x = np.arange(len(mean))

    plt.figure(figsize=(9, 5))
    plt.plot(x, mean, linewidth=2, label="Mean")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25, label="Std")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    rewards = load_runs(os.path.join(RUNS_DIR, "simple_spread_rewards.npy"))
    coordination = load_runs(os.path.join(RUNS_DIR, "simple_spread_coord.npy"))

    plot_with_band(
        rewards,
        title="Multi-Agent PPO Simple Spread Reward",
        xlabel="Update",
        ylabel="Average Step Reward",
        output_path=os.path.join(PLOTS_DIR, "simple_spread_reward.png"),
    )

    plot_with_band(
        coordination,
        title="Multi-Agent PPO Simple Spread Coordination",
        xlabel="Update",
        ylabel="Mean Pairwise Distance",
        output_path=os.path.join(PLOTS_DIR, "simple_spread_coordination.png"),
    )

    print(f"Saved plots to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
