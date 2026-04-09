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


def plot_with_band(values, title, xlabel, ylabel, output_path, y_lim=None):
    """
    values shape: (num_seeds, num_updates)
    """

    x = np.arange(values.shape[1])

    mean = values.mean(axis=0)
    std = values.std(axis=0)

    plt.figure(figsize=(10, 5))

    # -------------------------
    # Plot individual seeds
    # -------------------------
    for i in range(values.shape[0]):
        plt.plot(x, values[i], linewidth=1.2, alpha=0.5, label=f"Seed {i}")

    # -------------------------
    # Mean + Std band
    # -------------------------
    plt.plot(x, mean, linewidth=2.5, label="Mean", color="black")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2, color="black")

    # -------------------------
    # Formatting
    # -------------------------
    if y_lim is not None:
        plt.ylim(y_lim)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load data
    rewards = load_runs(os.path.join(RUNS_DIR, "simple_spread_rewards2.npy"))
    success = load_runs(os.path.join(RUNS_DIR, "simple_spread_coord2.npy"))

    # Reward plot
    plot_with_band(
        rewards,
        title="Multi-Agent PPO Simple Spread Reward",
        xlabel="Episode",
        ylabel="Episode Mean Reward",
        output_path=os.path.join(PLOTS_DIR, "simple_spread_reward2.png"),
    )

    # Success rate plot
    plot_with_band(
        success,
        title="Multi-Agent PPO Simple Spread Success Rate",
        xlabel="Episode",
        ylabel="Success Rate",
        output_path=os.path.join(PLOTS_DIR, "simple_spread_success2.png"),
    )

    print(f"Saved plots to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
