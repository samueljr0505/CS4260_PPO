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
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    x = np.arange(len(mean))

    # clamp for stability if this is success rate
    if "Success" in title:
        mean = np.clip(mean, 0, 1)
        std = np.clip(std, 0, 1)

    plt.figure(figsize=(9, 5))
    plt.plot(x, mean, linewidth=2, label="Mean")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25, label="Std")

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
    rewards = load_runs(os.path.join(RUNS_DIR, "simple_spread_rewards1.npy"))
    success = load_runs(os.path.join(RUNS_DIR, "simple_spread_coord1.npy"))

    # Reward plot
    plot_with_band(
        rewards,
        title="Multi-Agent PPO Simple Spread Reward",
        xlabel="Update",
        ylabel="Episode Mean Reward",
        output_path=os.path.join(PLOTS_DIR, "simple_spread_reward.png"),
    )

    # Success rate plot
    plot_with_band(
        success,
        title="Multi-Agent PPO Simple Spread Success Rate",
        xlabel="Update",
        ylabel="Success Rate",
        output_path=os.path.join(PLOTS_DIR, "simple_spread_success.png"),
    )

    print(f"Saved plots to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
