import os
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = "runs"
PLOTS_DIR = "plots"

# Explicit key mapping — no string building that causes typos
ALL_KEYS = {
    "reward":  "all_rewards",
    "success": "all_success",
}


def load_runs(path):
    data = np.load(path, allow_pickle=True)
    runs = [np.asarray(run, dtype=np.float32) for run in data]
    min_len = min(len(run) for run in runs)
    return np.stack([run[:min_len] for run in runs], axis=0)


def rolling(arr, w=20):
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def plot_training_vs_validation(train_values, val_results, metric,
                                ylabel, title, output_path, window=20):
    """
    Left panel:  training curves (per seed + rolling mean)
    Right panel: validation distribution (scatter + mean±std per seed)

    Args:
        train_values : np.ndarray (num_seeds, num_updates)
        val_results  : list of dicts, one per seed, from evaluate_model()
        metric       : "reward" or "success"
        ylabel       : y-axis label for both panels
        title        : figure title
        output_path  : where to save the PNG
        window       : rolling average window size
    """
    all_key = ALL_KEYS[metric]   # "all_rewards" or "all_success"
    mean_key = f"mean_{metric}"  # "mean_reward" or "mean_success"
    std_key  = f"std_{metric}"   # "std_reward"  or "std_success"

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [3, 1]}
    )

    # ── Left panel: training curves ────────────────────────────────────
    ax = axes[0]
    x      = np.arange(train_values.shape[1])
    mean   = train_values.mean(axis=0)
    std    = train_values.std(axis=0)
    roll_x = x[window - 1:]

    colors_train = ["tab:blue", "tab:orange", "tab:green"]
    for i in range(train_values.shape[0]):
        ax.plot(x, train_values[i], linewidth=0.8, alpha=0.3,
                color=colors_train[i], label=f"Seed {i}")

    ax.fill_between(x, mean - std, mean + std, alpha=0.15, color="black")
    ax.plot(x, mean, linewidth=1.0, color="black", alpha=0.3)
    ax.plot(roll_x, rolling(mean, window), linewidth=2.5,
            color="black", label=f"Rolling mean (w={window})")

    ax.set_title(f"Training — {title}")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)

    # ── Right panel: validation per seed ───────────────────────────────
    ax = axes[1]

    val_episodes = [r[all_key]  for r in val_results]
    means        = [r[mean_key] for r in val_results]
    stds         = [r[std_key]  for r in val_results]

    colors_val = ["tab:blue", "tab:orange", "tab:green"]

    for i, (episodes, m, s) in enumerate(zip(val_episodes, means, stds)):
        # Jittered scatter of individual eval episodes
        jitter = np.random.normal(i, 0.06, size=len(episodes))
        ax.scatter(jitter, episodes, alpha=0.2, s=8, color=colors_val[i])
        # Mean ± std error bar
        ax.errorbar(i, m, yerr=s, fmt="o", color=colors_val[i],
                    markersize=8, capsize=6, linewidth=2,
                    label=f"Seed {i}: {m:.3f}±{s:.3f}")

    overall_mean = np.mean(means)
    ax.axhline(overall_mean, color="black", linewidth=1.5,
               linestyle="--", label=f"Overall: {overall_mean:.3f}")

    ax.set_title(f"Validation — {title}")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(val_results)))
    ax.set_xticklabels([f"Seed {i}" for i in range(len(val_results))])
    ax.legend(fontsize=8)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")


def print_summary(train_rewards, train_success, val_results):
    train_r = train_rewards.mean(axis=0)[-50:].mean()
    train_s = train_success.mean(axis=0)[-50:].mean()
    val_r   = np.mean([r["mean_reward"]  for r in val_results])
    val_s   = np.mean([r["mean_success"] for r in val_results])
    val_r_std = np.std([r["mean_reward"]  for r in val_results])
    val_s_std = np.std([r["mean_success"] for r in val_results])

    print("\n=== FINAL SUMMARY ===")
    print(f"{'Metric':<20} {'Train (last 50)':>16} {'Val Mean':>12} {'Val Std':>10}")
    print("-" * 62)
    print(f"{'Reward':<20} {train_r:>16.3f} {val_r:>12.3f} {val_r_std:>10.3f}")
    print(f"{'Success Rate':<20} {train_s:>16.3f} {val_s:>12.3f} {val_s_std:>10.3f}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load training data
    train_rewards = load_runs(os.path.join(RUNS_DIR, "simple_spread_rewards2.npy"))
    train_success = load_runs(os.path.join(RUNS_DIR, "simple_spread_coord2.npy"))

    # Load validation results saved by evaluate.py
    val_results = np.load(
        os.path.join(RUNS_DIR, "validation_results.npy"),
        allow_pickle=True
    ).tolist()

    # Reward plot
    plot_training_vs_validation(
        train_rewards, val_results,
        metric="reward",
        ylabel="Episode Reward",
        title="MAPPO Simple Spread — Reward",
        output_path=os.path.join(PLOTS_DIR, "validation_reward.png"),
    )

    # Success rate plot
    plot_training_vs_validation(
        train_success, val_results,
        metric="success",
        ylabel="Success Rate",
        title="MAPPO Simple Spread — Success Rate",
        output_path=os.path.join(PLOTS_DIR, "validation_success.png"),
    )

    print_summary(train_rewards, train_success, val_results)


if __name__ == "__main__":
    main()