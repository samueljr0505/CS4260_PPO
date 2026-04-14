import os
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = "../runs"
PLOTS_DIR = "../plots"


def load(path):
    data = np.load(path, allow_pickle=True)
    runs = [np.asarray(r, dtype=np.float32) for r in data]
    min_len = min(len(r) for r in runs)
    return np.stack([r[:min_len] for r in runs], axis=0)


def rolling(arr, w=20):
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def plot_ablation(adapted, ablation, ylabel, title, output_path, window=20):

    x_len = min(adapted.shape[1], ablation.shape[1])
    adapted  = adapted[:, :x_len]
    ablation = ablation[:, :x_len]

    x = np.arange(x_len)
    roll_x = x[window - 1:]

    adapted_mean  = adapted.mean(axis=0)
    ablation_mean = ablation.mean(axis=0)

    adapted_std   = adapted.std(axis=0)
    ablation_std  = ablation.std(axis=0)

    # 🔥 MATCH PLOT SIGNAL (IMPORTANT FIX)
    adapted_roll  = rolling(adapted_mean, window)
    ablation_roll = rolling(ablation_mean, window)

    # use LAST 100 of smoothed signal (more stable than raw last-50)
    adapted_final = adapted_roll[-100:].mean()
    ablation_final = ablation_roll[-100:].mean()
    gap = adapted_final - ablation_final

    fig, ax = plt.subplots(figsize=(11, 5))

    # per-seed lines
    for i in range(adapted.shape[0]):
        ax.plot(x, adapted[i],  color="tab:blue", linewidth=0.6, alpha=0.2)
        ax.plot(x, ablation[i], color="tab:orange", linewidth=0.6, alpha=0.2)

    # std bands
    ax.fill_between(x,
                    adapted_mean - adapted_std,
                    adapted_mean + adapted_std,
                    alpha=0.12, color="tab:blue")

    ax.fill_between(x,
                    ablation_mean - ablation_std,
                    ablation_mean + ablation_std,
                    alpha=0.12, color="tab:orange")

    # rolling means
    ax.plot(roll_x, adapted_roll,
            color="tab:blue", linewidth=2.5,
            label=f"Full MAPPO (final: {adapted_final:.3f})")

    ax.plot(roll_x, ablation_roll,
            color="tab:orange", linewidth=2.5, linestyle="--",
            label=f"Ablation (final: {ablation_final:.3f})")

    # annotation (FIXED)
    ax.annotate(
        f"Gap: {gap:+.4f}",
        xy=(x_len - 1, (adapted_mean[-1] + ablation_mean[-1]) / 2),
        fontsize=10,
        xytext=(-80, 10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-", lw=0.8),
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")
    print(f"Gap ({title}): {gap:+.6f}")


def plot_three_way(adapted, abl_critic, abl_shaping,
                   ylabel, title, output_path, window=20):

    x_len = min(adapted.shape[1], abl_critic.shape[1], abl_shaping.shape[1])

    adapted     = adapted[:, :x_len]
    abl_critic  = abl_critic[:, :x_len]
    abl_shaping = abl_shaping[:, :x_len]

    x = np.arange(x_len)
    roll_x = x[window - 1:]

    adapted_mean  = adapted.mean(axis=0)
    critic_mean   = abl_critic.mean(axis=0)
    shaping_mean  = abl_shaping.mean(axis=0)

    adapted_std   = adapted.std(axis=0)
    critic_std    = abl_critic.std(axis=0)
    shaping_std   = abl_shaping.std(axis=0)

    # 🔥 FIXED consistency
    adapted_roll = rolling(adapted_mean, window)
    critic_roll  = rolling(critic_mean, window)
    shaping_roll = rolling(shaping_mean, window)

    adapted_final = adapted_roll[-100:].mean()
    critic_final  = critic_roll[-100:].mean()
    shaping_final = shaping_roll[-100:].mean()

    gap_critic  = adapted_final - critic_final
    gap_shaping = adapted_final - shaping_final

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(x, adapted_mean-adapted_std, adapted_mean+adapted_std,
                    alpha=0.10, color="tab:blue")
    ax.fill_between(x, critic_mean-critic_std, critic_mean+critic_std,
                    alpha=0.10, color="tab:orange")
    ax.fill_between(x, shaping_mean-shaping_std, shaping_mean+shaping_std,
                    alpha=0.10, color="tab:green")

    ax.plot(roll_x, adapted_roll,
            color="tab:blue", linewidth=2.5,
            label=f"Full MAPPO ({adapted_final:.3f})")

    ax.plot(roll_x, critic_roll,
            color="tab:orange", linestyle="--", linewidth=2.5,
            label=f"Local critic ({critic_final:.3f}, {gap_critic:+.4f})")

    ax.plot(roll_x, shaping_roll,
            color="tab:green", linestyle=":", linewidth=2.5,
            label=f"No shaping ({shaping_final:.3f}, {gap_shaping:+.4f})")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")


def print_summary(adapted_r, ablation_r, adapted_s, ablation_s, label="Ablation"):

    def final(x):
        return rolling(x.mean(axis=0), 20)[-100:].mean()

    a_r  = final(adapted_r)
    ab_r = final(ablation_r)
    a_s  = final(adapted_s)
    ab_s = final(ablation_s)

    print(f"\n=== {label.upper()} SUMMARY ===")
    print(f"{'Metric':<20} {'Full MAPPO':>12} {label:>14} {'Gap':>12}")
    print("-" * 60)
    print(f"{'Reward':<20} {a_r:>12.4f} {ab_r:>14.4f} {a_r-ab_r:>+12.4f}")
    print(f"{'Success Rate':<20} {a_s:>12.4f} {ab_s:>14.4f} {a_s-ab_s:>+12.4f}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    adapted_rewards = load(os.path.join(RUNS_DIR, "simple_spread_rewards2.npy"))
    adapted_success = load(os.path.join(RUNS_DIR, "simple_spread_coord2.npy"))

    critic_rewards = load(os.path.join(RUNS_DIR, "simple_spread_reward_ablation.npy"))
    critic_success = load(os.path.join(RUNS_DIR, "simple_spread_success_ablation.npy"))

    shaping_rewards = load(os.path.join(RUNS_DIR, "simple_spread_reward_ablation_shaping.npy"))
    shaping_success = load(os.path.join(RUNS_DIR, "simple_spread_success_ablation_shaping.npy"))

    plot_ablation(adapted_rewards, critic_rewards,
                  "Episode Mean Reward",
                  "Ablation: Centralized Critic",
                  os.path.join(PLOTS_DIR, "critic_reward.png"))

    plot_ablation(adapted_success, critic_success,
                  "Success Rate",
                  "Ablation: Centralized Critic",
                  os.path.join(PLOTS_DIR, "critic_success.png"))

    plot_ablation(adapted_rewards, shaping_rewards,
                  "Episode Mean Reward",
                  "Ablation: Reward Shaping",
                  os.path.join(PLOTS_DIR, "shaping_reward.png"))

    plot_ablation(adapted_success, shaping_success,
                  "Success Rate",
                  "Ablation: Reward Shaping",
                  os.path.join(PLOTS_DIR, "shaping_success.png"))

    plot_three_way(adapted_rewards, critic_rewards, shaping_rewards,
                   "Episode Mean Reward",
                   "Three-way Ablation",
                   os.path.join(PLOTS_DIR, "threeway_reward.png"))

    plot_three_way(adapted_success, critic_success, shaping_success,
                   "Success Rate",
                   "Three-way Ablation",
                   os.path.join(PLOTS_DIR, "threeway_success.png"))

    print_summary(adapted_rewards, critic_rewards,
                  adapted_success, critic_success,
                  label="Local Critic")

    print_summary(adapted_rewards, shaping_rewards,
                  adapted_success, shaping_success,
                  label="No Shaping")


if __name__ == "__main__":
    main()