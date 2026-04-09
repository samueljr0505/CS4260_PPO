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
    """
    Overlays full MAPPO vs one ablation condition.
    Per-seed lines are faded, rolling mean of the cross-seed mean is the
    primary visual signal.

    adapted  : np.ndarray (num_seeds, num_updates) — full MAPPO
    ablation : np.ndarray (num_seeds, num_updates) — ablation condition
    """
    x_len = min(adapted.shape[1], ablation.shape[1])
    adapted  = adapted[:, :x_len]
    ablation = ablation[:, :x_len]
    x = np.arange(x_len)
    roll_x = x[window - 1:]

    adapted_mean  = adapted.mean(axis=0)
    ablation_mean = ablation.mean(axis=0)
    adapted_std   = adapted.std(axis=0)
    ablation_std  = ablation.std(axis=0)

    fig, ax = plt.subplots(figsize=(11, 5))

    # ── Per-seed lines (faded) ─────────────────────────────────────────
    for i in range(adapted.shape[0]):
        ax.plot(x, adapted[i],  color="tab:blue",   linewidth=0.6, alpha=0.2)
        ax.plot(x, ablation[i], color="tab:orange",  linewidth=0.6, alpha=0.2)

    # ── Std bands ─────────────────────────────────────────────────────
    ax.fill_between(x,
                    adapted_mean  - adapted_std,
                    adapted_mean  + adapted_std,
                    alpha=0.12, color="tab:blue")
    ax.fill_between(x,
                    ablation_mean - ablation_std,
                    ablation_mean + ablation_std,
                    alpha=0.12, color="tab:orange")

    # ── Rolling means (primary signal) ────────────────────────────────
    ax.plot(roll_x, rolling(adapted_mean,  window),
            color="tab:blue",   linewidth=2.5,
            label=f"Full MAPPO — centralized critic  "
                  f"(final: {adapted_mean[-50:].mean():.1f})")
    ax.plot(roll_x, rolling(ablation_mean, window),
            color="tab:orange", linewidth=2.5, linestyle="--",
            label=f"Ablation  "
                  f"(final: {ablation_mean[-50:].mean():.1f})")

    # ── Gap annotation ─────────────────────────────────────────────────
    gap = adapted_mean[-50:].mean() - ablation_mean[-50:].mean()
    ax.annotate(
        f"Gap: {abs(gap):.1f}",
        xy=(x_len - 1, (adapted_mean[-1] + ablation_mean[-1]) / 2),
        fontsize=10, color="black",
        xytext=(-60, 0), textcoords="offset points",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")


def plot_three_way(adapted, abl_critic, abl_shaping, ylabel, title,
                   output_path, window=20):
    """
    Three-way comparison: Full MAPPO vs local critic vs no reward shaping.
    Shows all conditions on one plot so the relative contribution of each
    component is immediately visible.

    adapted     : (num_seeds, num_updates) full MAPPO
    abl_critic  : (num_seeds, num_updates) local critic ablation
    abl_shaping : (num_seeds, num_updates) no reward shaping ablation
    """
    x_len = min(adapted.shape[1], abl_critic.shape[1], abl_shaping.shape[1])
    adapted     = adapted[:,     :x_len]
    abl_critic  = abl_critic[:,  :x_len]
    abl_shaping = abl_shaping[:, :x_len]

    x      = np.arange(x_len)
    roll_x = x[window - 1:]

    adapted_mean  = adapted.mean(axis=0)
    critic_mean   = abl_critic.mean(axis=0)
    shaping_mean  = abl_shaping.mean(axis=0)

    adapted_std   = adapted.std(axis=0)
    critic_std    = abl_critic.std(axis=0)
    shaping_std   = abl_shaping.std(axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))

    # ── Std bands ─────────────────────────────────────────────────────
    ax.fill_between(x,
                    adapted_mean  - adapted_std,
                    adapted_mean  + adapted_std,
                    alpha=0.10, color="tab:blue")
    ax.fill_between(x,
                    critic_mean   - critic_std,
                    critic_mean   + critic_std,
                    alpha=0.10, color="tab:orange")
    ax.fill_between(x,
                    shaping_mean  - shaping_std,
                    shaping_mean  + shaping_std,
                    alpha=0.10, color="tab:green")

    # ── Rolling means ─────────────────────────────────────────────────
    gap_critic  = adapted_mean[-50:].mean() - critic_mean[-50:].mean()
    gap_shaping = adapted_mean[-50:].mean() - shaping_mean[-50:].mean()

    ax.plot(roll_x, rolling(adapted_mean,  window),
            color="tab:blue",   linewidth=2.5,
            label=f"Full MAPPO — centralized + shaping  "
                  f"(final: {adapted_mean[-50:].mean():.1f})")
    ax.plot(roll_x, rolling(critic_mean,   window),
            color="tab:orange", linewidth=2.5, linestyle="--",
            label=f"Ablation: local critic only  "
                  f"(final: {critic_mean[-50:].mean():.1f}, "
                  f"gap: {gap_critic:+.1f})")
    ax.plot(roll_x, rolling(shaping_mean,  window),
            color="tab:green",  linewidth=2.5, linestyle=":",
            label=f"Ablation: no reward shaping  "
                  f"(final: {shaping_mean[-50:].mean():.1f}, "
                  f"gap: {gap_shaping:+.1f})")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")


def print_summary(adapted_r, ablation_r, adapted_s, ablation_s,
                  label="Ablation"):
    a_r  = adapted_r.mean(axis=0)[-50:].mean()
    ab_r = ablation_r.mean(axis=0)[-50:].mean()
    a_s  = adapted_s.mean(axis=0)[-50:].mean()
    ab_s = ablation_s.mean(axis=0)[-50:].mean()

    print(f"\n=== {label.upper()} SUMMARY "
          f"(mean over last 50 updates, averaged across seeds) ===")
    print(f"{'Metric':<20} {'Full MAPPO':>12} {label:>14} {'Gap':>10}")
    print("-" * 58)
    print(f"{'Reward':<20} {a_r:>12.3f} {ab_r:>14.3f} {a_r - ab_r:>+10.3f}")
    print(f"{'Success Rate':<20} {a_s:>12.3f} {ab_s:>14.3f} {a_s - ab_s:>+10.3f}")
    print()
    print("Positive gap = Full MAPPO outperforms ablation (expected).")
    print("Negative gap = Ablation outperformed — investigate.")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Load all conditions ────────────────────────────────────────────
    adapted_rewards  = load(os.path.join(RUNS_DIR, "simple_spread_rewards2.npy"))
    adapted_success  = load(os.path.join(RUNS_DIR, "simple_spread_coord2.npy"))

    critic_rewards   = load(os.path.join(RUNS_DIR, "simple_spread_reward_ablation.npy"))
    critic_success   = load(os.path.join(RUNS_DIR, "simple_spread_success_ablation.npy"))

    shaping_rewards  = load(os.path.join(RUNS_DIR, "simple_spread_reward_ablation_shaping.npy"))
    shaping_success  = load(os.path.join(RUNS_DIR, "simple_spread_success_ablation_shaping.npy"))

    # ── Individual ablation plots ──────────────────────────────────────

    # Centralized critic ablation
    plot_ablation(
        adapted_rewards, critic_rewards,
        ylabel="Episode Mean Reward",
        title="Ablation: Centralized vs Local Critic — Reward",
        output_path=os.path.join(PLOTS_DIR, "ablation_critic_reward.png"),
    )
    plot_ablation(
        adapted_success, critic_success,
        ylabel="Success Rate",
        title="Ablation: Centralized vs Local Critic — Success Rate",
        output_path=os.path.join(PLOTS_DIR, "ablation_critic_success.png"),
    )

    # Reward shaping ablation
    plot_ablation(
        adapted_rewards, shaping_rewards,
        ylabel="Episode Mean Reward",
        title="Ablation: With vs Without Reward Shaping — Reward",
        output_path=os.path.join(PLOTS_DIR, "ablation_shaping_reward.png"),
    )
    plot_ablation(
        adapted_success, shaping_success,
        ylabel="Success Rate",
        title="Ablation: With vs Without Reward Shaping — Success Rate",
        output_path=os.path.join(PLOTS_DIR, "ablation_shaping_success.png"),
    )

    # ── Three-way comparison (the most useful single figure) ───────────
    plot_three_way(
        adapted_rewards, critic_rewards, shaping_rewards,
        ylabel="Episode Mean Reward",
        title="Ablation Study: Component Contributions — Reward",
        output_path=os.path.join(PLOTS_DIR, "ablation_threeway_reward.png"),
    )
    plot_three_way(
        adapted_success, critic_success, shaping_success,
        ylabel="Success Rate",
        title="Ablation Study: Component Contributions — Success Rate",
        output_path=os.path.join(PLOTS_DIR, "ablation_threeway_success.png"),
    )

    # ── Summaries ─────────────────────────────────────────────────────
    print_summary(adapted_rewards, critic_rewards,
                  adapted_success, critic_success,
                  label="Local Critic")

    print_summary(adapted_rewards, shaping_rewards,
                  adapted_success, shaping_success,
                  label="No Shaping")


if __name__ == "__main__":
    main()