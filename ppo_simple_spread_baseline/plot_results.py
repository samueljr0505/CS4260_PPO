import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# =========================
# LOAD COMBINED FILES
# =========================
all_rewards = np.load("runs/simple_spread_rewards.npy", allow_pickle=True)
all_coord = np.load("runs/simple_spread_coord.npy", allow_pickle=True)

# Convert to lists of arrays (one per seed)
all_rewards = [np.array(r, dtype=np.float32) for r in all_rewards]
all_coord = [np.array(c, dtype=np.float32) for c in all_coord]

# =========================
# ALIGN LENGTHS
# =========================
min_len_r = min(len(r) for r in all_rewards)
all_rewards = [r[:min_len_r] for r in all_rewards]

min_len_c = min(len(c) for c in all_coord)
all_coord = [c[:min_len_c] for c in all_coord]

# Convert to (seeds, episodes)
all_rewards = np.stack(all_rewards, axis=0)
all_coord = np.stack(all_coord, axis=0)

# =========================
# PLOT FUNCTION
# =========================
def plot_with_seeds(values, title, xlabel, ylabel, filename, y_lim=None):
    x = np.arange(values.shape[1])

    mean = values.mean(axis=0)
    std = values.std(axis=0)

    plt.figure(figsize=(10, 5))

    # -------------------------
    # Individual seeds (lighter)
    # -------------------------
    for i in range(values.shape[0]):
        plt.plot(
            x,
            values[i],
            linewidth=1.0,
            alpha=0.35,
            linestyle="--",
            label=f"Seed {i}"
        )

    # -------------------------
    # Mean (bold)
    # -------------------------
    plt.plot(x, mean, color="black", linewidth=2.8, label="Mean")

    # -------------------------
    # Std band
    # -------------------------
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.2,
        color="black"
    )

    # -------------------------
    # Formatting improvements
    # -------------------------
    if y_lim is not None:
        plt.ylim(y_lim)

    plt.grid(True, linestyle="--", alpha=0.3)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# =========================
# PLOT 1: REWARD
# =========================
plot_with_seeds(
    all_rewards,
    title="PPO Simple Spread - Reward",
    xlabel="Episode",
    ylabel="Episode Reward",
    filename="plots/simple_spread_reward.png",
)

# =========================
# PLOT 2: SUCCESS / COORDINATION
# =========================
plot_with_seeds(
    all_coord,
    title="PPO Simple Spread - Success Rate",
    xlabel="Episode",
    ylabel="Success Rate (0 → 1)",
    filename="plots/simple_spread_success.png",
    y_lim=(0, 0.6),
)

print("Saved plots to plots/")