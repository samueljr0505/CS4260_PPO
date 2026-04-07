import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# =========================
# LOAD COMBINED FILES
# =========================
all_rewards = np.load("runs/simple_spread_rewards.npy", allow_pickle=True)
all_coord = np.load("runs/simple_spread_coord.npy", allow_pickle=True)

# If saved as (seeds, episodes)
# shape: [num_seeds][num_episodes]
all_rewards = [np.array(r) for r in all_rewards]
all_coord = [np.array(c) for c in all_coord]

# =========================
# ALIGN LENGTHS
# =========================
min_len_r = min(len(r) for r in all_rewards)
all_rewards = [r[:min_len_r] for r in all_rewards]

min_len_c = min(len(c) for c in all_coord)
all_coord = [c[:min_len_c] for c in all_coord]

# =========================
# MEAN + STD
# =========================
reward_mean = np.mean(all_rewards, axis=0)
reward_std = np.std(all_rewards, axis=0)

coord_mean = np.mean(all_coord, axis=0)
coord_std = np.std(all_coord, axis=0)

# =========================
# PLOT 1: REWARD
# =========================
plt.figure()

plt.plot(reward_mean, label="Mean Reward")
plt.fill_between(
    range(len(reward_mean)),
    reward_mean - reward_std,
    reward_mean + reward_std,
    alpha=0.3
)

plt.title("PPO Simple Spread - Reward")
plt.xlabel("Episode")
plt.ylabel("Average Step Reward")
plt.legend()
plt.tight_layout()
plt.savefig("plots/simple_spread_reward.png", dpi=300)
plt.show()

# =========================
# PLOT 2: COORDINATION
# =========================
plt.figure()

plt.plot(coord_mean, label="Mean Coordination (Pairwise Distance)")
plt.fill_between(
    range(len(coord_mean)),
    coord_mean - coord_std,
    coord_mean + coord_std,
    alpha=0.3
)

plt.title("PPO Simple Spread - Coordination Metric")
plt.xlabel("Episode")
plt.ylabel("Mean Pairwise Distance")
plt.legend()
plt.tight_layout()
plt.savefig("plots/simple_spread_coordination.png", dpi=300)
plt.show()