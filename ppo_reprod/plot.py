import numpy as np
import matplotlib.pyplot as plt

seeds = [0, 1, 2]

# =========================
# LOAD REWARDS
# =========================
all_rewards = []
for s in seeds:
    r = np.load(f"runs/rewards_seed{s}.npy")
    all_rewards.append(r)

min_len_r = min(len(r) for r in all_rewards)
all_rewards = [r[:min_len_r] for r in all_rewards]

reward_mean = np.mean(all_rewards, axis=0)
reward_std = np.std(all_rewards, axis=0)

# =========================
# LOAD VELOCITY
# =========================
all_vel = []
for s in seeds:
    v = np.load(f"runs/velocity_seed{s}.npy")
    all_vel.append(v)

min_len_v = min(len(v) for v in all_vel)
all_vel = [v[:min_len_v] for v in all_vel]

vel_mean = np.mean(all_vel, axis=0)
vel_std = np.std(all_vel, axis=0)

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

plt.title("PPO HalfCheetah-v5 - Episode Return")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend()
plt.savefig("runs/reward_plot.png")
plt.show()

# =========================
# PLOT 2: VELOCITY
# =========================
plt.figure()
plt.plot(vel_mean, label="Mean Forward Velocity")
plt.fill_between(
    range(len(vel_mean)),
    vel_mean - vel_std,
    vel_mean + vel_std,
    alpha=0.3
)

plt.title("PPO HalfCheetah-v5 - Forward Velocity (Success Metric)")
plt.xlabel("Episode")
plt.ylabel("Velocity Proxy (x-final - x-start)")
plt.legend()
plt.savefig("runs/velocity_plot.png")
plt.show()