import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# CONFIG
# =========================
seeds = [0, 1, 2]
window = 50
save_dir = "runs"
os.makedirs(save_dir, exist_ok=True)

# =========================
# HELPERS
# =========================
def rolling_mean_std(x, window=50):
    x = np.asarray(x)

    if len(x) < window:
        raise ValueError("Window larger than data length")

    mean = np.convolve(x, np.ones(window) / window, mode="valid")

    std = np.array([
        np.std(x[i - window:i]) if i >= window else np.nan
        for i in range(len(x))
    ])[window - 1:]

    return mean, std


def load_and_align(file_prefix):
    data = []

    for s in seeds:
        arr = np.load(f"{file_prefix}_seed{s}.npy")
        data.append(arr)

    min_len = min(len(x) for x in data)
    data = [x[:min_len] for x in data]

    return np.array(data)


# =========================
# LOAD DATA
# =========================
all_rewards = load_and_align("runs/rewards")
all_vel = load_and_align("runs/velocity")

reward_mean = np.mean(all_rewards, axis=0)
vel_mean = np.mean(all_vel, axis=0)

# =========================
# ROLLING STATS
# =========================
reward_roll_mean, reward_roll_std = rolling_mean_std(reward_mean, window)
vel_roll_mean, vel_roll_std = rolling_mean_std(vel_mean, window)

# =========================
# PLOT 1: REWARD
# =========================
plt.figure(figsize=(10, 5))

# per-seed curves (important for papers)
for r in all_rewards:
    plt.plot(r, alpha=0.2)

x = np.arange(len(reward_roll_mean))

plt.plot(x, reward_roll_mean, label=f"Rolling Mean (window={window})", linewidth=2)

plt.fill_between(
    x,
    reward_roll_mean - reward_roll_std,
    reward_roll_mean + reward_roll_std,
    alpha=0.25
)

plt.title("PPO HalfCheetah-v5 - Episode Return")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig(f"{save_dir}/reward_plot.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# PLOT 2: VELOCITY
# =========================
plt.figure(figsize=(10, 5))

for v in all_vel:
    plt.plot(v, alpha=0.2)

x = np.arange(len(vel_roll_mean))

plt.plot(x, vel_roll_mean, label=f"Rolling Mean (window={window})", linewidth=2)

plt.fill_between(
    x,
    vel_roll_mean - vel_roll_std,
    vel_roll_mean + vel_roll_std,
    alpha=0.25
)

plt.title("PPO HalfCheetah-v5 - Forward Velocity Proxy")
plt.xlabel("Episode")
plt.ylabel("Velocity")
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig(f"{save_dir}/velocity_plot.png", dpi=300, bbox_inches="tight")
plt.show()

print("Done. Saved plots to runs/")