import gymnasium as gym
import torch
import numpy as np
import os
import imageio

from model import ActorCritic
from ppo import PPO
from buffer import RolloutBuffer


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def record_gif(model, env_name="HalfCheetah-v5", filename="runs/cheetah.gif"):
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()

    frames = []
    done = False
    steps = 0

    while not done and steps < 1000:
        obs_t = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            action, _ = model.get_action(obs_t)

        obs, _, term, trunc, _ = env.step(action.numpy())
        done = term or trunc

        frame = env.render()
        frames.append(frame)

        steps += 1

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.mimsave(filename, frames, fps=30)
    print(f"GIF saved to {filename}")


def train(seed):
    set_seed(seed)

    env = gym.make("HalfCheetah-v5")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim)
    ppo = PPO(model)
    buffer = RolloutBuffer()

    episode_rewards = []
    episode_velocities = []

    obs, _ = env.reset(seed=seed)

    ep_reward = 0
    x_positions = []

    total_steps = 5_000_000

    for step in range(total_steps):

        obs_t = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            action, logprob = model.get_action(obs_t)
            value = model.critic(model.shared(obs_t)).squeeze()

        next_obs, reward, terminated, truncated, _ = env.step(action.numpy())

        done = terminated or truncated

        buffer.add(obs_t, action, logprob, reward, done, value)

        # track success metric (forward progress)
        x_positions.append(env.unwrapped.data.qpos[0])

        obs = next_obs
        ep_reward += reward

        if done:
            episode_rewards.append(ep_reward)

            # SUCCESS METRIC: forward velocity proxy
            forward_velocity = x_positions[-1] - x_positions[0]
            episode_velocities.append(forward_velocity)

            obs, _ = env.reset()
            ep_reward = 0
            x_positions = []

        # PPO update
        if step % 2048 == 0 and step > 0:
            ppo.update(buffer)
            buffer.clear()

            print(
                f"[Seed {seed}] Step {step} | "
                f"Avg Reward: {np.mean(episode_rewards[-10:]) if episode_rewards else 0:.3f} | "
                f"Avg Vel: {np.mean(episode_velocities[-10:]) if episode_velocities else 0:.3f}"
            )

    os.makedirs("runs", exist_ok=True)

    torch.save(model.state_dict(), f"runs/ppo_halfcheetah_seed{seed}.pt")
    np.save(f"runs/rewards_seed{seed}.npy", np.array(episode_rewards))
    np.save(f"runs/velocity_seed{seed}.npy", np.array(episode_velocities))


if __name__ == "__main__":

    seeds = [0, 1, 2]

    for seed in seeds:
        train(seed)