import gymnasium as gym
import imageio
import torch
import numpy as np
import os

from model import ActorCritic


def load_model(path, obs_dim, act_dim):
    model = ActorCritic(obs_dim, act_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


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

        obs, _, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated

        frames.append(env.render())
        steps += 1

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.mimsave(filename, frames, fps=30)

    print(f"GIF saved to {filename}")


if __name__ == "__main__":

    env = gym.make("HalfCheetah-v5")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model_path = "runs/ppo_halfcheetah_seed0.pt"

    model = load_model(model_path, obs_dim, act_dim)

    record_gif(model)