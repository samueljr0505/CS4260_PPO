import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    from mpe2 import simple_spread_v3
except ImportError:
    from pettingzoo.mpe import simple_spread_v3

from model import MultiAgentActorCritic
from utils import success_rate


def find_best_seed(model, seed_range=range(0, 500), num_agents=3,
                   num_landmarks=3, max_cycles=25):
    """
    Searches for a seed where:
    1. Agents start far from landmarks (interesting navigation)
    2. Episode reward is reasonably good (agents actually succeed)
    """
    try:
        from mpe2 import simple_spread_v3 as env_module
    except ImportError:
        from pettingzoo.mpe import simple_spread_v3 as env_module

    results = []

    model.eval()

    for seed in seed_range:
        env = env_module.parallel_env(
            N=num_agents,
            max_cycles=max_cycles,
            continuous_actions=False,
            local_ratio=0.0,
        )
        obs_dict, _ = env.reset(seed=int(seed))

        # Measure initial distance from each agent to nearest landmark
        total_initial_dist = 0.0
        for agent in env.agents:
            obs = obs_dict[agent]
            min_dist = min(
                ((obs[4 + i*2]**2 + obs[4 + i*2 + 1]**2)**0.5)
                for i in range(num_landmarks)
            )
            total_initial_dist += min_dist

        # Run episode
        ep_reward  = 0.0
        ep_success = []

        for _ in range(max_cycles):
            if not env.agents:
                break
            action_dict = {}
            for agent in env.agents:
                obs = torch.tensor(obs_dict[agent], dtype=torch.float32)
                with torch.no_grad():

                    # Use soft sampling with low temperature:
                    temperature = 0.3  # lower = more greedy, higher = more random
                    logits = model.actor(obs) / temperature
                    soft_dist = torch.distributions.Categorical(logits=logits)
                    action = soft_dist.sample()
                action_dict[agent] = int(action.item())

            next_obs, rewards, terminations, truncations, _ = env.step(action_dict)
            done = any(terminations.values()) or any(truncations.values())
            ep_reward  += float(sum(rewards.values()))
            ep_success.append(success_rate(next_obs, num_landmarks=num_landmarks))
            obs_dict = next_obs
            if done:
                break

        env.close()
        mean_success = float(np.mean(ep_success)) if ep_success else 0.0
        results.append((seed, total_initial_dist, ep_reward, mean_success))

    model.train()

    # Filter: only seeds where reward is decent AND success is meaningful
    # Require reward > -150 and success > 0.15 to ensure agents actually navigate
    good = [(s, d, r, sr) for s, d, r, sr in results if r > -150 and sr > 0.15]

    if not good:
        # Relax threshold if nothing qualifies
        good = [(s, d, r, sr) for s, d, r, sr in results if r > -170 and sr > 0.10]

    if not good:
        print("Warning: no good seeds found, using seed with best reward")
        good = sorted(results, key=lambda x: x[2], reverse=True)[:10]

    # Among good seeds, pick the one with largest initial distance
    best = max(good, key=lambda x: x[1])
    seed, dist, reward, succ = best

    print(f"\nBest demo seed: {seed}")
    print(f"  Initial distance: {dist:.3f}")
    print(f"  Episode reward:   {reward:.2f}")
    print(f"  Success rate:     {succ:.3f}")

    return seed


def run_demo(model, num_episodes=1, seed=999, num_agents=3,
             num_landmarks=3, max_cycles=25):

    try:
        from mpe2 import simple_spread_v3 as env_module
    except ImportError:
        from pettingzoo.mpe import simple_spread_v3 as env_module

    env = env_module.parallel_env(
        N=num_agents,
        max_cycles=max_cycles,
        continuous_actions=False,
        local_ratio=0.0,
        render_mode="rgb_array",
    )

    model.eval()
    all_frames  = []
    all_rewards = []
    all_success = []

    for ep in range(num_episodes):
        obs_dict, _ = env.reset(seed=int(seed) if ep == 0 else None)
        frames     = []
        ep_reward  = 0.0
        ep_success = []

        # Capture initial frame
        frame = env.render()
        if frame is not None:
            frames.append(frame.copy())

        for _ in range(max_cycles):
            if not env.agents:
                break

            action_dict = {}
            for agent in env.agents:
                obs = torch.tensor(obs_dict[agent], dtype=torch.float32)
                with torch.no_grad():
                    dist_t = model.policy(obs)
                    action  = dist_t.probs.argmax()
                action_dict[agent] = int(action.item())

            next_obs, rewards, terminations, truncations, _ = env.step(action_dict)
            done = any(terminations.values()) or any(truncations.values())

            ep_reward  += float(sum(rewards.values()))
            ep_success.append(success_rate(next_obs, num_landmarks=num_landmarks))

            # Capture frame AFTER step so we see the result of the action
            frame = env.render()
            if frame is not None:
                frames.append(frame.copy())

            obs_dict = next_obs
            if done:
                break

        all_frames.append(frames)
        all_rewards.append(ep_reward)
        all_success.append(float(np.mean(ep_success)) if ep_success else 0.0)

        print(f"Episode {ep:02d} | "
              f"Reward: {ep_reward:7.2f} | "
              f"Success: {all_success[-1]:.3f} | "
              f"Frames: {len(frames)}")

    env.close()
    model.train()
    return all_frames, all_rewards, all_success


def save_gif(frames, path, fps=12):
    """
    Saves a list of RGB numpy arrays as a smooth animated GIF.

    fps=12 gives fluid motion — increase to 15 for faster playback,
    decrease to 8 if you want to see each step more clearly.
    """
    if not frames:
        print("No frames to save.")
        return

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # no whitespace
    img = ax.imshow(frames[0])

    def update(frame):
        img.set_data(frame)
        return [img]

    ani = animation.FuncAnimation(
        fig, update,
        frames=frames,
        interval=1000 // fps,  # ms between frames
        blit=True,
        repeat=True,
    )

    ani.save(path, writer="pillow", fps=fps)
    plt.close()
    print(f"Saved GIF: {path}  ({len(frames)} frames @ {fps}fps)")

def main():
    os.makedirs("demos", exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────
    try:
        from mpe2 import simple_spread_v3 as env_module
    except ImportError:
        from pettingzoo.mpe import simple_spread_v3 as env_module

    env = env_module.parallel_env(N=3, max_cycles=25,
                                   continuous_actions=False,
                                   local_ratio=0.0)
    obs_dict, _ = env.reset()
    first_agent = env.agents[0]
    obs_dim   = obs_dict[first_agent].shape[0]
    act_dim   = int(env.action_space(first_agent).n)
    state_dim = np.asarray(env.state()).shape[0]
    env.close()

    model = MultiAgentActorCritic(obs_dim=obs_dim, act_dim=act_dim,
                                   state_dim=state_dim)
    checkpoint = torch.load("pt_files/model_seed0.pt", map_location="cpu")
    model.load_state_dict(checkpoint)
    print("Loaded model from pt_files/model_seed2.pt")

    # ── Find seed with dramatic navigation behavior ────────────────────
    print("Searching for best demo seed (agents start far, still succeed)...")
    demo_seed = find_best_seed(model, seed_range=range(0, 200))

    if demo_seed is None:
        print("No good seed found, falling back to seed 42")
        demo_seed = 42

    # ── Run and save GIF with found seed ──────────────────────────────
    frames_list, rewards, success = run_demo(
        model,
        num_episodes=1,
        seed=demo_seed,
    )

    save_gif(
        frames_list[0],
        path="demos/demo2.gif",
        fps=8,  # slightly slower so movement is clearly visible
    )

    print(f"\nSaved demos/demo_navigation.gif")
    print(f"Seed {demo_seed} | Reward: {rewards[0]:.2f} | Success: {success[0]:.3f}")

if __name__ == "__main__":
    main()