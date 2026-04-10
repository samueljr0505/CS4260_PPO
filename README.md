# CS4260_Final_Project_PPO
# CS4260 Final Project — PPO: Reproduction and Multi-Agent Adaptation

**Track 2: PPO**
**Canonical Paper**: Proximal Policy Optimization Algorithms (Schulman et al., 2017)
**Reproduction Target**: HalfCheetah-v5 (MuJoCo)
**Adaptation Target**: MPE2 Simple Spread (Multi-Agent Coordination)

---

## Project Overview

This project reproduces single-agent PPO on the MuJoCo HalfCheetah-v5 locomotion benchmark, then adapts it to a cooperative multi-agent setting using MAPPO (Multi-Agent PPO) on the MPE2 Simple Spread environment.

The adaptation extends PPO with:
- **Centralized critic** using global environment state (CTDE — Centralized Training, Decentralized Execution)
- **Coverage-based reward shaping** to provide dense per-agent coordination signal
- **Entropy scheduling** to prevent premature policy convergence
- **Separate actor/critic learning rates** and clipped value loss for training stability

---

## Repository Structure

```
CS4260_Final_Project_PPO/
│
├── ppo_reprod/                  # Phase 1: Canonical PPO reproduction
│   ├── train.py                      # PPO training on HalfCheetah-v5
│   ├── model.py                      # Actor-Critic network
│   ├── ppo.py                        # PPO update logic
│   ├── buffer.py                     # Rollout buffer
|   ├── plot.py                       # plotting script
|   ├── gif.py                        # gif creator script
|   ├── images/                       # Saved images and gif from execution
│   └── runs/                         # Saved training data (.npy files) and other execution files
|   
│
├── ppo_simple_spread_baseline/       # Phase 2a: Unmodified PPO on Simple Spread
│   ├── train_simple_spread.py        # Naive single-agent PPO baseline
│   ├── model.py                      # Standard ActorCritic
│   ├── utils.py                      # success_rate metric
|   ├── random_baseline.py            # random baseline script to compare training with
|   ├── plot_results.py               # plotting script for baseline
|   ├── plots/                        # all plots from scripts 
│   └── runs/                         # Baseline results
│
├── ppo_simple_spread_adapted/        # Phase 2b: MAPPO adaptation
│   ├── train.py                      # MAPPO training with ablation flag
│   ├── model.py                      # MultiAgentActorCritic (centralized critic)
│   ├── ppo.py                        # MultiAgentPPO with entropy scheduling
│   ├── buffer.py                     # MultiAgentRolloutBuffer with GAE
│   ├── utils.py                      # success_rate metric
│   ├── evaluate.py                   # Validation on held-out seeds
│   ├── render_demo.py                # GIF demo generation
│   │
│   ├── runs/                         # Training results
│   │   ├── simple_spread_rewards2.npy          # Full MAPPO rewards
│   │   ├── simple_spread_coord2.npy            # Full MAPPO success rate
│   │   ├── simple_spread_reward_ablation.npy   # Local critic ablation
│   │   ├── simple_spread_success_ablation.npy
│   │   ├── simple_spread_reward_ablation_shaping.npy  # No shaping ablation
│   │   ├── simple_spread_success_ablation_shaping.npy
│   │   └── validation_results.npy              # Held-out validation
│   │
│   ├── pt_files/                     # Saved model checkpoints
│   │   ├── model_seed0.pt
│   │   ├── model_seed1.pt
│   │   └── model_seed2.pt
│   │
│   ├── plots/                        # All generated figures
│   │   ├── simple_spread_reward2.png
│   │   ├── simple_spread_success2.png
│   │   ├── validation_reward.png
│   │   ├── validation_success.png
│   │   ├── ablation_critic_reward.png
│   │   ├── ablation_critic_success.png
│   │   ├── ablation_shaping_reward.png
│   │   ├── ablation_shaping_success.png
│   │   ├── ablation_threeway_reward.png
│   │   └── ablation_threeway_success.png
│   │
│   ├── demos/
│   │   └── demo_navigation.gif       # Qualitative agent behavior demo
│   │
│   └── plot_scripts/
│       ├── plot.py                   # Training curve plots
│       ├── plot_validation.py        # Training vs validation plots
│       └── plot_ablation.py          # Ablation comparison plots
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/CS4260_PPO.git
cd CS4260_Final_Project_PPO
```

### 2. Create and Activate Conda Environment

```bash
conda create -n cs4260_ppo python=3.9
conda activate cs4260_ppo
```

### 3. Install Dependencies

```bash
pip install torch torchvision
pip install gymnasium
pip install mujoco
pip install mpe2
pip install pettingzoo
pip install numpy matplotlib pillow
```

> **Note**: If `mpe2` is not available, the code automatically falls back to `pettingzoo.mpe`. Both are supported but will transition later on.

### 4. Verify MuJoCo Installation

```bash
python -c "import gymnasium; env = gymnasium.make('HalfCheetah-v5'); print('MuJoCo OK')"
```

### 5. Verify MPE2 Installation

```bash
python -c "from mpe2 import simple_spread_v3; print('MPE2 OK')"
```

---
## Running the Code

### Phase 1 — Canonical PPO Reproduction (HalfCheetah-v5)

```bash
cd ppo_halfcheetah
python train.py
```

Results saved to `runs/`. Training runs 3 seeds by default.
Run `python plot.py` to generate reward and forward velocity plots.

---

### Phase 2a — Baseline PPO on Simple Spread

```bash
cd ppo_simple_spread_baseline
python train.py
```

Runs unmodified single-agent PPO with continuous actions on Simple Spread across 3 seeds.
Results saved to:
- `runs/simple_spread_rewards.npy`
- `runs/simple_spread_coord.npy`

---

### Phase 2b — Adapted MAPPO on Simple Spread

All training modes are controlled via a single command line argument.
Files are named automatically — nothing overwrites anything.

#### Full MAPPO (centralized critic + reward shaping)

```bash
cd ppo_simple_spread_adapted
python train.py full
```

Results saved to:
- `runs/simple_spread_rewards_mappo_full.npy`
- `runs/simple_spread_success_mappo_full.npy`
- `pt_files/model_mappo_full_seed{0,1,2}.pt`

#### Ablation — Local Critic Only (centralized critic removed)

```bash
python train.py ablation_critic
```

Results saved to:
- `runs/simple_spread_rewards_ablation_critic.npy`
- `runs/simple_spread_success_ablation_critic.npy`
- `pt_files/model_ablation_critic_seed{0,1,2}.pt`

#### Ablation — No Reward Shaping

```bash
python train.py ablation_shaping
```

Results saved to:
- `runs/simple_spread_rewards_ablation_shaping.npy`
- `runs/simple_spread_success_ablation_shaping.npy`
- `pt_files/model_ablation_shaping_seed{0,1,2}.pt`

#### All modes at a glance

| Command | Centralized Critic | Reward Shaping | File Suffix |
|---|---|---|---|
| `python train.py full` | ✓ | ✓ | `mappo_full` |
| `python train.py ablation_critic` | ✗ | ✓ | `ablation_critic` |
| `python train.py ablation_shaping` | ✓ | ✗ | `ablation_shaping` |
| `python train.py ablation_both` | ✗ | ✗ | `ablation_both` |

---

### Validation

```bash
python evaluate.py
```

Evaluates each trained seed on held-out environment seeds (999, 1000, 1001) with no gradient updates and greedy action selection. Results saved to `runs/validation_results.npy`.

---

### Generate Plots

```bash
# Training curves (reward + success rate)
python plot_training.py

# Training vs validation side-by-side
python plot_validation.py

# Ablation study (two-way and three-way comparisons)
python plot_ablation.py
```

All plots saved to `plots/`.

---

### Generate Demo GIF

```bash
python render_demo.py
```

Searches 1000 seeds to find an episode where landmarks are well separated and all three agents navigate to unique landmarks. GIF saved to `demos/demo_navigation.gif`. Open in a browser to view — GitHub preview will show only the first frame.
## Key Hyperparameters

| Parameter              | Value |
|------------------------|---|
| Rollout steps          | 4096 |
| PPO epochs             | 15 |
| Batch size             | 256 |
| Actor learning rate    | 3e-4 |
| Critic learning rate   | 1e-3 |
| Clip epsilon           | 0.2 |
| GAE lambda             | 0.95 |
| Discount gamma         | 0.99 |
| Entropy coef (initial) | 0.01 |
| Entropy coef (final)   | 0.003 |
| Max grad norm          | 0.5 |
| Coverage bonus         | 1.0 |
| Coverage threshold     | 0.15 |
| Seeds                  | 0, 1, 2 |
| Episodes               | 300 |
| Max cycles per episode | 25 |
| Num agents             | 3 |
| Num landmarks          | 3 |

---

## Environment Details

| Setting | Canonical (HalfCheetah) | Baseline | Adapted MAPPO |
|---|---|---|---|
| Environment | HalfCheetah-v5 | Simple Spread | Simple Spread |
| Action space | Continuous | Continuous | Discrete |
| Agents | 1 | 3 (independent) | 3 (coordinated) |
| Critic input | Local obs | Local obs | Global state |
| Reward shaping | No | No | Yes |

---

## Results Summary

### Training Performance (mean over last 50 updates, 3 seeds)

| Condition | Reward | Success Rate |
|---|---|---|
| Baseline (unmodified PPO) | ~-80 (no learning) | ~0.10 (no trend) |
| Full MAPPO | -110.5 | 0.221 |
| Ablation: local critic | -111.9 | 0.199 |
| Ablation: no shaping | -120.5 | ~0.10 |

### Validation (100 episodes, held-out seeds)

| Metric | Training | Validation | Gap |
|---|---|---|---|
| Reward | -110.5 | -115.7 | 4.8% |
| Success Rate | 0.221 | 0.191 | 13.6% |

---

## Known Limitations

**Clustering behavior**: Agents occasionally converge on nearby landmarks while leaving a distant landmark uncovered. This occurs because agents act on local observations at execution time and cannot observe whether a teammate has already claimed a landmark. This is a known limitation of decentralized execution in MAPPO and is expected behavior at this training scale. Addressing it fully would require explicit communication protocols or role assignment mechanisms, which are identified as future work.

**Small centralized critic gap**: The centralized critic produces a consistent but modest improvement (1.4 reward points, 0.021 success rate) over a local critic. This is consistent with the Simple Spread environment where each agent's local observation already encodes the positions of all landmarks and other agents, limiting the information gain from global state access. The benefit of centralization would be more pronounced in larger environments with more agents or true partial observability.

---

## Dependencies

| Package | Version |
|---|---|
| Python | 3.9 |
| PyTorch | ≥ 2.0 |
| Gymnasium | ≥ 0.29 |
| MuJoCo | ≥ 3.0 |
| MPE2 / PettingZoo | latest |
| NumPy | ≥ 1.24 |
| Matplotlib | ≥ 3.7 |
| Pillow | ≥ 9.0 |

---

## References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347
- Yu, C., Velu, A., Vinitsky, E., Wang, Y., Bayen, A., & Wu, Y. (2022). *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*. NeurIPS 2022
- Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*. NeurIPS 2017