import numpy as np
def coordination_metric(env):
    state = env.state()

    state = np.array(state)

    # Simple Spread layout:
    # first 2 values = agent 0 position
    # next 2 = agent 1 position
    # next 2 = agent 2 position

    n_agents = 3
    pos_dim = 2

    positions = []

    for i in range(n_agents):
        start = i * 8   # IMPORTANT: stride depends on MPE layout
        pos = state[start:start+pos_dim]
        positions.append(pos)

    positions = np.array(positions)

    dists = []
    for i in range(n_agents):
        for j in range(i+1, n_agents):
            dists.append(np.linalg.norm(positions[i] - positions[j]))

    return float(np.mean(dists))