import numpy as np


def coordination_metric(obs_dict):
    positions = []

    for obs in obs_dict.values():
        obs = np.asarray(obs, dtype=np.float32)
        positions.append(obs[2:4])

    positions = np.asarray(positions, dtype=np.float32)

    if len(positions) < 2:
        return 0.0

    pairwise_distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pairwise_distances.append(np.linalg.norm(positions[i] - positions[j]))

    return float(np.mean(pairwise_distances))
