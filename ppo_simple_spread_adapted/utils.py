import numpy as np

def success_rate(obs_dict, num_landmarks=3, threshold=0.2):
    """
    Success = every landmark is within `threshold`
    distance of at least one agent.
    """

    if not obs_dict:
        return 0.0

    agent_pos = []
    landmark_pos = None

    for obs in obs_dict.values():
        obs = np.asarray(obs, dtype=np.float32)

        # agent position
        agent_pos.append(obs[2:4])

        # landmark relative positions
        lm = obs[4:4 + 2 * num_landmarks].reshape(num_landmarks, 2)

        # convert relative -> absolute (consistent across agents)
        if landmark_pos is None:
            landmark_pos = lm + obs[2:4]

    agent_pos = np.array(agent_pos)

    success = 0

    for lm in landmark_pos:
        dists = np.linalg.norm(agent_pos - lm, axis=1)
        if np.min(dists) < threshold:
            success += 1

    return success / num_landmarks