import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return returns, advantages


def compute_rewards_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    rewards_to_go = np.zeros_like(rewards)
    running_sum = 0.0
    for i in reversed(range(len(rewards))):
        running_sum = rewards[i] + gamma * running_sum
        rewards_to_go[i] = running_sum
    return rewards_to_go
