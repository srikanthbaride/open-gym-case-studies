import numpy as np
import random

def set_global_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)

def epsilon_greedy_action(q_row: np.ndarray, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_row))
    return int(np.argmax(q_row))

def linear_decay(start: float, end: float, current_step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return end
    frac = min(1.0, current_step / total_steps)
    return start + frac * (end - start)
