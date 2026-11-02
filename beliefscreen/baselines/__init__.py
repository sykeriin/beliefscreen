import numpy as np


def random_scores(n: int, seed: int = 0) -> np.ndarray:
    """Filtering control: random ranking."""
    return np.random.default_rng(seed).random(n)
