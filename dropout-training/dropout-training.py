import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x = np.asarray(x, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    if p >= 1.0:
        return np.zeros_like(x), np.zeros_like(x)
    pattern = (rng.random(x.shape) < 1 - p) / (1 - p)
    return x * pattern, pattern