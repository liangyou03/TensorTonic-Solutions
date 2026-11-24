import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    w,s,g = (np.asarray(i,dtype = float) for i in [w,s,g])
    new_s = beta*s + (1-beta) * g * g
    new_w = w - g * lr / np.sqrt(new_s+eps)
    return new_w,new_s
    pass