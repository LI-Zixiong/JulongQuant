"""
Utilities for controlling random seeds and experiment reproducibility.
"""

import os
import random

import numpy as np

try:
    import torch
except ImportError:
    torch = None

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducible experiments.

    This function controls randomness from:
    - Python built-in random module
    - NumPy
    - PyTorch, if installed
    - CUDA, if available

    Parameters
    ----------
    seed : int, default=42
        Random seed used to make experiments more reproducible.

    deterministic : bool, default=True
        Whether to enable deterministic cuDNN behavior in PyTorch.
        This may make training slower but improves reproducibility.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    
    if seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return
    
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)
    print("Python random:", random.randint(0, 100))
    print("NumPy random:", np.random.randn(3))
    if torch is not None:
        print("Torch random:", torch.randn(3))