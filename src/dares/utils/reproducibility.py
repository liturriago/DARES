"""Reproducibility helpers: global random seeding."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fixes random seeds for reproducibility across libraries.

    Sets the Python ``random`` module, NumPy and PyTorch (CPU and CUDA) seeds
    and configures cuDNN for deterministic behavior.

    Args:
        seed (int): The seed value to use for all random number generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
