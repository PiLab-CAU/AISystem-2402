# utils/seed_utils.py

import random
import numpy as np
import torch

def set_global_seed(seed: int = 42) -> None:
    """
    Set seed for reproducibility.
    
    Args:
        seed: Seed value for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False