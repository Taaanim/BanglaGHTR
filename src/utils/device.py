"""
Hardware and reproducibility utilities.
"""

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(gpu_id: int = 0) -> torch.device:
    """Detects available hardware and returns optimal PyTorch device."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        prop = torch.cuda.get_device_properties(device)
        print(f"[Hardware] Using GPU: {prop.name} (Compute Capability: {prop.major}.{prop.minor}, VRAM: {prop.total_memory / (1024**3):.2f} GB)")
        return device
    print("[Hardware] CUDA unavailable, falling back to CPU.")
    return torch.device("cpu")
