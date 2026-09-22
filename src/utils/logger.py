"""
Experiment logger and environment recorder.
"""

import os
import sys
import logging
import json
import torch

def setup_logger(name: str, log_dir: str = "logs", log_file: str = "experiment.log") -> logging.Logger:
    """Configures multi-handler logger with console and file output."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, log_file), encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def record_environment_snapshot(output_path: str = "logs/env_snapshot.json") -> dict:
    """Captures exact system, PyTorch, CUDA, and GPU versions for research reproducibility."""
    snapshot = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    return snapshot
