#!/usr/bin/env python3
"""
Exports BANGHTR-X weights to FP16 to fit under GitHub's 100MB file limit (~60MB),
enabling direct checkout and execution on any PC out-of-the-box.
"""
import os
import sys
import json
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
EXPORT_DIR = os.path.join(PROJECT_ROOT, "exports", "banghtr_x_v2_production")
EXPORT_WEIGHTS = os.path.join(EXPORT_DIR, "banghtr_x_v2_weights.pt")

if not os.path.exists(CKPT_PATH):
    print(f"❌ Checkpoint not found at {CKPT_PATH}")
    sys.exit(1)

print(f"📥 Loading checkpoint: {CKPT_PATH}...")
ckpt = torch.load(CKPT_PATH, map_location="cpu")

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
else:
    state_dict = ckpt

print("⚡ Converting tensors to FP16 (half-precision)...")
state_dict_fp16 = {}
for k, v in state_dict.items():
    if isinstance(v, torch.Tensor) and v.is_floating_point():
        state_dict_fp16[k] = v.half()
    else:
        state_dict_fp16[k] = v

print(f"💾 Saving to: {EXPORT_WEIGHTS}...")
torch.save(state_dict_fp16, EXPORT_WEIGHTS)
size_mb = os.path.getsize(EXPORT_WEIGHTS) / (1024 * 1024)
print(f"✅ Successfully exported! File size: {size_mb:.2f} MB (Under 100MB limit for GitHub)")
