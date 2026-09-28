#!/usr/bin/env python3
"""
Syncs the best Stage 2 v3 checkpoint to production export directory.
Run: .venv/bin/python scripts/export_production_v3.py
"""

import os
import sys
import json
import datetime
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
EXPORT_DIR = os.path.join(PROJECT_ROOT, "exports", "banghtr_x_v2_production")
EXPORT_WEIGHTS = os.path.join(EXPORT_DIR, "banghtr_x_v2_weights.pt")
EXPORT_META = os.path.join(EXPORT_DIR, "model_meta.json")

os.makedirs(EXPORT_DIR, exist_ok=True)

if not os.path.exists(CKPT_PATH):
    print(f"❌ Checkpoint not found at {CKPT_PATH}")
    sys.exit(1)

print(f"📥 Loading best checkpoint from {CKPT_PATH}...")
ckpt = torch.load(CKPT_PATH, map_location="cpu")

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
    epoch = ckpt.get("epoch", 0)
    metrics = ckpt.get("metrics", {})
else:
    state_dict = ckpt
    epoch = None
    metrics = {}

# Save model weights only (stripping optimizer state to reduce file size)
torch.save(state_dict, EXPORT_WEIGHTS)
size_mb = os.path.getsize(EXPORT_WEIGHTS) / (1024 * 1024)
print(f"Stripped weights exported to: {EXPORT_WEIGHTS} ({size_mb:.1f} MB)")

# Update metadata
meta = {
    "model_name": "BanglaGHTR",
    "num_classes": 170,
    "hidden_dim": 384,
    "encoder_layers": 6,
    "decoder_layers": 4,
    "num_heads": 8,
    "dim_feedforward": 1536,
    "use_matra_attn": True,
    "use_grapheme_moe": True,
    "moe_aux_weight": 0.01,
    "stem_feat_dim": 384,
    "source_checkpoint": "checkpoints/best_model.pt",
    "epoch": epoch,
    "metrics": metrics,
    "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open(EXPORT_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"✅ Metadata saved to: {EXPORT_META}")
print("🎉 Production export synchronized successfully with latest v3 model!")
