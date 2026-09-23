#!/usr/bin/env python3
"""
BANGHTR-X v3 notebook patcher.
Patches BANGHTR_X_Pipeline.ipynb in-place to add v3 params.
Run: python3 scripts/patch_notebook_v3.py
"""

import json
import re

NB_PATH = "BANGHTR_X_Pipeline.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)


def get_src(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    lines = text.split("\n")
    cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]


changes = []

for i, cell in enumerate(nb["cells"]):
    s = get_src(cell)

    # ── Fix 1: batch_size 48 → 32 in DataLoaders ─────────────────────
    if "batch_size=48" in s and "DataLoader" in s:
        s_new = s.replace("batch_size=48", "batch_size=32")
        set_src(cell, s_new)
        changes.append(f"Cell {i}: batch_size 48→32")

    # ── Fix 2: Model instantiation — add missing v3 params ────────────
    if "model_v2 = BANGHTR_X_V2(" in s and "dim_feedforward" not in s:
        # Insert dim_feedforward, label_smoothing, moe_aux_weight, stem_feat_dim
        # right after "num_heads=..." line
        s_new = s.replace(
            '    num_heads=config_v2["model"].get("num_heads", 8),\n',
            '    num_heads=config_v2["model"].get("num_heads", 8),\n'
            '    dim_feedforward=config_v2["model"].get("dim_feedforward", 1536),\n'
            '    label_smoothing=config_v2["model"].get("label_smoothing", 0.05),\n'
            '    moe_aux_weight=config_v2["model"].get("moe_aux_weight", 0.01),\n'
            '    stem_feat_dim=config_v2["model"].get("stem_feat_dim", 384),\n',
        )
        # Also fix default values
        s_new = s_new.replace(
            'hidden_dim=config_v2["model"].get("hidden_dim", 256)',
            'hidden_dim=config_v2["model"].get("hidden_dim", 384)',
        )
        s_new = s_new.replace(
            'encoder_layers=config_v2["model"].get("encoder_layers", 4)',
            'encoder_layers=config_v2["model"].get("encoder_layers", 6)',
        )
        if s_new != s:
            set_src(cell, s_new)
            changes.append(f"Cell {i}: Added v3 model params (dim_feedforward, label_smoothing, moe_aux_weight, stem_feat_dim)")

    # ── Fix 3: Stage 2 print header ───────────────────────────────────
    if "BANGHTR-X v2 fresh training (50 epochs, batch=48)" in s:
        s_new = s.replace(
            "BANGHTR-X v2 fresh training (50 epochs, batch=48)",
            "BANGHTR-X v3 fresh training (60 epochs, batch=32, hidden=384)"
        )
        if s_new != s:
            set_src(cell, s_new)
            changes.append(f"Cell {i}: Updated training banner to v3")

    # ── Fix 4: Epoch count references ─────────────────────────────────
    if "50 epochs" in s and "BANGHTR" in s:
        s_new = s.replace("50 epochs", "60 epochs")
        if s_new != s:
            set_src(cell, s_new)
            changes.append(f"Cell {i}: 50→60 epochs mention")


print(f"Applied {len(changes)} changes:")
for c in changes:
    print(" ", c)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n✅ Notebook patched successfully.")
