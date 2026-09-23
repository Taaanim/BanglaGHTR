#!/usr/bin/env python3
"""
BANGHTR-X v3 Build Verification Script.
Run from project root: python3 scripts/verify_v3_build.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

print("=" * 60)
print("BANGHTR-X v3 Build Verification")
print("=" * 60)

# ── Test 1: ConvNeXt v3 Stem ─────────────────────────────────────
print("\n[1] Testing ConvNeXt v3 Stem (4-stage, dual pool)...")
from src.models.vision.backbones import ConvNeXtStem

stem = ConvNeXtStem(in_channels=1, hidden_dim=384)
x = torch.randn(2, 1, 64, 512)
feat = stem(x)
max_feat = stem.get_max_feat()
print(f"    Input:     {list(x.shape)}")
print(f"    Output:    {list(feat.shape)}  (expected [2, ~128, 384])")
print(f"    max_feat:  {list(max_feat.shape)}  (expected [2, ~128, 384])")
assert feat.shape == (2, feat.shape[1], 384), "❌ Stem output shape wrong!"
print("    ✅ Stem OK")

# ── Test 2: Matra Attention v3 ───────────────────────────────────
print("\n[2] Testing Matra Attention v3 (structural gating)...")
from src.models.grapheme.matra_attention import MatraAttentionModule

matra = MatraAttentionModule(hidden_dim=384, num_heads=8, stem_feat_dim=384)
T = feat.shape[1]
matra_out = matra(feat, max_feat=max_feat)
print(f"    Output: {list(matra_out.shape)}  (expected [2, {T}, 384])")
assert matra_out.shape == feat.shape, "❌ Matra output shape mismatch!"
print("    ✅ Matra Attention OK")
# Test fallback (no max_feat)
fallback_out = matra(feat, max_feat=None)
assert fallback_out.shape == feat.shape
print("    ✅ Fallback (content gate) OK")

# ── Test 3: MoE v3 ──────────────────────────────────────────────
print("\n[3] Testing MoE v3 (contextual router + aux loss)...")
from src.models.grapheme.diacritic_expert import GraphemeMoEFusion

moe = GraphemeMoEFusion(hidden_dim=384)
moe_out = moe(feat)
print(f"    Output:   {list(moe_out.shape)}")
print(f"    aux_loss: {moe.aux_loss.item():.4f}  (should be ~1.0 initially)")
assert moe_out.shape == feat.shape, "❌ MoE output shape wrong!"
print("    ✅ MoE OK")

# ── Test 4: Full BANGHTR-X v3 Model ─────────────────────────────
print("\n[4] Testing full BANGHTR_X_V2 (v3 config)...")
from src.models.banghtr_x import BANGHTR_X_V2

model = BANGHTR_X_V2(
    num_classes=170,
    in_channels=1,
    hidden_dim=384,
    encoder_layers=6,
    decoder_layers=4,
    num_heads=8,
    dim_feedforward=1536,
    use_matra_attn=True,
    use_grapheme_moe=True,
    decoder_type="hybrid",
    moe_aux_weight=0.01,
    stem_feat_dim=384,
)
model.eval()
imgs = torch.randn(2, 1, 64, 512)
with torch.no_grad():
    out = model(imgs)

print(f"    ctc_log_probs:  {list(out['ctc_log_probs'].shape)}")
assert out["ctc_log_probs"].shape[2] == 170, "❌ CTC num_classes wrong!"
print("    ✅ Full model forward OK")

# ── Test 5: Training forward (with targets) ──────────────────────
print("\n[5] Testing training forward (with attention targets)...")
model.train()
targets = torch.randint(5, 170, (2, 12))
out_train = model(imgs, target_tokens=targets, mode="train")
assert "attn_loss" in out_train
assert "moe_aux_loss" in out_train
print(f"    attn_loss:     {out_train['attn_loss'].item():.4f}")
print(f"    moe_aux_loss:  {out_train['moe_aux_loss'].item():.4f}")
print("    ✅ Training forward OK")

# ── Test 6: Parameter Count ──────────────────────────────────────
print("\n[6] Parameter count breakdown...")
counts = model.count_parameters()
for k, v in counts.items():
    print(f"    {k:<22}: {v:>12,}")

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED — BANGHTR-X v3 is ready to train!")
print("=" * 60)
print("\nNext step: Run the notebook BANGHTR_X_Pipeline.ipynb")
print("Expected CER after full training: 12-16%")
