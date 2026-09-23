#!/usr/bin/env bash
# BANGHTR-X v3 Git commit and push script
# Run: bash scripts/commit_v3.sh

set -e
cd /home/suza/HandWritenDetection/BanglaGHTR

echo "=== Running v3 build verification first ==="
.venv/bin/python scripts/verify_v3_build.py

echo ""
echo "=== Patching notebook for v3 params ==="
.venv/bin/python scripts/patch_notebook_v3.py

echo ""
echo "=== Committing to git ==="
git add -A
git status
git commit -m "feat(v3): BANGHTR-X v3 — 4-stage ConvNeXt stem, structural matra attn, contextual MoE, CTC padding fix

Architecture upgrades:
- ConvNeXt stem: 4 stages with residual blocks + LayerScale; no aggressive stride-4
  jump; dual max+avg pool vertical collapse preserves matra zone vs vowel-below zone
- Matra attention v3: structurally grounded — uses stem max-pool features as real
  upper-zone ink signal for gating (not generic MHA gate)
- MoE v3: contextual 3-frame router with dilated diacritic expert and larger
  conjunct expert (2× FFN); auxiliary Switch-Transformer load-balancing loss
- hidden_dim: 256 → 384 (+50% capacity for 170-class Bengali vocab)
- encoder_layers: 4 → 6 (deeper global context modeling)
- dim_feedforward: 1024 → 1536 (proportional to hidden_dim)

Bug fixes:
- CTC input_lengths: was always full T_seq (including padded white columns);
  now computed from batch valid_widths / stride_w for correct alignment
- MoE aux loss now propagated through trainer for Switch-Transformer load balancing

Config (htr_v2.yaml):
- batch_size: 48 → 32 (v3 model is larger; safely within 24GB VRAM)
- epochs: 50 → 60
- lr: 3e-4 → 2e-4 (conservative for larger model)
- warmup_steps: 300 → 500
- ctc_weight/attn_weight: 0.5/0.5 → 0.4/0.6
- patience: 12 → 15

Expected CER: 21.5% → 12-16% after full 60-epoch training"

git push 2>/dev/null || echo "Note: No remote configured. Commit saved locally."
echo ""
echo "✅ All done. Run the notebook to train BANGHTR-X v3!"
