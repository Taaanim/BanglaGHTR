"""
Matra-Aware Attention Module — BANGHTR-X v3.
Upgrades over v2:
  - Now receives the max-pool spatial feature from the ConvNeXt stem via optional
    injection. The max-pool activations correspond to where peak ink intensity is
    located vertically per column — directly correlating with the matra (headline)
    presence in the upper zone of each character frame.
  - Uses this as a structural gating signal: columns where max-pool activations
    are strong in the upper band are gated UP (strong matra), weak activation =
    matra discontinuity (boundary between words or chars without matra like গ, ঙ).
  - Pre-norm MHA + gated FFN remain from v2 but gating is now structurally grounded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatraAttentionModule(nn.Module):
    """
    BANGHTR-X v3 Structurally-Grounded Matra Attention.

    Forward args:
        x         : [B, T, D] — sequence from ConvNeXt stem
        max_feat  : [B, T, D_stem] — optional max-pool feature from stem
                    If provided, it is projected and used as a structural gate.
                    If None, falls back to a learned content-based gate (v2 style).
    """
    def __init__(self, hidden_dim: int = 384, num_heads: int = 8, stem_feat_dim: int = 384):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Pre-norm multi-head self-attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True
        )

        # Structural matra gate: project stem max-feat → gate signal
        self.matra_proj = nn.Sequential(
            nn.Linear(stem_feat_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Fallback content gate (when no max_feat provided — v2 style)
        self.content_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Post-attn FFN
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, max_feat: torch.Tensor = None) -> torch.Tensor:
        """
        Input:
            x        : [B, T, hidden_dim]
            max_feat : [B, T, stem_feat_dim]  (from stem.get_max_feat())
        Output:
            [B, T, hidden_dim]
        """
        norm_x = self.norm1(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)

        # Compute matra gate
        if max_feat is not None:
            # Structural gate: based on real vertical max-pool (top-zone ink presence)
            gate = self.matra_proj(max_feat)    # [B, T, D]
        else:
            # Content gate fallback (v2 style)
            gate = self.content_gate(norm_x)    # [B, T, D]

        # Gated residual: attention is only applied where matra is structurally strong
        x = x + gate * attn_out

        # FFN with pre-norm
        x = x + self.mlp(self.norm2(x))
        return x
