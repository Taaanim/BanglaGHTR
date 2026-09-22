"""
Matra-Aware Attention Module.
Specialized architectural module that models the headline (মাত্রা / Matra)
continuity across Bengali handwritten sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MatraAttentionModule(nn.Module):
    """
    Computes headline attention along sequence frames and dynamically fusions
    headline continuity features with local character representations.
    """
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multi-head attention across sequence dimension
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Matra-specific gating projection
        self.matra_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, T, hidden_dim]
        Output: [B, T, hidden_dim]
        """
        norm_x = self.norm1(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)

        # Gate attention output based on matra headline alignment
        gate = self.matra_gate(norm_x)
        x = x + gate * attn_out

        # Feedforward refinement
        x = x + self.mlp(self.norm2(x))
        return x
