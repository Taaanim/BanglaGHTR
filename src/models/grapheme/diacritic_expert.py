"""
Diacritic and Conjunct Expert modules — BANGHTR-X v3.
Upgrades over v2:
  - Contextual router: 3-frame conv window so router can see neighboring tokens
    (distinguishes 'ক' from 'কা' — vowel-kar spans adjacent columns)
  - Diacritic expert: uses dilated convolution for spatially spread vowel signs
  - Conjunct expert: larger capacity FFN for fused stroke clusters
  - Auxiliary load-balancing loss to prevent router collapse (all routes to one expert)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiacriticExpert(nn.Module):
    """
    Expert specializing in high-frequency diacritical marks (kars and folas).
    Uses a dilated 1D conv to capture spread-out vowel sign patterns across time.
    """
    def __init__(self, hidden_dim: int = 384):
        super().__init__()
        # Temporal dilated conv to capture marks that span ≥2 columns
        self.dilated_conv = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=2, dilation=2, groups=hidden_dim
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.expert_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        # Dilated temporal convolution
        residual = x
        xc = x.permute(0, 2, 1)                    # [B, D, T]
        xc = self.dilated_conv(xc).permute(0, 2, 1)  # [B, T, D]
        x = self.norm1(residual + xc)
        x = self.norm2(x + self.expert_ffn(x))
        return x


class ConjunctExpert(nn.Module):
    """
    Expert specializing in dense compound conjuncts (Yuktakshar).
    Larger FFN capacity to model complex fused strokes.
    """
    def __init__(self, hidden_dim: int = 384):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.expert_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),   # 2× expansion
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        return x + self.expert_ffn(self.norm1(x))


class GraphemeMoEFusion(nn.Module):
    """
    BANGHTR-X v3 Mixture-of-Experts fusion.

    Improvements over v2:
      1. Contextual router: 3-frame depthwise temporal conv before gating,
         so the router can distinguish a standalone consonant from one followed by a vowel-kar.
      2. Redesigned diacritic expert with dilated convolution.
      3. Larger conjunct expert with 2× FFN capacity.
      4. Auxiliary load-balancing loss (stored as self.aux_loss) to prevent
         all tokens routing to a single expert.
    """
    def __init__(self, hidden_dim: int = 384):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.diacritic_expert = DiacriticExpert(hidden_dim)
        self.conjunct_expert = ConjunctExpert(hidden_dim)

        # Contextual pre-router: 3-frame depthwise temporal conv for local context
        self.context_conv = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1, groups=hidden_dim
        )
        self.context_norm = nn.LayerNorm(hidden_dim)

        # 3-way router on contextual features
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 3),
        )

        # Stored for use in loss computation (initialized to zero)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [B, T, hidden_dim]
        Output: [B, T, hidden_dim]

        Side effect: updates self.aux_loss (load-balancing auxiliary loss).
        """
        B, T, D = x.shape

        # Compute contextual features for routing
        ctx = x.permute(0, 2, 1)                              # [B, D, T]
        ctx = self.context_conv(ctx).permute(0, 2, 1)         # [B, T, D]
        ctx = self.context_norm(ctx)

        # Router softmax weights
        router_logits = self.router(ctx)                       # [B, T, 3]
        weights = F.softmax(router_logits, dim=-1)             # [B, T, 3]

        w_main = weights[..., 0:1]   # [B, T, 1]
        w_diac = weights[..., 1:2]
        w_conj = weights[..., 2:3]

        # Expert outputs
        out_diac = self.diacritic_expert(x)
        out_conj = self.conjunct_expert(x)

        # Weighted mixture
        fused = w_main * x + w_diac * out_diac + w_conj * out_conj

        # ── Auxiliary Load-Balancing Loss ────────────────────────────────────
        # Encourages equal usage of all 3 experts.
        # Loss = num_experts * sum(f_i * P_i) where f_i is fraction of tokens
        # routed to expert i, and P_i is mean routing probability for expert i.
        # (Switch Transformer style)
        num_experts = 3
        # Mean routing prob per expert: [3]
        mean_probs = weights.mean(dim=[0, 1])       # [3]
        # Fraction of tokens most-routed to each expert: [3]
        top_expert = weights.argmax(dim=-1)          # [B, T]
        frac_tokens = torch.stack([
            (top_expert == i).float().mean() for i in range(num_experts)
        ])
        self.aux_loss = num_experts * (frac_tokens * mean_probs).sum()
        # ─────────────────────────────────────────────────────────────────────

        return fused
