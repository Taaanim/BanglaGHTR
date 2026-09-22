"""
Diacritic and Conjunct Expert modules.
Auxiliary feature specialists that focus on fine diacritical marks (কার, ফলা)
and dense compound ligatures (যুক্তবর্ণ).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiacriticExpert(nn.Module):
    """
    Expert network specializing in high-frequency diacritical marks (kars and folas).
    Operates in parallel with the main sequence pathway.
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.expert_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert_net(x)

class ConjunctExpert(nn.Module):
    """
    Expert network specializing in dense compound conjuncts (Yuktakshar).
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.expert_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert_net(x)

class GraphemeMoEFusion(nn.Module):
    """
    Mixture of Experts (MoE) fusion layer that dynamically weights Main, Diacritic,
    and Conjunct representations using a learnable gating router.
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.diacritic_expert = DiacriticExpert(hidden_dim)
        self.conjunct_expert = ConjunctExpert(hidden_dim)

        # 3-way router (Main, Diacritic, Conjunct)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, T, hidden_dim]
        Output: [B, T, hidden_dim]
        """
        weights = self.router(x) # [B, T, 3]

        w_main = weights[..., 0:1]
        w_diac = weights[..., 1:2]
        w_conj = weights[..., 2:3]

        out_diac = self.diacritic_expert(x)
        out_conj = self.conjunct_expert(x)

        fused = w_main * x + w_diac * out_diac + w_conj * out_conj
        return fused
