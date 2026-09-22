"""
Transformer Encoder for BANGHTR-X v2.
Replaces BiLSTM with a multi-layer Transformer encoder for global
context modeling over visual feature sequences.
"""

import torch
import torch.nn as nn
from ..positional_encoding import SinusoidalPositionalEncoding


class TransformerEncoderBlock(nn.Module):
    """
    Single Pre-Norm Transformer Encoder block.
    Pre-norm is preferred over post-norm for stable deep training
    (as shown in 'On Layer Normalization in the Transformer Architecture').
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            src_key_padding_mask: [B, T] True for padded positions
        Returns:
            [B, T, D]
        """
        # Pre-norm self-attention
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(
            norm_x, norm_x, norm_x,
            key_padding_mask=src_key_padding_mask
        )
        x = x + attn_out

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class BanglaTransformerEncoder(nn.Module):
    """
    Multi-layer Transformer encoder with sinusoidal positional encoding.
    Designed to sit on top of the ConvNeXt visual stem + Matra/MoE features.
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.d_model = d_model

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model, max_len=max_seq_len, dropout=dropout
        )

        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final layer norm (pre-norm architecture requires this)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] from visual stem + domain modules
            src_key_padding_mask: [B, T] True for padded positions
        Returns:
            encoder_out: [B, T, D] contextual representations
        """
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        x = self.final_norm(x)
        return x


TransformerEncoder = BanglaTransformerEncoder
