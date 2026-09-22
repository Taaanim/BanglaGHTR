"""
Positional Encoding modules for BANGHTR-X v2.
Provides sinusoidal (encoder) and learned (decoder) positional embeddings.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding as in 'Attention Is All You Need'.
    Used for variable-length encoder sequences (image feature maps).
    """
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # Register as buffer (not a parameter, but saved/loaded with state_dict)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input tensor
        Returns:
            [B, T, D] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embedding for decoder token sequences.
    Used for fixed-length decoder inputs (target text tokens).
    """
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input embeddings
        Returns:
            [B, T, D] with learned positional embedding added
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, T]
        pos_emb = self.embedding(positions)  # [1, T, D]
        return self.dropout(x + pos_emb)
