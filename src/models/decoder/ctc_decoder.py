"""
CTC Sequence Decoder for Handwritten Text Recognition.
Supports greedy decoding and log-probability emission for CTC loss.
"""

import torch
import torch.nn as nn
from typing import List

class CTCDecoder(nn.Module):
    """
    Projects hidden representations to vocabulary logits and performs CTC decoding.
    Blank token is expected at index 0.
    """
    def __init__(self, hidden_dim: int, num_classes: int, blank_idx: int = 0):
        super().__init__()
        self.num_classes = num_classes
        self.blank_idx = blank_idx
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, T, hidden_dim]
        Output: log_probs [T, B, num_classes] for torch.nn.CTCLoss
        """
        logits = self.classifier(x) # [B, T, C]
        log_probs = logits.log_softmax(dim=-1) # [B, T, C]
        return log_probs.permute(1, 0, 2) # [T, B, C] for PyTorch CTCLoss

    def decode_greedy(self, log_probs: torch.Tensor) -> List[List[int]]:
        """
        Performs greedy CTC collapse (argmax, deduplicate consecutive tokens, remove blank).
        Input: log_probs [T, B, C] or [B, T, C]
        Output: List of decoded token index lists for each batch sample.
        """
        if log_probs.dim() == 3 and log_probs.shape[1] != log_probs.shape[0]:
            # Convert [T, B, C] -> [B, T, C] if necessary
            preds = log_probs.permute(1, 0, 2).argmax(dim=-1) # [B, T]
        else:
            preds = log_probs.argmax(dim=-1)

        batch_results = []
        for seq in preds:
            decoded = []
            prev_token = None
            for token_idx in seq.tolist():
                if token_idx != self.blank_idx and token_idx != prev_token:
                    decoded.append(token_idx)
                prev_token = token_idx
            batch_results.append(decoded)

        return batch_results
