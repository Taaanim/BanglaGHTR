"""
Reward functions for Self-Critical Sequence Training (SCST).
Provides differentiable reward shaping for REINFORCE policy gradient.
"""

from typing import List, Dict
import editdistance
import unicodedata

from ..data.normalizer import normalize_bengali_text
from ..data.grapheme_parser import split_grapheme_clusters


def _cer_single(pred: str, ref: str) -> float:
    """Character Error Rate between two strings."""
    p = normalize_bengali_text(pred)
    r = normalize_bengali_text(ref)
    if len(r) == 0:
        return 0.0 if len(p) == 0 else 1.0
    return editdistance.eval(p, r) / len(r)


def _wer_single(pred: str, ref: str) -> float:
    """Word Error Rate between two strings."""
    p_words = normalize_bengali_text(pred).split()
    r_words = normalize_bengali_text(ref).split()
    if len(r_words) == 0:
        return 0.0 if len(p_words) == 0 else 1.0
    return editdistance.eval(p_words, r_words) / len(r_words)


def _bg_cer_single(pred: str, ref: str) -> float:
    """Bangla Grapheme-aware CER for a single pair."""
    p_clusters = split_grapheme_clusters(pred)
    r_clusters = split_grapheme_clusters(ref)
    if len(r_clusters) == 0:
        return 0.0 if len(p_clusters) == 0 else 1.0
    return editdistance.eval(p_clusters, r_clusters) / len(r_clusters)


def compute_rewards(
    predictions: List[str],
    references: List[str],
    cer_weight: float = 0.6,
    wer_weight: float = 0.3,
    bg_cer_weight: float = 0.1
) -> List[float]:
    """
    Computes per-sample rewards for SCST training.
    Reward = weighted combination of (1 - error_rate) metrics.
    Higher reward = better prediction.

    Args:
        predictions: List of predicted strings
        references: List of reference strings
        cer_weight: Weight for CER-based reward
        wer_weight: Weight for WER-based reward
        bg_cer_weight: Weight for BG-CER reward
    Returns:
        List of float rewards, one per sample. Range [0, 1].
    """
    rewards = []
    for pred, ref in zip(predictions, references):
        cer_r = 1.0 - min(_cer_single(pred, ref), 1.0)
        wer_r = 1.0 - min(_wer_single(pred, ref), 1.0)
        bg_cer_r = 1.0 - min(_bg_cer_single(pred, ref), 1.0)

        reward = cer_weight * cer_r + wer_weight * wer_r + bg_cer_weight * bg_cer_r
        rewards.append(reward)

    return rewards


def compute_reward_stats(rewards: List[float]) -> Dict[str, float]:
    """Computes statistics over a batch of rewards."""
    if not rewards:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

    import numpy as np
    r = np.array(rewards)
    return {
        "mean": float(r.mean()),
        "min": float(r.min()),
        "max": float(r.max()),
        "std": float(r.std())
    }
