"""
Evaluation metrics for Bengali Handwritten Text Recognition.
Implements CER (Character Error Rate), WER (Word Error Rate),
BG-CER (Bangla Grapheme-aware Character Error Rate),
Accuracy (exact match), and NED (Normalized Edit Distance).
"""

from typing import List, Tuple
import editdistance
import unicodedata

from ..data.normalizer import normalize_bengali_text
from ..data.grapheme_parser import split_grapheme_clusters

def compute_cer(predictions: List[str], references: List[str]) -> float:
    """
    Computes standard Character Error Rate (CER).
    CER = Sum(EditDistance(pred, ref)) / Sum(Len(ref))
    """
    total_dist = 0
    total_len = 0

    for pred, ref in zip(predictions, references):
        p_norm = normalize_bengali_text(pred)
        r_norm = normalize_bengali_text(ref)

        total_dist += editdistance.eval(p_norm, r_norm)
        total_len += max(1, len(r_norm))

    return total_dist / float(total_len) if total_len > 0 else 0.0

def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Computes Word Error Rate (WER).
    """
    total_dist = 0
    total_len = 0

    for pred, ref in zip(predictions, references):
        p_words = normalize_bengali_text(pred).split()
        r_words = normalize_bengali_text(ref).split()

        total_dist += editdistance.eval(p_words, r_words)
        total_len += max(1, len(r_words))

    return total_dist / float(total_len) if total_len > 0 else 0.0

def compute_bg_cer(predictions: List[str], references: List[str]) -> float:
    """
    Computes Bangla Grapheme-aware Character Error Rate (BG-CER).
    Evaluates edit distance at the level of constituent grapheme clusters (akshars).
    """
    total_dist = 0
    total_len = 0

    for pred, ref in zip(predictions, references):
        p_clusters = split_grapheme_clusters(pred)
        r_clusters = split_grapheme_clusters(ref)

        total_dist += editdistance.eval(p_clusters, r_clusters)
        total_len += max(1, len(r_clusters))

    return total_dist / float(total_len) if total_len > 0 else 0.0

def compute_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Computes exact match accuracy.
    A prediction is correct only if it matches the reference exactly
    (after Unicode normalization).
    """
    if not predictions:
        return 0.0
    correct = 0
    for pred, ref in zip(predictions, references):
        p_norm = normalize_bengali_text(pred)
        r_norm = normalize_bengali_text(ref)
        if p_norm == r_norm:
            correct += 1
    return correct / len(predictions)

def compute_ned(predictions: List[str], references: List[str]) -> float:
    """
    Computes Normalized Edit Distance (NED).
    NED = 1 - (1/N) * Sum(EditDistance(pred, ref) / max(len(pred), len(ref), 1))

    Higher NED = better (1.0 = perfect).
    This metric normalizes per-sample edit distance by the longer string,
    avoiding bias toward short samples.
    """
    if not predictions:
        return 0.0

    total_ned = 0.0
    for pred, ref in zip(predictions, references):
        p_norm = normalize_bengali_text(pred)
        r_norm = normalize_bengali_text(ref)

        dist = editdistance.eval(p_norm, r_norm)
        max_len = max(len(p_norm), len(r_norm), 1)
        total_ned += 1.0 - (dist / max_len)

    return total_ned / len(predictions)

def compute_all_metrics(predictions: List[str], references: List[str]) -> dict:
    """
    Computes all metrics at once and returns as a dictionary.
    Convenient for evaluation reporting.
    """
    return {
        "cer": compute_cer(predictions, references),
        "wer": compute_wer(predictions, references),
        "bg_cer": compute_bg_cer(predictions, references),
        "accuracy": compute_accuracy(predictions, references),
        "ned": compute_ned(predictions, references)
    }
