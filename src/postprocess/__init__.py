"""
BanglaGHTR Post-Processing Package
Provides dynamic multi-decoder selection and ensemble scoring.
"""
from .selector import select_best_prediction, BanglaPostProcessor

__all__ = ["select_best_prediction", "BanglaPostProcessor"]
