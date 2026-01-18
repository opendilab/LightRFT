"""
Evaluation utilities for LightRFT.

This module provides tools for evaluating model performance on various benchmarks.
"""

from .math_eval_utils import (
    normalize_answer,
    extract_answer,
    compare_answers,
    evaluate_predictions,
)

__all__ = [
    "normalize_answer",
    "extract_answer",
    "compare_answers",
    "evaluate_predictions",
]
