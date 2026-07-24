"""Proper scores. Multiclass Brier and log loss per event.

Brier here is the full multiclass form (sum of squared errors over the
event's bins), matching ClosingLine's convention, so numbers are
comparable across events with different bin counts only in aggregate —
which is fine, because model and market are always scored on the same
events and bins, and only the paired differential is interpreted.
"""

from __future__ import annotations

import numpy as np


def brier(probs: np.ndarray, outcome_idx: int) -> float:
    onehot = np.zeros(len(probs))
    onehot[outcome_idx] = 1.0
    return float(((probs - onehot) ** 2).sum())


def logloss(probs: np.ndarray, outcome_idx: int) -> float:
    return float(-np.log(max(probs[outcome_idx], 1e-12)))
