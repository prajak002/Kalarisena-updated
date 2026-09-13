"""Calibration/discrimination metrics for SCVC evaluation (paper Table 3b:
AUROC, AUPRC, Brier, ECE). Implemented in plain numpy since this repo's
dependency list (requirements.pipeline.txt) does not include scikit-learn and
these have simple closed forms.
"""

from __future__ import annotations

import numpy as np


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney U statistic form of AUROC."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    rank_sum_pos = ranks[: len(pos)].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y.sum(), 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(precision, recall))


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.mean((y_score - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_score >= lo) & (y_score < hi if i < n_bins - 1 else y_score <= hi)
        if not np.any(mask):
            continue
        conf = y_score[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
