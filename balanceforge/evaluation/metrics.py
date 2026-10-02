"""Evaluation metrics for imbalanced classification.

Contract: **every metric here is maximised** (``gmean_error`` is the single
exception and is minimised; it is used as an auxiliary diagnostic only).

Primary metric: ``gmean`` = geometric mean of per-class recall, the standard
imbalance-robust aggregate (Kubat & Holte). Secondary: macro-F1, balanced
accuracy, minority recall/precision, Matthews correlation.

INV-10: ``gmean == 0`` whenever any class recall is ``0``; ``gmean == 1`` for a
perfect classifier; ``gmean <= balanced_accuracy`` always (the geometric mean
never exceeds the arithmetic mean of the same recalls).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from balanceforge.core.errors import E400
from balanceforge.core.types import Dataset

__all__ = [
    "PRIMARY_METRIC",
    "METRIC_DIRECTION",
    "compute_metrics",
    "gmean_score",
    "aggregate_across_seeds",
    "significant_win",
    "collect_failures",
]

PRIMARY_METRIC = "gmean"

METRIC_DIRECTION: dict[str, int] = {
    "gmean": 1,
    "macro_f1": 1,
    "balanced_accuracy": 1,
    "mcc": 1,
    "minority_recall": 1,
    "minority_precision": 1,
    "au_prc": 1,
    "n_train_after": 1,
    "gmean_error": -1,
}


def _minority_class(y: np.ndarray) -> int:
    """The rarest class, with ties broken by the smallest label.

    ``np.argmin`` alone would return index 0 for a balanced dataset regardless
    of which class is which, so the "minority" column of a balanced run would
    be whichever class happened to be encoded 0. Sorting by
    ``(count, label)`` makes the choice deterministic and meaningful.
    """
    cls, counts = np.unique(y, return_counts=True)
    return int(cls[int(np.lexsort((cls, counts))[0])])


def gmean_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Geometric mean of per-class recall — the primary imbalance metric."""
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    recalls = np.clip(np.asarray(recalls, dtype=np.float64), 0.0, 1.0)
    if recalls.size == 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(recalls > 0.0, np.log(recalls), -np.inf)
    if np.any(recalls <= 0.0):
        return 0.0
    return float(np.exp(np.mean(logs)))


def _au_prc(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    """Binary AUPRC using the minority probability column (nan when degenerate)."""
    if classes.size != 2:
        return float("nan")
    minor = _minority_class(y_true)
    if proba.ndim == 1:
        score = proba
    else:
        col = int(np.flatnonzero(classes == minor)[0])
        if col >= proba.shape[1]:
            return float("nan")
        score = proba[:, col]
    y_bin = (y_true == minor).astype(np.int64)
    if y_bin.min() == y_bin.max():
        return float("nan")
    return float(average_precision_score(y_bin, score))


def compute_metrics(
    holdout: Dataset, y_pred: np.ndarray, proba: np.ndarray
) -> dict[str, float]:
    """Compute the full metric block for one (dataset, method, seed) cell.

    ``proba`` columns must follow ``np.unique(y_train)`` ordering.
    """
    y_true = holdout.y
    classes = np.unique(y_true)
    if np.unique(y_pred).size and not set(np.unique(y_pred)).issubset(set(classes.tolist())):
        raise E400("y_pred contains labels absent from the holdout")

    minor = _minority_class(y_true)
    is_bin = classes.size == 2
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    out: dict[str, float] = {
        "gmean": gmean_score(y_true, y_pred),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "minority_recall": float(
            recall_score(y_true == minor, y_pred == minor, zero_division=0)
        ),
        "minority_precision": float(
            precision_score(y_true == minor, y_pred == minor, zero_division=0)
        ),
        "au_prc": _au_prc(y_true, proba, classes) if is_bin else float("nan"),
    }
    # Per-class recall detail (used for failure-case analysis).
    recalls = np.asarray(
        recall_score(y_true, y_pred, average=None, zero_division=0), dtype=np.float64
    )
    for c, r in zip(classes.tolist(), recalls.tolist(), strict=False):
        out[f"recall_cls{int(c)}"] = float(r)
    out["_cm"] = cm.tolist()  # type: ignore[assignment]
    return out


def aggregate_across_seeds(rows: list[dict[str, float]], metric: str) -> dict[str, float]:
    """mean / std / n over seeds for one metric (nan-safe)."""
    vals = np.array(
        [float(r[metric]) for r in rows if np.isfinite(float(r.get(metric, np.nan)))],
        dtype=np.float64,
    )
    if vals.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=0)),
        "n": int(vals.size),
        "sem": float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0,
    }


def significant_win(
    method_mean: float,
    method_std: float,
    base_mean: float,
    base_std: float,
    threshold: float = 0.0,
) -> dict[str, float | bool]:
    """Two-sided significance gate used across every BalanceForge comparison.

    A win requires **all three**:
      1. ``mean_method - mean_base > threshold`` (the declared effect size);
      2. ``mean_method - mean_base > 0.5 * (std_method + std_base)``
         (the 1/2-sigma-sum significance gate from the delivery contract);
      3. both stds are finite (i.e. the comparison is not degenerate).
    """
    diff = float(method_mean) - float(base_mean)
    half_sigma = 0.5 * (float(method_std) + float(base_std))
    finite = bool(
        np.isfinite(method_mean)
        and np.isfinite(base_mean)
        and np.isfinite(method_std)
        and np.isfinite(base_std)
    )
    return {
        "diff": diff,
        "half_sigma_sum": half_sigma,
        "effect_ok": bool(diff > threshold),
        "significance_ok": bool(diff > half_sigma),
        "significant": bool(finite and diff > threshold and diff > half_sigma),
    }


def collect_failures(
    holdout: Dataset,
    y_pred: np.ndarray,
    n: int = 5,
) -> list[dict[str, float | int]]:
    """Return the ``n`` worst false negatives + false positives with features.

    Used for the mandated failure-case analysis; values are *derived from the
    run*, never hand-written.
    """
    y_true = holdout.y
    minor = _minority_class(y_true)
    fn = np.flatnonzero((y_true == minor) & (y_pred != minor))
    fp = np.flatnonzero((y_true != minor) & (y_pred == minor))
    out: list[dict[str, float | int]] = []
    for kind, idxs in (("false_negative", fn), ("false_positive", fp)):
        for i in idxs[:n]:
            row = holdout.X[i]
            out.append(
                {
                    "kind": kind,
                    "index": int(i),
                    "x0": float(row[0]),
                    "x1": float(row[1]),
                    "true": int(y_true[i]),
                    "pred": int(y_pred[i]),
                }
            )
    return out
