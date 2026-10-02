"""Evaluation layer: imbalance-aware metrics and the significance gate."""

from balanceforge.evaluation.metrics import (
    METRIC_DIRECTION,
    PRIMARY_METRIC,
    aggregate_across_seeds,
    collect_failures,
    compute_metrics,
    gmean_score,
    significant_win,
)

__all__ = [
    "METRIC_DIRECTION",
    "PRIMARY_METRIC",
    "aggregate_across_seeds",
    "collect_failures",
    "compute_metrics",
    "gmean_score",
    "significant_win",
]

__author__ = "晨星"
