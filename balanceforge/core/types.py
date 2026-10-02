"""Typed data contracts shared across every BalanceForge stage.

Semantic contract (single source of truth, relied on by every module):

* ``Dataset.X`` is ``float64`` ``(n_samples, n_features)`` — never object dtype.
* ``Dataset.y`` is ``int64`` ``(n_samples,)`` with labels in ``[0, n_classes)``.
* ``SplitData`` indexes are always disjoint and cover the parent dataset.
* Every metric named ``*_lower_is_better`` is minimised; all others maximised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "gmean_error",
        "imbalance_ratio_log",
        "fit_time_sec",
    }
)


@dataclass(frozen=True)
class Dataset:
    """An immutable labelled dataset."""

    X: np.ndarray
    y: np.ndarray
    name: str = "unnamed"

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.int64).ravel()
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.y).size)

    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.y)

    def counts(self) -> np.ndarray:
        """Per-class sample counts ordered by ``classes``."""
        cls = self.classes
        return np.array([int((self.y == c).sum()) for c in cls], dtype=np.int64)

    def imbalance_ratio(self) -> float:
        """max(count) / min(count). ``inf`` when some class is absent."""
        c = self.counts()
        c = c[c > 0]
        if c.size == 0:
            return float("inf")
        return float(c.max() / c.min())

    def subset(self, idx: np.ndarray, name: str | None = None) -> Dataset:
        idx = np.asarray(idx, dtype=np.int64)
        return Dataset(
            X=self.X[idx],
            y=self.y[idx],
            name=name if name is not None else self.name,
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        c = self.counts()
        return (
            f"Dataset(name={self.name!r}, n={self.n_samples}, d={self.n_features}, "
            f"classes={self.n_classes}, counts={c.tolist()}, IR={self.imbalance_ratio():.1f})"
        )


@dataclass(frozen=True)
class SplitData:
    """Disjoint train/holdout indices over a parent :class:`Dataset`."""

    train_idx: np.ndarray
    holdout_idx: np.ndarray

    def __post_init__(self) -> None:
        tr = np.asarray(self.train_idx, dtype=np.int64).ravel()
        ho = np.asarray(self.holdout_idx, dtype=np.int64).ravel()
        if np.intersect1d(tr, ho).size:
            raise ValueError("train/holdout index overlap detected")
        if tr.size == 0 or ho.size == 0:
            raise ValueError("train/holdout must both be non-empty")
        object.__setattr__(self, "train_idx", tr)
        object.__setattr__(self, "holdout_idx", ho)

    def apply(self, ds: Dataset) -> tuple[Dataset, Dataset]:
        return ds.subset(self.train_idx, f"{ds.name}_train"), ds.subset(
            self.holdout_idx, f"{ds.name}_holdout"
        )


@dataclass(frozen=True)
class ImbalanceProfile:
    """Geometric statistics that characterise *how* a dataset is imbalanced.

    These are the routing signals consumed by :mod:`balanceforge.policy`.
    Every field is computable from train data only (no holdout leakage).
    """

    imbalance_ratio: float
    minority_fraction: float
    overlap_ratio: float
    """Fraction of minority points whose 1-NN lies in the majority class (0=separable, 1=fully overlapped)."""
    minority_cluster_count: int
    """Estimated number of minority modes (gaps-based cluster count)."""
    minority_noise_fraction: float
    """Fraction of minority points whose own 1-NN is also minority (label-noise proxy)."""
    knn_entropy: float
    """Normalised mean 1-NN neighbourhood class entropy (0=pure, 1=ambiguous)."""
    geometric: str
    """One of ``separable`` / ``overlap`` / ``multimodal`` / ``sparse``."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "imbalance_ratio": self.imbalance_ratio,
            "minority_fraction": self.minority_fraction,
            "overlap_ratio": self.overlap_ratio,
            "minority_cluster_count": self.minority_cluster_count,
            "minority_noise_fraction": self.minority_noise_fraction,
            "knn_entropy": self.knn_entropy,
            "geometric": self.geometric,
        }


@dataclass
class MethodResult:
    """Outcome of running one balancing method on one dataset."""

    dataset: str
    method: str
    seed: int
    metrics: dict[str, float] = field(default_factory=dict)
    n_train_after: int = 0
    route: str = ""
    elapsed_sec: float = 0.0
    skipped: bool = False
    skip_reason: str = ""

    def key(self) -> str:
        return f"{self.dataset}|{self.method}|{self.seed}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "method": self.method,
            "seed": self.seed,
            "metrics": dict(self.metrics),
            "n_train_after": self.n_train_after,
            "route": self.route,
            "elapsed_sec": round(float(self.elapsed_sec), 4),
            "skipped": bool(self.skipped),
            "skip_reason": self.skip_reason,
        }


@runtime_checkable
class Balancer(Protocol):
    """Any object that can rebalance a training split.

    Implementations return a dataset with the same feature space and the same
    label set; only the class distribution changes.
    """

    name: str

    def fit_resample(self, data: Dataset, seed: int) -> Dataset:  # pragma: no cover
        ...


@runtime_checkable
class Classifier(Protocol):
    """Minimal probabilistic classifier contract used by the evaluation layer."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Classifier:  # pragma: no cover
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        ...
