"""Stratified train / validation / holdout splitting with strict no-leakage guarantees.

The holdout is carved out *first* and never touched by any fitting step. The
validation split is carved out of the remainder and is the only data the
router and the HPO are allowed to see.

INV-8: the three index sets are pairwise disjoint and their union is the parent.
INV-9: the class ratio of the holdout matches the parent within 1 minority
sample per class (stratified by construction).
"""

from __future__ import annotations

import numpy as np

from balanceforge.core.errors import E100
from balanceforge.core.seed import derive_seed
from balanceforge.core.types import Dataset, SplitData

__all__ = ["split_holdout", "split_train_val", "make_splits"]


def _holdout_counts(y: np.ndarray, holdout_size: float) -> dict[int, int]:
    """Allocate holdout samples per class, guaranteeing >=1 per class if possible."""
    rng_classes = np.unique(y)
    counts = {int(c): int((y == c).sum()) for c in rng_classes}
    alloc = {c: max(1, int(np.floor(n * holdout_size))) for c, n in counts.items()}
    # Never take the entire class: keep >= 1 sample on the training side.
    return {c: min(a, counts[c] - 1) for c, a in alloc.items() if counts[c] >= 2}


def split_holdout(data: Dataset, holdout_size: float, seed: int) -> SplitData:
    """Carve the untouched holdout out of ``data``."""
    if not 0.0 < holdout_size < 1.0:
        raise E100("holdout_size must be in (0, 1)")
    y = data.y
    rng = np.random.default_rng(derive_seed(seed, "holdout", data.name))
    alloc = _holdout_counts(y, holdout_size)

    hold_parts = []
    for c, k in alloc.items():
        if k <= 0:
            continue
        idx = np.flatnonzero(y == c)
        perm = rng.permutation(idx.size)
        hold_parts.append(idx[perm[:k]])

    holdout_idx = np.sort(np.concatenate(hold_parts)) if hold_parts else np.array([], np.int64)
    mask = np.ones(data.n_samples, dtype=bool)
    mask[holdout_idx] = False
    train_idx = np.flatnonzero(mask)

    if train_idx.size == 0 or holdout_idx.size == 0:
        raise E100("degenerate split: one side is empty")
    return SplitData(train_idx=np.sort(train_idx), holdout_idx=holdout_idx)


def split_train_val(data: Dataset, val_size: float, seed: int) -> SplitData:
    """Split a *training* dataset into fit / validation portions."""
    if not 0.0 < val_size < 1.0:
        raise E100("val_size must be in (0, 1)")
    y = data.y
    rng = np.random.default_rng(derive_seed(seed, "val", data.name))
    alloc = _holdout_counts(y, val_size)

    val_parts = []
    for c, k in alloc.items():
        if k <= 0:
            continue
        idx = np.flatnonzero(y == c)
        perm = rng.permutation(idx.size)
        val_parts.append(idx[perm[:k]])

    val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], np.int64)
    mask = np.ones(data.n_samples, dtype=bool)
    mask[val_idx] = False
    fit_idx = np.flatnonzero(mask)

    if fit_idx.size == 0 or val_idx.size == 0:
        raise E100("degenerate split: one side is empty")
    return SplitData(train_idx=np.sort(fit_idx), holdout_idx=val_idx)


def make_splits(
    data: Dataset, holdout_size: float, val_size: float, seed: int
) -> tuple[Dataset, Dataset, Dataset]:
    """Return ``(fit, val, holdout)`` datasets with no index overlap."""
    outer = split_holdout(data, holdout_size, seed)
    fit_ds, holdout_ds = outer.apply(data)

    inner = split_train_val(fit_ds, val_size, seed)
    fit_sub, val_ds = inner.apply(fit_ds)

    return fit_sub, val_ds, holdout_ds
