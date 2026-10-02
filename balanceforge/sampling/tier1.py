"""Tier-1 offline fallback: pure numpy / scikit-learn balancers.

These run when ``imbalanced-learn`` is unavailable (no network, no wheel, or a
hostile platform). They are *not* weaker science — the flagship route depends
entirely on them — only slower and slightly less expressive.

Implemented from scratch in numpy:
  * :func:`numpy_random_oversample` — ROS
  * :func:`numpy_smote` — SMOTE (linearly interpolated along the k-NN segment)
  * :func:`numpy_random_undersample` — RUS
  * :func:`numpy_cluster_centroids` — k-means prototype compression
  * :func:`numpy_tomek_links` — Tomek link removal
  * :func:`numpy_edited_nn` — ENN removal

INV-13 (geometric equivalence, not bit-identity): every synthesised point is a
convex combination ``(1-lambda)*x_i + lambda*x_j`` of two *genuine k-NN* minority
samples. The test suite verifies this exactly: for each synthetic row it
searches all ``(i, j)`` neighbour pairs and checks that some pair reproduces the
row to ``1e-9``. Bit-identity is impossible because ``imblearn`` and this module
consume the RNG stream in a different order — the geometry is the contract, the
draw order is not.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from balanceforge.core.errors import E200
from balanceforge.core.types import Dataset

__all__ = [
    "numpy_random_oversample",
    "numpy_smote",
    "numpy_random_undersample",
    "numpy_cluster_centroids",
    "numpy_tomek_links",
    "numpy_edited_nn",
    "TIER1_SPECS",
    "build_tier1",
    "tier1_names",
    "available_tier1",
]

_EPS = 1e-12


def _minority_index(y: np.ndarray) -> np.ndarray:
    """Indices of the rarest class; ties broken by the smallest label.

    ``np.argmin`` alone returns index 0 on a balanced dataset, so the
    "minority" would silently be whichever class happens to be encoded 0.
    """
    cls, counts = np.unique(y, return_counts=True)
    minor_label = int(cls[int(np.lexsort((cls, counts))[0])])
    return np.flatnonzero(y == minor_label)


def _stacked(data: Dataset, extra_X: np.ndarray, extra_y: np.ndarray) -> Dataset:
    if extra_X.shape[0] == 0:
        return Dataset(X=data.X.copy(), y=data.y.copy(), name=data.name)
    return Dataset(
        X=np.vstack([data.X, extra_X]),
        y=np.concatenate([data.y, extra_y]),
        name=data.name,
    )


def numpy_random_oversample(data: Dataset, target_ratio: float = 1.0, seed: int = 0) -> Dataset:
    """Random over-sampling of the minority class up to ``target_ratio``."""
    rng = np.random.default_rng(seed)
    min_idx = _minority_index(data.y)
    n_min = min_idx.size
    if n_min == 0:
        raise E200("no minority samples to over-sample")
    majority_n = data.n_samples - n_min
    target = int(np.floor(majority_n * target_ratio))
    if target <= n_min:
        return Dataset(X=data.X.copy(), y=data.y.copy(), name=data.name)
    extra_n = target - n_min
    pick = rng.choice(min_idx, size=extra_n, replace=True)
    return _stacked(data, data.X[pick], data.y[pick])


def numpy_smote(
    data: Dataset, k_neighbors: int = 5, target_ratio: float = 1.0, seed: int = 0
) -> Dataset:
    """SMOTE: synthesise minority points along their k-NN segments.

    For each minority point ``x_i`` with neighbours ``x_j`` in
    ``[1, k]``, draw a ``lambda ~ U(0, 1)`` and emit ``x_i + lambda * (x_j - x_i)``.

    INV-14: the generated points lie inside the convex hull of the minority
    samples, so their per-dimension min/max never exceed the minority hull's.
    INV-15: with ``target_ratio`` already satisfied the dataset is returned
    unchanged (idempotent).
    """
    rng = np.random.default_rng(seed)
    min_idx = _minority_index(data.y)
    n_min = min_idx.size
    if n_min < 2:
        raise E200("SMOTE needs >= 2 minority samples")
    majority_n = data.n_samples - n_min
    target = int(np.floor(majority_n * target_ratio))
    extra_n = target - n_min
    if extra_n <= 0:
        return Dataset(X=data.X.copy(), y=data.y.copy(), name=data.name)

    minority = data.X[min_idx]
    k = int(min(k_neighbors, n_min - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(minority)
    _, idx = nn.kneighbors(minority)  # column 0 is self

    base = rng.choice(n_min, size=extra_n, replace=True)
    nbr_sel = rng.integers(1, k + 1, size=extra_n)
    lam = rng.random(extra_n)[:, None]

    x_i = minority[base]
    x_j = minority[idx[base, nbr_sel]]
    synth = x_i + lam * (x_j - x_i)
    return _stacked(data, synth, data.y[min_idx[0]] * np.ones(extra_n, dtype=np.int64))


def numpy_random_undersample(
    data: Dataset, target_ratio: float = 1.0, seed: int = 0
) -> Dataset:
    """Random under-sampling of the majority class down to ``target_ratio``."""
    rng = np.random.default_rng(seed)
    min_idx = _minority_index(data.y)
    n_min = min_idx.size
    maj_idx = np.setdiff1d(np.arange(data.n_samples), min_idx)
    target = int(np.ceil(n_min * target_ratio))
    if target >= maj_idx.size:
        return Dataset(X=data.X.copy(), y=data.y.copy(), name=data.name)
    keep_maj = rng.choice(maj_idx, size=target, replace=False)
    keep = np.concatenate([min_idx, keep_maj])
    keep.sort()
    return Dataset(X=data.X[keep], y=data.y[keep], name=data.name)


def numpy_cluster_centroids(data: Dataset, target_ratio: float = 1.0, seed: int = 0) -> Dataset:
    """k-means prototype compression of the majority class."""
    min_idx = _minority_index(data.y)
    n_min = min_idx.size
    maj_idx = np.setdiff1d(np.arange(data.n_samples), min_idx)
    n_clusters = int(max(1, min(maj_idx.size, np.ceil(n_min * target_ratio))))
    if n_clusters >= maj_idx.size:
        return Dataset(X=data.X.copy(), y=data.y.copy(), name=data.name)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=3)
    km.fit(data.X[maj_idx])
    proto = km.cluster_centers_
    proto_y = data.y[maj_idx][0] * np.ones(proto.shape[0], dtype=np.int64)
    return _stacked(data, proto, proto_y)


def numpy_tomek_links(data: Dataset) -> Dataset:
    """Remove Tomek links: cross-class nearest-neighbour pairs.

    A Tomek link is ``(i, j)`` with ``y_i != y_j`` and no third point closer to
    both than they are to each other. Removing the majority member of every link
    is the standard remedy (a link is removed if either member is majority, but
    the minority is preserved to avoid destroying signal).
    """
    y = data.y
    n = data.n_samples
    nn = NearestNeighbors(n_neighbors=2).fit(data.X)
    _, idx = nn.kneighbors(data.X)
    drop: set[int] = set()
    for i in range(n):
        j = int(idx[i, 1])
        if y[i] == y[j]:
            continue
        if _is_majority(y, i):
            drop.add(int(i))
        if _is_majority(y, j):
            drop.add(int(j))
        if i in drop and j in drop:
            break
    if not drop:
        return Dataset(X=data.X.copy(), y=data.y.copy(), name=data.name)
    keep = np.setdiff1d(np.arange(n), np.fromiter(drop, dtype=np.int64, count=len(drop)))
    return Dataset(X=data.X[keep], y=data.y[keep], name=data.name)


def _is_majority(y: np.ndarray, i: int) -> bool:
    return y[i] == _majority_label(y)


def _majority_label(y: np.ndarray) -> int:
    """The most common class; ties broken by the largest label (complement of
    :func:`_minority_index`, so the two partitions stay disjoint)."""
    cls, counts = np.unique(y, return_counts=True)
    return int(cls[int(np.lexsort((-cls, -counts))[0])])


def numpy_edited_nn(data: Dataset, k: int = 1) -> Dataset:
    """ENN: drop any point whose k-NN majority label differs from its own."""
    y = data.y
    n = data.n_samples
    kk = int(min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(data.X)
    _, idx = nn.kneighbors(data.X)
    drop: set[int] = set()
    for i in range(n):
        neigh = y[idx[i, 1:]]
        if neigh.size == 0:
            continue
        vals, counts = np.unique(neigh, return_counts=True)
        if int(vals[int(np.argmax(counts))]) != int(y[i]):
            drop.add(int(i))
    if not drop:
        return Dataset(X=data.X.copy(), y=data.y.copy(), name=data.name)
    keep = np.setdiff1d(np.arange(n), np.fromiter(drop, dtype=np.int64, count=len(drop)))
    return Dataset(X=data.X[keep], y=data.y[keep], name=data.name)


def available_tier1() -> bool:
    """Tier-1 only needs numpy + sklearn, which are the offline floor."""
    try:
        import sklearn  # noqa: F401
    except Exception:
        return False
    return True


TIER1_SPECS: dict[str, dict[str, object]] = {
    "ros": {"fn": numpy_random_oversample, "desc": "numpy random over-sampling"},
    "smote": {"fn": numpy_smote, "desc": "numpy SMOTE"},
    "smote_k10": {"fn": numpy_smote, "k": 10, "desc": "numpy SMOTE (k=10)"},
    "rus": {"fn": numpy_random_undersample, "desc": "numpy random under-sampling"},
    "cc": {"fn": numpy_cluster_centroids, "desc": "numpy k-means centroid compression"},
    "tomek": {"fn": numpy_tomek_links, "desc": "numpy Tomek link removal"},
    "enn": {"fn": numpy_edited_nn, "desc": "numpy EditedNearestNeighbours"},
    "none": {"fn": None, "desc": "no balancing (naive reference)"},
}


def tier1_names() -> tuple[str, ...]:
    return tuple(TIER1_SPECS)


def build_tier1(name: str, data: Dataset, seed: int) -> Dataset:
    """Apply a Tier-1 strategy; unknown names raise ``E200``."""
    if name not in TIER1_SPECS:
        raise E200(f"unknown tier1 strategy {name!r}; valid: {sorted(TIER1_SPECS)}")
    spec = TIER1_SPECS[name]
    fn = spec["fn"]
    if fn is None:
        return Dataset(X=data.X.copy(), y=data.y.copy(), name=data.name)
    if name == "enn":
        return numpy_edited_nn(data, k=int(spec.get("k", 1)))  # type: ignore[arg-type]
    if name in {"tomek"}:
        return fn(data)
    return fn(data, seed=seed)  # type: ignore[operator]
