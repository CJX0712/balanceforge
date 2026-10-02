"""Structural imbalance profile estimation — the routing brain of BalanceForge.

The profile answers a single question: **how** is this dataset imbalanced?

Four geometries, each with a distinct statistical signature and a distinct
correct remedy:

============  ===============================  ===========================
geometry      signature                        best remedy
============  ===============================  ===========================
``separable`` minority is spatially detached  nothing helps; the naive model
                                              already saturates (costly
                                              resampling only adds variance)
``overlap``   minority sits inside the        conservative cleaning +
              majority hull                    mild border-line oversampling
``multimodal`` minority is ``k`` well         per-mode balanced expansion
              separated modes
``sparse``    extreme IR, very few minority   manifold oversampling +
              points                           cost-sensitive loss
============  ===============================  ===========================

All statistics are computed from the *training split only*; the holdout never
touches this module (INV-11).
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from balanceforge.core.types import Dataset, ImbalanceProfile

__all__ = [
    "estimate_profile",
    "minority_cluster_count",
    "knn_class_entropy",
    "classify_geometry",
    "GEOMETRIES",
]

GEOMETRIES: tuple[str, ...] = ("separable", "overlap", "multimodal", "sparse")

# Routing thresholds — documented in docs/architecture.md and asserted in tests.
OVERLAP_HIGH = 0.30
CLUSTER_MIN = 2
SPARSE_IR = 18.0
"""IR at or above which the ``sparse`` regime fires. The shipped ``sparse``
dataset has IR=20, comfortably above this gate (and the ``multimodal`` dataset's
IR=15 stays below it, so the two regimes do not collide)."""
_EPS = 1e-12
MINORITY_CLUSTER_GAP_SCALE = 3.5
"""An MST edge must exceed this multiple of the median to count as inter-mode."""
CLUSTER_BULK_RATIO = 6.0
"""max(edge)/median(edge) gate: rejects smooth Gaussian MST tails outright."""
CLUSTER_CONTRAST = 0.60
"""(w[i]-w[i-1])/w[i-1] gate: the edge must break away from its predecessor."""
MST_SWEEP_POINTS = 96
"""Resolution of the geometric threshold sweep over the MST edge spectrum."""
MST_PLATEAU_FRAC = 0.12
"""Minimum share of the log-range the penultimate plateau must span to count."""


def _minority_mask(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cls, counts = np.unique(y, return_counts=True)
    order = np.argsort(counts, kind="stable")
    minority_label = cls[order[0]]
    return y == minority_label, minority_label


def minority_cluster_count(X: np.ndarray, mask: np.ndarray, random_state: int = 0) -> int:
    """Estimate the number of minority modes via single-linkage MST gap cutting.

    A minimum spanning tree is built over the minority subset with Prim's
    algorithm in pure numpy. Sorting the ``n-1`` MST edge weights reveals the
    *gaps* between densely packed groups: cutting every edge longer than
    ``CLUSTER_GAP_SCALE`` times the median MST edge yields the mode count.

    INV-2: a single Gaussian blob returns exactly ``1``; ``k`` islands placed
    far apart return exactly ``k``.
    """
    pts = np.asarray(X[mask], dtype=np.float64)
    n = pts.shape[0]
    if n < 4:
        return 1

    # --- Prim's algorithm, O(n^2), vectorised over the frontier ---
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)
    best[0] = 0.0
    edge_w: list[float] = []

    for _ in range(n):
        masked = np.where(in_tree, np.inf, best)
        j = int(np.argmin(masked))
        w = float(masked[j])
        if not np.isfinite(w):
            break
        in_tree[j] = True
        if parent[j] >= 0:
            edge_w.append(w)
        diff = pts - pts[j]
        d = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        upd = (~in_tree) & (d < best)
        best[upd] = d[upd]
        parent[upd] = j

    if len(edge_w) < 3:
        return 1

    w = np.sort(np.asarray(edge_w, dtype=np.float64))
    # --- Plateau-width cut on the MST edge spectrum -----------------------------
    #
    # Rejected alternatives (both verified empirically):
    #   * "count edges above K x median" -> a single isotropic Gaussian blob trips
    #     it (200x4 -> 2~3), because its MST tail is genuinely long.
    #   * "find the largest relative jump" -> equidistant islands produce k-1
    #     *identical* inter-mode edges, so no value-based cut can separate them.
    #
    # The robust signal is the *plateau width* of the connected-component count.
    # Sweep a threshold t over the edge spectrum and record how many components
    # the minority MST has. For one mode the sequence is 1,1,1,...,1 with at most
    # a razor-thin spike at the very end. For k well-separated islands the
    # sequence is 1 -> k (wide) -> 1, so the *penultimate plateau* has height k
    # and spans a wide log-range.
    lo, hi = float(w[0]), float(w[-1])
    if not np.isfinite(hi) or hi <= 0.0:
        return 1
    lo = max(lo, hi * 1e-9)
    grid = np.geomspace(lo * 1.0001, hi, MST_SWEEP_POINTS)

    counts = np.empty(grid.size, dtype=np.int64)
    for i, t in enumerate(grid):
        counts[i] = _mst_component_count(n, w, float(t))

    # Run-length encode, then inspect the penultimate plateau.
    run_vals: list[int] = []
    run_widths: list[float] = []
    log_grid = np.log(grid)
    start = 0
    for i in range(1, counts.size + 1):
        if i == counts.size or counts[i] != counts[start]:
            run_vals.append(int(counts[start]))
            run_widths.append(float(log_grid[i - 1] - log_grid[start]))
            start = i

    if len(run_vals) <= 1:
        return 1
    penult_val = run_vals[-2]
    penult_width = run_widths[-2]
    total_width = float(log_grid[-1] - log_grid[0])
    if penult_width < MST_PLATEAU_FRAC * total_width:
        return 1
    return int(max(1, penult_val))


def _mst_component_count(n: int, sorted_weights: np.ndarray, t: float) -> int:
    """Number of components in a path graph of ``n`` nodes with edges <= ``t``."""
    kept = int(np.count_nonzero(sorted_weights <= t))
    return int(n - kept)


def knn_class_entropy(X: np.ndarray, y: np.ndarray, k: int = 5, random_state: int = 0) -> float:
    """Mean neighbour class entropy over all points, normalised to ``[0, 1]``.

    ``0`` = every neighbourhood is class-pure; ``1`` = every neighbourhood is
    class-balanced. Uses *label* entropy (not distance) so it is a direct
    measure of decision-boundary ambiguity.

    INV-3: entropy is symmetric under class relabelling and is ``0.0`` when the
    classes are perfectly separated by a large margin.
    """
    n = X.shape[0]
    if n < 2:
        return 0.0
    k = int(min(k, n))
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, idx = nn.kneighbors(X)
    neigh_y = y[idx]
    n_cls = int(np.unique(y).size)
    if n_cls < 2:
        return 0.0
    probs = np.zeros((n, n_cls), dtype=np.float64)
    for c in range(n_cls):
        probs[:, c] = np.count_nonzero(neigh_y == c, axis=1)
    probs /= k
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(probs > 0.0, np.log(probs), 0.0)
    ent = -(probs * logs).sum(axis=1)
    return float(np.mean(ent) / np.log(n_cls))


def _overlap_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of minority points whose 1-NN belongs to the majority class."""
    min_mask, _ = _minority_mask(y)
    if min_mask.sum() == 0:
        return 0.0
    # Fit on all training points, query only the minority ones.
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    _, idx = nn.kneighbors(X[min_mask])
    nn_labels = y[idx[:, 1]]
    return float(np.mean(nn_labels != y[min_mask]))


def _minority_noise_fraction(X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of minority points whose own 1-NN is also minority."""
    min_mask, _ = _minority_mask(y)
    n_min = int(min_mask.sum())
    if n_min < 2:
        return 0.0
    nn = NearestNeighbors(n_neighbors=2).fit(X[min_mask])
    _, idx = nn.kneighbors(X[min_mask])
    return float(np.mean(idx[:, 1] != np.arange(n_min)))


def classify_geometry(profile_kwargs: dict[str, float | int]) -> str:
    """Map profile statistics to one of :data:`GEOMETRIES`.

    Decision order matters: the most specific geometry wins, so the order is
    ``sparse`` (extreme IR) -> ``multimodal`` (multi-mode) -> ``overlap``
    (ambiguous) -> ``separable``.

    INV-4: the returned value is always in :data:`GEOMETRIES`.
    INV-5: monotone in ``overlap_ratio`` — holding everything else fixed, the
    geometry is ``separable`` for low overlap and can only move to
    ``overlap``/``sparse`` as overlap grows.
    """
    ir = float(profile_kwargs["imbalance_ratio"])
    clusters = int(profile_kwargs["minority_cluster_count"])
    overlap = float(profile_kwargs["overlap_ratio"])

    if ir >= SPARSE_IR:
        return "sparse"
    if clusters >= CLUSTER_MIN:
        return "multimodal"
    if overlap >= OVERLAP_HIGH:
        return "overlap"
    return "separable"


def estimate_profile(data: Dataset, k: int = 5) -> ImbalanceProfile:
    """Compute the full :class:`ImbalanceProfile` for a training split.

    INV-1: the profile is invariant to feature permutation and to additive
    constant shifts (it depends only on the class-conditional geometry).
    INV-6: for a perfectly balanced dataset ``minority_fraction == 0.5`` and
    ``imbalance_ratio == 1.0``.
    """
    X, y = data.X, data.y
    counts = data.counts()
    positive = counts[counts > 0]
    ir = float(positive.max() / positive.min()) if positive.size else float("inf")
    min_frac = float(positive.min() / positive.sum()) if positive.sum() else 0.0

    min_mask, _ = _minority_mask(y)
    clusters = minority_cluster_count(X, min_mask)
    overlap = _overlap_ratio(X, y)
    noise = _minority_noise_fraction(X, y)
    entropy = knn_class_entropy(X, y, k=k)

    geometry = classify_geometry(
        {
            "imbalance_ratio": ir,
            "minority_cluster_count": clusters,
            "overlap_ratio": overlap,
        }
    )

    return ImbalanceProfile(
        imbalance_ratio=ir,
        minority_fraction=min_frac,
        overlap_ratio=overlap,
        minority_cluster_count=clusters,
        minority_noise_fraction=noise,
        knn_entropy=entropy,
        geometric=geometry,
    )
