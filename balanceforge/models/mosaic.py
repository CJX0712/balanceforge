"""MOSAIC — Minority-Over-Synthesis with Adaptive Contamination gating.

The flagship rebalancer. Pipeline over the *fit split only*:

1. **Condense the majority** — interior majority mass carries no decision
   information, so keep the boundary shell (``condense_majority``).
2. **Segment the minority** — each local mode is expanded on its own so a
   far-flung mode is never interpolated against an unrelated one
   (``per_mode_expand``).
3. **Gate every segment by contamination** — a SMOTE segment whose interior
   comes within ``tau`` (the majority's own median 1-NN radius) of majority
   mass is synthesising label noise, so it is rejected
   (``contamination_gated_smote``, INV-16/17).
4. **Emit** a rebalanced :class:`~balanceforge.core.types.Dataset` plus an
   audit dict that records what was condensed, synthesised and rejected.

Every random draw comes from a single seeded ``numpy`` generator, so the
whole pipeline is deterministic for a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from balanceforge.core.types import Dataset, ImbalanceProfile


@dataclass(frozen=True)
class MosaicConfig:
    """Hyper-parameters of the flagship. All fields are keyword-constructible."""

    sigma_c: float = 0.5
    """Contamination tolerance scale (fraction of a segment's interior)."""
    k_neighbors: int = 5
    """Minority neighbours considered per segment."""
    target_ratio: float = 1.0
    """Desired minority/majority count ratio after rebalancing."""
    condense_keep: float = 0.8
    """Fraction of majority mass kept by the condensation stage."""
    contamination_ceiling: float = 0.5
    """Segments with a contamination rate above this are rejected."""
    regime_synth_budget: dict[str, float] = field(
        default_factory=lambda: {
            "separable": 0.25,
            "overlap": 0.60,
            "multimodal": 0.80,
            "sparse": 1.00,
        }
    )
    """Per-regime cap on synthesised points as a fraction of the minority."""
    offset_grid: tuple[float, ...] = tuple(
        round(-1.5 + 0.25 * i, 4) for i in range(13)
    )
    """Per-class logit offsets searched by the calibrator (always contains 0)."""


@dataclass(frozen=True)
class RebalanceResult:
    """Rebalanced dataset plus the audit trail of every stage decision."""

    data: Dataset
    audit: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.audit)


# --------------------------------------------------------------------- utils


def _pairwise(X: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix (n, n)."""
    sq = (X * X).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def _minority_label(y: np.ndarray) -> int:
    """Label of the (first) least frequent class."""
    cls, cnt = np.unique(y, return_counts=True)
    return int(cls[int(np.argmin(cnt))])


def _kmeans(X: np.ndarray, k: int, seed: int, iters: int = 12) -> np.ndarray:
    """Minimal Lloyd k-means (deterministic for a fixed seed)."""
    rng = np.random.default_rng(seed)
    k = min(k, X.shape[0])
    centers = X[rng.choice(X.shape[0], size=k, replace=False)].copy()
    labels = np.full(X.shape[0], -1, dtype=np.int64)
    for _ in range(iters):
        d = _pairwise_to(X, centers)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            members = X[labels == c]
            if members.shape[0]:
                centers[c] = members.mean(axis=0)
    return labels


def _pairwise_to(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Distance matrix between X (n, d) and centres C (k, d)."""
    sq_x = (X * X).sum(axis=1)[:, None]
    sq_c = (C * C).sum(axis=1)[None, :]
    d2 = sq_x + sq_c - 2.0 * (X @ C.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


# ------------------------------------------------------------------- stage 0


def majority_radius(maj: np.ndarray) -> float:
    """Median 1-NN distance among majority points — the gate's ``tau``.

    INV-16: strictly positive for any non-degenerate majority cloud.
    """
    X = np.asarray(maj, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return 0.0
    d = _pairwise(X)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def segment_contamination(
    seg_a: np.ndarray, seg_b: np.ndarray, maj: np.ndarray, tau: float
) -> float:
    """Fraction of a segment's interior whose nearest majority point is within ``tau``.

    0 = the segment lives in clean minority space; 1 = it is buried in the
    majority. The segment is the straight line between its two minority
    endpoints, sampled at 9 interior stations (endpoints included).
    """
    a = np.asarray(seg_a, dtype=np.float64).reshape(1, -1)
    b = np.asarray(seg_b, dtype=np.float64).reshape(1, -1)
    M = np.asarray(maj, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] == 0:
        return 0.0
    t = np.linspace(0.0, 1.0, 9)[:, None]
    pts = (1.0 - t) * a + t * b
    nn = _pairwise_to(pts, M).min(axis=1)
    return float((nn <= tau).mean())


# ------------------------------------------------------------------- stage 1


def _condense_indices(
    d: Dataset, keep: float
) -> tuple[np.ndarray, np.ndarray]:
    """Indices kept / dropped by majority condensation (deterministic).

    Majority points are ranked by their distance to the nearest minority
    point — the smallest distances sit on the decision boundary and are the
    only ones that carry information. Ties break by original row order, so
    the result needs no RNG at all.
    """
    X, y = d.X, d.y
    min_cls = _minority_label(y)
    min_mask = y == min_cls
    maj_idx = np.flatnonzero(~min_mask)
    if maj_idx.size == 0 or keep >= 1.0:
        return np.arange(X.shape[0], dtype=np.int64), np.empty(0, dtype=np.int64)
    Xm = X[maj_idx]
    Xmin = X[min_mask]
    if Xmin.shape[0]:
        d2min = _pairwise_to(Xm, Xmin).min(axis=1)
    else:
        d2min = np.full(Xm.shape[0], np.inf)
    n_keep = max(1, int(round(keep * Xm.shape[0])))
    order = np.argsort(d2min, kind="stable")
    kept_maj = np.sort(maj_idx[order[:n_keep]])
    dropped_maj = np.sort(np.setdiff1d(maj_idx, kept_maj))
    kept_idx = np.sort(np.concatenate([np.flatnonzero(min_mask), kept_maj]))
    return kept_idx, dropped_maj


def condense_majority(
    d: Dataset, keep: float, seed: int
) -> tuple[np.ndarray, int]:
    """Drop interior majority points, keep the boundary shell.

    Returns ``(X_kept, n_removed)`` where ``X_kept`` is the full feature
    matrix (all minority + kept majority), rows in original order. The
    ``seed`` is accepted for signature symmetry with the other stages; the
    ranking itself is deterministic.
    """
    kept_idx, dropped_idx = _condense_indices(d, keep)
    return d.X[kept_idx], int(dropped_idx.size)


# ------------------------------------------------------------------- stage 2


def per_mode_expand(
    d: Dataset, n_modes: int, n_synth: int, cfg: MosaicConfig, seed: int
) -> np.ndarray:
    """SMOTE inside each minority mode separately.

    A single-mode minority is returned untouched (empty synthesis) — there is
    nothing to gain from interpolating a unimodal blob with itself.
    """
    if n_modes <= 1 or n_synth <= 0:
        return np.empty((0, d.n_features), dtype=np.float64)
    X, y = d.X, d.y
    min_cls = _minority_label(y)
    Xm = X[y == min_cls]
    if Xm.shape[0] < 2:
        return np.empty((0, d.n_features), dtype=np.float64)
    k_eff = min(n_modes, Xm.shape[0] // 2)
    if k_eff <= 1:
        return np.empty((0, d.n_features), dtype=np.float64)
    labels = _kmeans(Xm, k_eff, seed)
    rng = np.random.default_rng(seed)
    k = max(1, min(cfg.k_neighbors, Xm.shape[0] - 1))
    out: list[np.ndarray] = []
    for c in range(k_eff):
        Xc = Xm[labels == c]
        if Xc.shape[0] < 2:
            continue
        dm = _pairwise(Xc)
        np.fill_diagonal(dm, np.inf)
        nn = np.argsort(dm, axis=1)[:, : min(k, Xc.shape[0] - 1)]
        for _ in range(n_synth):
            i = int(rng.integers(Xc.shape[0]))
            j = int(nn[i, rng.integers(nn.shape[1])])
            g = float(rng.random())
            out.append((1.0 - g) * Xc[i] + g * Xc[j])
    if not out:
        return np.empty((0, d.n_features), dtype=np.float64)
    return np.vstack(out)


# ------------------------------------------------------------------- stage 3


def contamination_gated_smote(
    d: Dataset, n_synth: int, cfg: MosaicConfig, seed: int
) -> tuple[np.ndarray, float, float, int]:
    """Gated SMOTE over minority 1-NN segments.

    Returns ``(synth_X, contamination, mean_contamination, rejected)``:

    * ``synth_X`` — the synthesised minority points (empty when nothing passed).
    * ``contamination`` — fraction of ``synth_X`` sitting within ``tau`` of the
      majority (should be ~0 for a healthy gate).
    * ``mean_contamination`` — mean gate score over *all* candidate segments.
    * ``rejected`` — number of segments refused for exceeding
      ``cfg.contamination_ceiling``.
    """
    rng = np.random.default_rng(seed)
    X, y = d.X, d.y
    min_cls = _minority_label(y)
    min_mask = y == min_cls
    Xm = X[min_mask]
    maj = X[~min_mask]
    empty = np.empty((0, X.shape[1]), dtype=np.float64)
    if Xm.shape[0] < 2 or n_synth <= 0:
        return empty, 0.0, 0.0, 0

    tau = majority_radius(maj) if maj.shape[0] >= 2 else 0.0
    k = max(1, min(cfg.k_neighbors, Xm.shape[0] - 1))
    dm = _pairwise(Xm)
    np.fill_diagonal(dm, np.inf)
    nn = np.argsort(dm, axis=1)[:, :k]

    accepted: list[tuple[np.ndarray, np.ndarray]] = []
    total = 0.0
    rejected = 0
    for i in rng.permutation(Xm.shape[0]):
        j = int(nn[i, rng.integers(k)])
        c = (
            segment_contamination(Xm[i], Xm[j], maj, tau)
            if maj.shape[0] and tau > 0.0
            else 0.0
        )
        total += c
        if c > cfg.contamination_ceiling:
            rejected += 1
        else:
            accepted.append((Xm[i], Xm[j]))
    segments = Xm.shape[0]
    mean_contam = total / segments if segments else 0.0

    if not accepted:
        return empty, 0.0, mean_contam, rejected

    pick = rng.integers(0, len(accepted), size=n_synth)
    gaps = rng.random(n_synth)
    pts = []
    for pi, g in zip(pick, gaps, strict=True):
        a, b = accepted[int(pi)]
        pts.append((1.0 - g) * a + g * b)
    synth = np.vstack(pts)

    if maj.shape[0] and tau > 0.0:
        nn_maj = _pairwise_to(synth, maj).min(axis=1)
        contam = float((nn_maj <= tau).mean())
    else:
        contam = 0.0
    return synth, contam, mean_contam, rejected


# ----------------------------------------------------------------- pipeline


def rebalance(
    fit: Dataset, profile: ImbalanceProfile, cfg: MosaicConfig, seed: int
) -> RebalanceResult:
    """Run the full MOSAIC pipeline on a fit split.

    Deterministic for a fixed ``seed`` (every draw comes from one
    ``numpy.random.default_rng``). The result's class counts move toward
    ``cfg.target_ratio`` without ever dropping a class.
    """
    y = fit.y
    audit: dict[str, Any] = {
        "seed": int(seed),
        "n_before": int(fit.n_samples),
        "counts_before": fit.counts().tolist(),
    }

    # --- stage 1: condense the majority -----------------------------------
    kept_idx, dropped_idx = _condense_indices(fit, cfg.condense_keep)
    audit["majority_removed"] = int(dropped_idx.size)
    d = Dataset(X=fit.X[kept_idx], y=fit.y[kept_idx], name=fit.name)
    audit["counts_after_condense"] = d.counts().tolist()
    cls = d.classes
    cnt = d.counts()
    min_cls = int(cls[int(np.argmin(cnt))])
    n_maj = int(cnt.max())
    n_min = int(cnt.min())

    target_min = int(round(cfg.target_ratio * n_maj))
    need = max(0, target_min - n_min)
    audit["synth_needed"] = need

    synth_parts: list[np.ndarray] = []

    # --- stage 2: per-mode expansion --------------------------------------
    n_modes = max(1, int(profile.minority_cluster_count))
    audit["minority_modes"] = n_modes
    if n_modes > 1 and need > 0:
        per_mode = max(1, need // n_modes)
        Xs = per_mode_expand(d, n_modes=n_modes, n_synth=per_mode, cfg=cfg, seed=seed)
        if Xs.shape[0]:
            synth_parts.append(Xs)

    # --- stage 3: contamination-gated SMOTE tops it up ---------------------
    remaining = need - sum(p.shape[0] for p in synth_parts)
    if remaining > 0:
        Xg, contam, mean_contam, rejected = contamination_gated_smote(
            d, remaining, cfg, seed
        )
        audit["gate_contamination"] = round(contam, 6)
        audit["gate_mean_contamination"] = round(mean_contam, 6)
        audit["gate_rejected_segments"] = rejected
        if Xg.shape[0]:
            synth_parts.append(Xg)

    if synth_parts:
        X_new = np.vstack([d.X, *synth_parts])
        y_new = np.concatenate(
            [d.y, *[np.full(p.shape[0], min_cls, dtype=np.int64) for p in synth_parts]]
        )
    else:
        X_new, y_new = d.X, d.y

    out = Dataset(X=X_new, y=y_new, name=fit.name)
    audit["n_after"] = int(out.n_samples)
    audit["counts_after"] = out.counts().tolist()
    audit["imbalance_ratio_before"] = fit.imbalance_ratio()
    audit["imbalance_ratio_after"] = out.imbalance_ratio()
    audit["classes_kept"] = bool(set(np.unique(y_new)) == set(np.unique(y)))
    return RebalanceResult(data=out, audit=audit)
