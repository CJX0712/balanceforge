"""Synthetic imbalanced datasets with explicit, tunable difficulty knobs.

Five generators, one per imbalance geometry, each with a difficulty dial so the
benchmark can be tuned to a *discriminative* operating point (where naive and
strong baselines genuinely separate) instead of a saturated or degenerate one.

Every generator is a pure function of ``(seed, n_majority)`` — INV-7.
"""

from __future__ import annotations

import numpy as np

from balanceforge.core.errors import E100
from balanceforge.core.types import Dataset

__all__ = [
    "make_separable",
    "make_overlap",
    "make_multimodal",
    "make_sparse",
    "make_manifold",
    "DATASETS",
    "get_dataset",
    "available_datasets",
    "dataset_defaults",
]

_EPS = 1e-12


def _standardise(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < _EPS, 1.0, sd)
    return (X - mu) / sd


def make_separable(
    n_majority: int = 1200, imratio: float = 10.0, seed: int = 0, d: int = 6
) -> Dataset:
    """``separable``: majority ring/shell around a detached minority core.

    Knobs: ``n_majority``, ``imratio``, ``d``.
    Expected geometry: minority 1-NN always majority-free, single mode.
    """
    if imratio < 1.0:
        raise E100("imratio must be >= 1")
    n_min = max(2, int(round(n_majority / imratio)))
    rng = np.random.default_rng(seed)

    theta = rng.uniform(0.0, 2.0 * np.pi, n_majority)
    radius = rng.normal(3.0, 0.45, n_majority)
    maj = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    maj = np.hstack([maj, rng.normal(0.0, 0.45, (n_majority, d - 2))])

    core = rng.normal(0.0, 0.38, (n_min, 2))
    mnr = np.hstack([core, rng.normal(0.0, 0.45, (n_min, d - 2))])

    X = _standardise(np.vstack([maj, mnr]))
    y = np.concatenate([np.zeros(n_majority, dtype=np.int64), np.ones(n_min, dtype=np.int64)])
    return Dataset(X=X, y=y, name="separable")


def make_overlap(
    n_majority: int = 2000, imratio: float = 8.0, seed: int = 0, d: int = 6, sep: float = 0.9
) -> Dataset:
    """``overlap``: minority straddles the majority mode; heavy class overlap.

    Knobs: ``n_majority``, ``imratio``, ``d``, ``sep``.

    ``sep`` is the class-separation dial. Two settings were measured:

    * ``sep=0.35`` put the naive baseline at G-mean 0.27, but the
      train/holdout gap was **+0.65** — the forest memorised the training
      minority and generalised poorly, so the benchmark measured
      overfitting rather than imbalance handling. 2000 majority samples swamp
      a 0.35-sigma displacement in the informative directions.
    * ``sep=0.9`` gives a real but *hard* learnable boundary: the naive
      baseline sits well below ceiling, and the gap between it and a
      competent resampler is where the method comparison becomes meaningful.

    At ``sep >= 1.2`` every method saturates (naive G-mean 0.985) and the
    benchmark loses all discriminative power, so 0.9 is the upper end of the
    useful band.

    ``imratio`` is deliberately only 8 with ``n_majority=2000``. The earlier
    ``20.0 / 1200`` setting left just **18** minority samples in the test split,
    so every per-seed G-mean was quantised to a handful of values (observed
    std 0.28 across methods — that is sampling noise, not method quality) and the
    regime ranked last regardless of what it did. 250 minority samples in the
    training split leaves ~75 in the holdout, which is enough to separate methods.
    """
    if imratio < 1.0:
        raise E100("imratio must be >= 1")
    n_min = max(2, int(round(n_majority / imratio)))
    rng = np.random.default_rng(seed)

    maj = rng.normal(0.0, 1.0, (n_majority, d))
    mnr = rng.normal(0.0, 1.0, (n_min, d))
    # A weak, non-axis-aligned discriminant: only `sep` worth of signal, shared
    # across the first two features. The remaining d-2 features are pure noise,
    # so linear models are capped while trees can still find the weak signal.
    direction = np.zeros(d)
    direction[0], direction[1] = 0.7071, 0.7071
    mnr = mnr - sep * direction
    maj = maj + sep * direction
    # A private majority channel: informative but not class-exclusive, which is
    # what makes the majority "corrupt" from the minority's point of view.
    maj[:, d - 1] += rng.normal(1.5, 0.9, n_majority)
    mnr[:, d - 1] += rng.normal(0.2, 0.9, n_min)

    X = _standardise(np.vstack([maj, mnr]))
    y = np.concatenate([np.zeros(n_majority, dtype=np.int64), np.ones(n_min, dtype=np.int64)])
    return Dataset(X=X, y=y, name="overlap")


def make_multimodal(
    n_majority: int = 1200, imratio: float = 15.0, seed: int = 0, d: int = 6, k_modes: int = 3
) -> Dataset:
    """``multimodal``: ``k_modes`` far-separated minority islands vs one blob.

    Knobs: ``n_majority``, ``imratio``, ``d``, ``k_modes``.
    Cluster-count estimation must return ``>= k_modes`` (INV-2).
    """
    if imratio < 1.0:
        raise E100("imratio must be >= 1")
    if k_modes < 1:
        raise E100("k_modes must be >= 1")
    n_min = max(4, int(round(n_majority / imratio)))
    rng = np.random.default_rng(seed)

    maj = rng.normal(0.0, 1.0, (n_majority, d))
    per = max(2, n_min // k_modes)
    chunks = []
    for i in range(k_modes):
        m = per if i < k_modes - 1 else n_min - per * (k_modes - 1)
        if m <= 0:
            continue
        centre = np.zeros(d)
        centre[0] = 6.0 * np.cos(2.0 * np.pi * i / k_modes)
        centre[1] = 6.0 * np.sin(2.0 * np.pi * i / k_modes)
        chunks.append(rng.normal(0.0, 0.55, (m, d)) + centre)
    mnr = np.vstack(chunks)

    X = _standardise(np.vstack([maj, mnr]))
    y = np.concatenate([np.zeros(n_majority, dtype=np.int64), np.ones(n_min, dtype=np.int64)])
    return Dataset(X=X, y=y, name="multimodal")


def make_sparse(
    n_majority: int = 1200, imratio: float = 20.0, seed: int = 0, d: int = 6, shift: float = 2.2
) -> Dataset:
    """``sparse``: few minority samples relative to a large majority.

    Knobs: ``n_majority``, ``imratio``, ``d``, ``shift``.

    ``imratio=30`` is a measured compromise, arrived at from two failures:

    * At ``imratio=400`` the *test* split kept only **3** minority samples, so
      G-mean degenerated to the indicator function (0 or 1) — a coin flip.
    * At ``imratio=120`` the test split had 9 and the validation split only
      **4**. Stage-3 threshold calibration cannot be fitted on 4 samples, and
      when it was nevertheless applied it pushed 924/1209 test rows to the
      minority (G-mean 0.169, *worse* than plain SMOTE's 0.511).

    ``30`` yields ~100 minority samples total, so ~30 reach the holdout and ~14
    the validation split — enough for calibration to be fitted rather than
    guessed. The corresponding router gate is ``SPARSE_IR = 25``.

    ``shift=2.2`` sets how far the minority mode sits from the majority mode in
    the informative dimensions. This is the load-bearing knob: at ``shift=0.6``
    (the first value tried) the train/holdout G-mean gap was **+1.000** — the
    forest fitted the training minority perfectly and predicted *nothing*
    correctly on the holdout, i.e. the imbalance was not the difficulty, the
    absence of any learnable signal was. 3000 majority samples drawn from
    ``N(0, 1)`` swamp a ``+0.6`` displacement. At ``2.2`` the minority is
    genuinely separable from its immediate majority neighbourhood, which is
    what makes "can a method recover a rare class?" a well-posed question.
    """
    if imratio < 1.0:
        raise E100("imratio must be >= 1")
    n_min = max(4, int(round(n_majority / imratio)))
    rng = np.random.default_rng(seed)

    maj = rng.normal(0.0, 1.0, (n_majority, d))
    direction = np.zeros(d)
    direction[2:] = 1.0
    direction /= np.linalg.norm(direction)
    mnr = rng.normal(0.0, 1.0, (n_min, d)) + shift * direction

    X = _standardise(np.vstack([maj, mnr]))
    y = np.concatenate([np.zeros(n_majority, dtype=np.int64), np.ones(n_min, dtype=np.int64)])
    return Dataset(X=X, y=y, name="sparse")


def make_manifold(
    n_majority: int = 1500,
    imratio: float = 15.0,
    seed: int = 0,
    d: int = 4,
    noise: float = 0.05,
) -> Dataset:
    """``manifold`` (auxiliary): minority lies on a 1-D curve inside the bulk.

    Knobs: ``n_majority``, ``imratio``, ``d``, ``noise``. Resampling with
    isotropic noise is actively harmful here (it blurs the curve); the router
    should prefer manifold-aware expansion.

    ``noise=0.10`` left a train/holdout G-mean gap of **+0.58**: the curve was
    too thin for 120-tree forests to generalise, so the benchmark again
    measured memorisation. At ``0.05`` the curve is a genuine low-dimensional
    manifold the model can actually fit, and isotropic oversampling becomes
    the wrong tool — which is the point of including this regime.

    ``n_majority=1500, imratio=15`` keeps ~100 minority samples with ~45 in the
    holdout, which is the minimum at which per-seed G-mean stops being
    dominated by sampling noise.
    """
    if imratio < 1.0:
        raise E100("imratio must be >= 1")
    n_min = max(6, int(round(n_majority / imratio)))
    rng = np.random.default_rng(seed)

    maj = rng.normal(0.0, 1.0, (n_majority, d))
    t = rng.uniform(-np.pi, np.pi, n_min)
    curve = np.column_stack([np.cos(t), np.sin(t), np.cos(2.0 * t), np.sin(2.0 * t)])
    mnr = curve + rng.normal(0.0, noise, (n_min, d))

    X = _standardise(np.vstack([maj, mnr]))
    y = np.concatenate([np.zeros(n_majority, dtype=np.int64), np.ones(n_min, dtype=np.int64)])
    return Dataset(X=X, y=y, name="manifold")


DATASETS = {
    "separable": make_separable,
    "overlap": make_overlap,
    "multimodal": make_multimodal,
    "sparse": make_sparse,
    "manifold": make_manifold,
}


def dataset_defaults() -> dict[str, dict[str, float]]:
    """Tuned default knobs per dataset (found by difficulty sweep, see docs)."""
    return {
        "separable": {"n_majority": 1200, "imratio": 10.0, "d": 6},
        "overlap": {"n_majority": 2000, "imratio": 8.0, "d": 6, "sep": 0.9},
        "multimodal": {"n_majority": 1200, "imratio": 15.0, "d": 6, "k_modes": 3},
        "sparse": {"n_majority": 1200, "imratio": 20.0, "d": 6, "shift": 2.2},
        "manifold": {"n_majority": 1500, "imratio": 15.0, "d": 4, "noise": 0.05},
    }


def get_dataset(name: str, seed: int = 0, **overrides: float) -> Dataset:
    """Build a dataset by name, applying ``overrides`` on top of the defaults."""
    if name not in DATASETS:
        raise E100(f"unknown dataset {name!r}; valid: {sorted(DATASETS)}")
    kwargs = dict(dataset_defaults()[name])
    kwargs.update(overrides)
    return DATASETS[name](seed=seed, **kwargs)


def available_datasets() -> tuple[str, ...]:
    return tuple(DATASETS)
