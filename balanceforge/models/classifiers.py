"""Classifiers, prior correction and prior-matched calibration.

Three pieces, all closed-form or grid-searched — no gradient fitting, so the
whole module is deterministic and dependency-light:

* :func:`log_odds` / :func:`apply_offsets` / :func:`prior_corrected_logit` —
  the score-space algebra behind calibration.
* :func:`wilson_lower_bound` / :func:`recall_per_class` — the evaluation
  primitives the calibrator optimises against.
* :class:`PriorMatchedCalibrator` — per-class logit offsets searched on a
  validation split to maximise G-mean (INV-19: never worse than the
  uncalibrated argmax on the tuning metric).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from balanceforge.core.types import Dataset

# --------------------------------------------------------------- score algebra


def log_odds(p: np.ndarray) -> np.ndarray:
    """Element-wise logit ``log(p / (1 - p))`` with clipping for stability."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return np.log(p) - np.log1p(-p)


def apply_offsets(z: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Row-wise softmax of ``z + offsets`` — a valid probability distribution."""
    z = np.asarray(z, dtype=np.float64) + np.asarray(offsets, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def prior_corrected_logit(proba: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Closed-form prior correction: ``logit(p) - log(pi)``.

    Dividing by the class prior removes the training distribution's bias.
    The correction is a *per-class constant*, so it never changes the
    ranking within a row (INV-20) — only the operating point moves.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return log_odds(proba)
    prior = counts / total
    return log_odds(proba) - np.log(prior)[None, :]


# ---------------------------------------------------------------- evaluation


def wilson_lower_bound(k: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for ``k/n``.

    The honest way to report a small-sample rate: the point estimate alone
    is meaningless at ``n = 10``. ``k = n = 0`` is defined as 0.
    """
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    radicand = p * (1.0 - p) / n + z * z / (4.0 * n * n)
    rad = z * float(np.sqrt(max(radicand, 0.0)))
    return float(max(0.0, (centre - rad) / denom))


def recall_per_class(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-class recall in ``np.unique(y)`` order."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    out = []
    for c in np.unique(y):
        mask = y == c
        out.append(float((pred[mask] == c).mean()) if mask.any() else 0.0)
    return np.array(out, dtype=np.float64)


# --------------------------------------------------------------- classifiers


def make_classifier(
    kind: str,
    seed: int,
    n_classes: int,
    class_weight: dict | str | None = None,
    n_estimators: int = 200,
    max_depth: int | None = None,
    max_features: str | int | float | None = "sqrt",
):
    """Build a seeded classifier. ``rf`` is the default base estimator.

    scikit-learn is a declared dependency; a weaker fallback learner is
    deliberately not substituted, because that would silently corrupt every
    benchmark number.
    """
    if kind in ("rf", "random_forest", "randomforest"):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"unknown classifier {kind!r}")


# --------------------------------------------------------------- calibration

#: Minimum per-class validation samples for an honest offset fit. Below this
#: the grid search overfits the split and the calibrator must be skipped
#: (the runner then falls back to the closed-form prior correction only).
ADMISSIBLE_MIN_PER_CLASS = 20


def calibration_is_admissible(val: Dataset) -> bool:
    """Is the validation split large enough to fit per-class offsets honestly?

    Guard-rail for INV-19: the calibrator is only trusted when every class
    has enough validation samples; otherwise its grid search would memorise
    the split instead of learning the operating point.
    """
    counts = val.counts()
    return bool(counts.size and int(counts.min()) >= ADMISSIBLE_MIN_PER_CLASS)


def class_weight_for(spec: str, n_classes: int) -> dict[int, float] | str | None:
    """Translate a class-weight spec into a sklearn-ready value.

    ``"balanced"`` → sklearn's own ``"balanced"``; ``""``/``"none"`` → ``None``
    (uniform); anything else raises so a typo fails loudly instead of
    silently training an unweighted model.
    """
    key = (spec or "").strip().lower()
    if key in ("", "none", "uniform"):
        return None
    if key == "balanced":
        return "balanced"
    raise ValueError(f"unknown class_weight spec {spec!r}")


@dataclass(frozen=True)
class PriorMatchedCalibrator:
    """Grid-searched per-class logit offsets, fitted on a validation split.

    The grid always contains the zero offset, so the fitted calibrator can
    never score below the uncalibrated argmax on the tuning metric (INV-19).
    When ``prior_correction`` is set, the closed-form prior term is applied
    *before* the offsets — that is the only ordering under which the two
    stages do not double-correct.
    """

    grid: tuple[float, ...] = field(
        default_factory=lambda: tuple(round(-1.5 + 0.25 * i, 4) for i in range(13))
    )
    iters: int = 3
    seed: int = 0
    prior_correction: bool = True
    offsets_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    fitted_: bool = field(default=False, init=False, repr=False)

    def fit(self, val: Dataset, proba: np.ndarray) -> PriorMatchedCalibrator:
        proba = np.asarray(proba, dtype=np.float64)
        k = proba.shape[1]
        counts = np.array([int((val.y == c).sum()) for c in val.classes], dtype=np.float64)
        base = (
            prior_corrected_logit(proba, counts) if self.prior_correction else log_odds(proba)
        )
        offsets = np.zeros(k, dtype=np.float64)
        best_score = self._score(val.y, base, offsets)
        # coordinate ascent over classes, repeated `iters` sweeps
        for _ in range(max(1, self.iters)):
            improved = False
            for c in range(k):
                local_best = offsets[c]
                for off in self.grid:
                    trial = offsets.copy()
                    trial[c] = off
                    s = self._score(val.y, base, trial)
                    if s > best_score + 1e-12:
                        best_score = s
                        local_best = off
                        improved = True
                offsets[c] = local_best
            if not improved:
                break
        object.__setattr__(self, "offsets_", offsets)
        object.__setattr__(self, "fitted_", True)
        return self

    @staticmethod
    def _score(y: np.ndarray, base: np.ndarray, offsets: np.ndarray) -> float:
        proba = apply_offsets(base, offsets)
        pred = proba.argmax(axis=1)
        recalls = recall_per_class(y, pred)
        if recalls.size == 0:
            return 0.0
        return float(np.exp(np.mean(np.log(np.maximum(recalls, 1e-12)))))

    def transform(self, proba: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Apply (optional prior correction +) fitted offsets, then softmax."""
        proba = np.asarray(proba, dtype=np.float64)
        base = (
            prior_corrected_logit(proba, counts) if self.prior_correction else log_odds(proba)
        )
        offsets = self.offsets_ if self.offsets_ is not None else np.zeros(proba.shape[1])
        return apply_offsets(base, offsets)

    def predict(self, proba: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Class labels after calibration."""
        return self.transform(proba, counts).argmax(axis=1)
