"""Tier-0 balancers: thin, uniform wrappers over the strongest open-source
imbalance-learning implementations.

Tier-0 packages (no reinvention, per the delivery contract):

==================  ==================================================
Role                Package / estimator
==================  ==================================================
over-sampling       ``imblearn.over_sampling``: SMOTE, ADASYN,
                    BorderlineSMOTE, SVMSMOTE, RandomOverSampler
under-sampling      ``imblearn.under_sampling``: RandomUnderSampler,
                    ClusterCentroids, TomekLinks, EditedNearestNeighbours
hybrid              ``imblearn.combine``: SMOTEENN, SMOTETomek
cost-sensitive      ``class_weight='balanced'`` (no resampling at all)
bagged ensembles    ``imblearn.ensemble``: BalancedRandomForest,
                    EasyEnsemble
==================  ==================================================

Every wrapper satisfies the same contract: ``fit_resample(data, seed) ->
Dataset`` with an unchanged feature space and label set. That uniformity is what
makes the cross-method benchmark fair (INV-12).
"""

from __future__ import annotations

import numpy as np

from balanceforge.core.errors import E200
from balanceforge.core.types import Dataset

__all__ = [
    "available_imblearn",
    "imblearn_version",
    "TIER0_SPECS",
    "build_tier0",
    "tier0_names",
]


def available_imblearn() -> bool:
    """True when ``imbalanced-learn`` is importable (Tier-0 online)."""
    try:
        import imblearn  # noqa: F401
    except Exception:
        return False
    return True


def imblearn_version() -> str:
    try:
        import imblearn

        return str(imblearn.__version__)
    except Exception:
        return "unavailable"


def _rf_estimator(kind: str, seed: int, n_estimators: int, **kw: object):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1, **kw)


TIER0_SPECS: dict[str, dict[str, object]] = {
    # ---- naive: no balancing at all (the "why bother" reference) ----
    "none": {"kind": "identity", "desc": "no balancing (naive reference)"},
    # ---- over-sampling ----
    "ros": {"kind": "over", "cls": "RandomOverSampler", "desc": "random over-sampling"},
    "smote": {
        "kind": "over",
        "cls": "SMOTE",
        "k": 5,
        "desc": "SMOTE (interpolate minority between k-NN)",
    },
    "adasyn": {
        "kind": "over",
        "cls": "ADASYN",
        "desc": "ADASYN (adaptive difficulty-weighted synthesis)",
    },
    "bsmote": {
        "kind": "over",
        "cls": "BorderlineSMOTE",
        "k": 5,
        "desc": "BorderlineSMOTE (only border-region minorities)",
    },
    "svmsmote": {
        "kind": "over",
        "cls": "SVMSMOTE",
        "desc": "SVMSMOTE (SVM margin as the synthesis region)",
    },
    # ---- under-sampling ----
    "rus": {
        "kind": "under",
        "cls": "RandomUnderSampler",
        "desc": "random under-sampling",
    },
    "cc": {
        "kind": "under",
        "cls": "ClusterCentroids",
        "desc": "ClusterCentroids (k-means prototype compression)",
    },
    "enn": {
        "kind": "clean",
        "cls": "EditedNearestNeighbours",
        "desc": "ENN (remove majority points mislabelled by 1-NN)",
    },
    "tomek": {
        "kind": "clean",
        "cls": "TomekLinks",
        "desc": "Tomek links (remove cross-class boundary pairs)",
    },
    # ---- hybrid ----
    "smoteenn": {
        "kind": "combo",
        "cls": "SMOTEENN",
        "desc": "SMOTE + ENN (over-sample then clean)",
    },
    "smotetomek": {
        "kind": "combo",
        "cls": "SMOTETomek",
        "desc": "SMOTE + Tomek (over-sample then prune)",
    },
    # ---- cost-sensitive / ensemble ----
    "class_weight": {
        "kind": "identity",
        "desc": "class_weight='balanced' (cost-sensitive, no resampling)",
        "class_weight": "balanced",
    },
    "balanced_rf": {
        "kind": "ensemble",
        "cls": "BalancedRandomForestClassifier",
        "desc": "BalancedRandomForest (bagging with per-tree balanced bootstrap)",
        "n_estimators": 120,
    },
    "easy_ensemble": {
        "kind": "ensemble",
        "cls": "EasyEnsembleClassifier",
        "desc": "EasyEnsemble (AdaBoost-style balanced sub-ensembles)",
        "n_estimators": 10,
    },
}


def tier0_names() -> tuple[str, ...]:
    return tuple(TIER0_SPECS)


# Estimators whose constructor does NOT accept random_state in imbalanced-learn
# 0.14.x (API drift: TomekLinks / EditedNearestNeighbours are deterministic).
_NO_RANDOM_STATE = frozenset({"TomekLinks", "EditedNearestNeighbours"})


def _sampler_kwargs(spec: dict[str, object], seed: int) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if (
        spec["kind"] in {"over", "under", "clean", "combo"}
        and str(spec.get("cls", "")) not in _NO_RANDOM_STATE
    ):
        kwargs["random_state"] = seed
    for key in ("k", "k_neighbors"):
        if key in spec:
            kwargs["k_neighbors"] = int(spec[key])  # type: ignore[arg-type]
    return kwargs


def build_tier0(name: str, data: Dataset, seed: int) -> Dataset:
    """Apply the named Tier-0 balancing strategy to ``data``.

    ``none`` and ``class_weight`` are pass-through: they change the *loss*, not
    the sample distribution, so the returned dataset is identical to the input.
    The caller is responsible for honouring the ``class_weight`` marker when it
    constructs the classifier.
    """
    if name not in TIER0_SPECS:
        raise E200(f"unknown tier0 strategy {name!r}; valid: {sorted(TIER0_SPECS)}")
    if not available_imblearn():
        raise E200("imbalanced-learn is unavailable; use the Tier-1 numpy path")

    spec = TIER0_SPECS[name]
    kind = spec["kind"]
    if kind == "identity":
        return Dataset(X=data.X.copy(), y=data.y.copy(), name=data.name)

    cls_name = str(spec["cls"])
    kwargs = _sampler_kwargs(spec, seed)

    if kind == "over":
        from imblearn.over_sampling import (
            ADASYN,
            SMOTE,
            SVMSMOTE,
            BorderlineSMOTE,
            RandomOverSampler,
        )

        table = {
            "RandomOverSampler": RandomOverSampler,
            "SMOTE": SMOTE,
            "ADASYN": ADASYN,
            "BorderlineSMOTE": BorderlineSMOTE,
            "SVMSMOTE": SVMSMOTE,
        }
    elif kind == "under":
        from imblearn.under_sampling import (
            ClusterCentroids,
            RandomUnderSampler,
        )

        table = {
            "RandomUnderSampler": RandomUnderSampler,
            "ClusterCentroids": ClusterCentroids,
        }
    elif kind == "clean":
        from imblearn.under_sampling import (
            EditedNearestNeighbours,
            TomekLinks,
        )

        table = {
            "EditedNearestNeighbours": EditedNearestNeighbours,
            "TomekLinks": TomekLinks,
        }
    elif kind == "combo":
        from imblearn.combine import SMOTEENN, SMOTETomek

        table = {"SMOTEENN": SMOTEENN, "SMOTETomek": SMOTETomek}
    elif kind == "ensemble":
        table = {}
    else:  # pragma: no cover - guarded by TIER0_SPECS
        raise E200(f"unhandled tier0 kind {kind!r}")

    if kind == "ensemble":
        return _build_ensemble(spec, data, seed)

    sampler = table[cls_name](**kwargs)
    try:
        Xr, yr = sampler.fit_resample(data.X, data.y)
    except (RuntimeError, ValueError) as exc:
        # Some samplers have hard preconditions that simply do not hold in every
        # imbalance geometry. ADASYN raises "Not any neigbours belong to the
        # majority class" on separable data (by design — it has no informative
        # region to sample from), and SVMSMOTE can fail on tiny minority sets.
        # That is a property of the *method*, not a bug, so the benchmark must
        # record it as a legitimate skip rather than crash the whole grid.
        raise E200(
            f"{cls_name} cannot run on this dataset: {type(exc).__name__}: {exc}"
        ) from exc
    if Xr.shape[1] != data.n_features:
        raise E200(f"feature space changed under {name}: {data.n_features} -> {Xr.shape[1]}")
    return Dataset(
        X=np.asarray(Xr, dtype=np.float64), y=np.asarray(yr, dtype=np.int64), name=data.name
    )


def _build_ensemble(spec: dict[str, object], data: Dataset, seed: int) -> Dataset:
    """Bagged-ensemble balancers change the estimator, not the data.

    They are surfaced through :data:`ENSEMBLE_MARKERS` so the pipeline knows to
    construct the corresponding estimator rather than resampling.
    """
    return Dataset(X=data.X.copy(), y=data.y.copy(), name=data.name)


ENSEMBLE_MARKERS: dict[str, dict[str, object]] = {
    "balanced_rf": {"estimator": "BalancedRandomForestClassifier", "n_estimators": 120},
    "easy_ensemble": {"estimator": "EasyEnsembleClassifier", "n_estimators": 10},
}


def build_ensemble_estimator(name: str, seed: int):
    """Instantiate a bagged-ensemble classifier for ``name`` (or ``None``)."""
    if name not in ENSEMBLE_MARKERS:
        return None
    if not available_imblearn():
        return None
    marker = ENSEMBLE_MARKERS[name]
    cls_name = str(marker["estimator"])
    n_estimators = int(marker["n_estimators"])
    if cls_name == "BalancedRandomForestClassifier":
        from imblearn.ensemble import BalancedRandomForestClassifier

        return BalancedRandomForestClassifier(
            n_estimators=n_estimators, random_state=seed, n_jobs=1
        )
    from imblearn.ensemble import EasyEnsembleClassifier

    return EasyEnsembleClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1)
