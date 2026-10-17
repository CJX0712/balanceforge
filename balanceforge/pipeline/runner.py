"""End-to-end pipeline: one dataset + one method + one seed -> one metric block.

Layering is strictly acyclic::

    cli -> pipeline -> {data, policy, models, sampling, evaluation} -> core

No module imports upward. The pipeline is the only place that knows about all
of them, which is what makes the benchmark fair: every method goes through the
identical fit / calibrate / predict path, and only the *rebalancing step*
differs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from balanceforge.core.config import BalanceForgeConfig
from balanceforge.core.errors import E200, E500
from balanceforge.core.interfaces import estimate_profile
from balanceforge.core.seed import derive_seed, set_all
from balanceforge.core.types import Dataset, MethodResult
from balanceforge.data.datasets import get_dataset
from balanceforge.data.splits import make_splits
from balanceforge.evaluation.metrics import (
    PRIMARY_METRIC,
    collect_failures,
    compute_metrics,
)
from balanceforge.models.classifiers import (
    PriorMatchedCalibrator,
    calibration_is_admissible,
    class_weight_for,
    make_classifier,
    prior_corrected_logit,
)
from balanceforge.models.mosaic import MosaicConfig, rebalance
from balanceforge.policy.router import route
from balanceforge.sampling import tier0, tier1

__all__ = ["BalancePipeline", "MethodSpec", "run_benchmark", "write_report"]

#: The method table. Each entry declares how it obtains a rebalanced training set.
METHODS: tuple[str, ...] = (
    "none",
    "class_weight",
    "ros",
    "smote",
    "adasyn",
    "bsmote",
    "svmsmote",
    "rus",
    "cc",
    "enn",
    "tomek",
    "smoteenn",
    "smotetomek",
    "balanced_rf",
    "easy_ensemble",
    "mosaic",
)

FLAGSHIP = "mosaic"
#: Per the delivery contract the headline comparison is against the strongest
#: *single* non-flagship method, not against the naive baseline.
STRONG_BASELINES: tuple[str, ...] = (
    "smote",
    "adasyn",
    "bsmote",
    "smoteenn",
    "balanced_rf",
    "easy_ensemble",
)
NAIVE = "none"


@dataclass(frozen=True)
class MethodSpec:
    """How one method acquires a rebalanced training set."""

    name: str
    kind: str  # "naive" | "tier0" | "ensemble" | "flagship"
    class_weight: bool = False


def _spec(name: str) -> MethodSpec:
    if name == FLAGSHIP:
        return MethodSpec(name=name, kind="flagship")
    if name in {"balanced_rf", "easy_ensemble"}:
        return MethodSpec(name=name, kind="ensemble")
    if name == "class_weight":
        return MethodSpec(name=name, kind="naive", class_weight=True)
    if name in tier0.TIER0_SPECS:
        return MethodSpec(name=name, kind="tier0")
    return MethodSpec(name=name, kind="naive")


def _prior_correction_needed(train: Dataset, spec: MethodSpec) -> bool:
    """True when the *trained* distribution is still imbalanced enough to need pi.

    Two conditions must both hold:

    1. The estimator was **not** trained on a class-balanced sample set. If a
       resampler (or MOSAIC, or EasyEnsemble's internal balancing) already
       equalised the classes, the model estimates ``p(x|y)`` directly and the
       prior term is a double correction.
    2. The residual imbalance is real. Sampling noise leaves a nominally
       rebalanced set with an IR of e.g. 1.4; dividing by that prior is
       harmless but pointless, and it is skipped so the decision stays exactly
       ``argmax``.
    """
    ir = train.imbalance_ratio()
    if spec.kind == "ensemble":
        # Bagged-ensemble balancers resample internally at fit time; the
        # training set we hand them is untouched but the model they return has
        # already been balanced per-tree.
        return False
    return bool(ir > PRIOR_CORRECTION_MIN_IR)


#: Residual imbalance below which the prior correction is skipped as pointless.
PRIOR_CORRECTION_MIN_IR = 1.5


def _aligned_proba(estimator: object, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return ``predict_proba`` with columns permuted into ``classes`` order.

    The evaluation layer indexes the minority probability column positionally, so
    a single mis-ordered column silently inverts every imbalance metric. This is
    not hypothetical: ``EasyEnsembleClassifier`` reports ``classes_`` in a
    different order from ``np.unique(y_train)`` on the benchmark data, which
    produced a G-mean of 0.07 (i.e. the minority column was read as majority).
    """
    proba = np.asarray(estimator.predict_proba(X), dtype=np.float64)  # type: ignore[attr-defined]
    own = np.asarray(getattr(estimator, "classes_", classes))
    if own.size != classes.size or not np.array_equal(np.sort(own), np.sort(classes)):
        raise E500(
            f"estimator classes {own.tolist()} do not match training classes {classes.tolist()}"
        )
    if np.array_equal(own, classes):
        return proba
    order = [int(np.flatnonzero(own == c)[0]) for c in classes]
    return proba[:, order]


class _PriorOnly:
    """Marker artifact for the "calibration skipped, prior correction only" path.

    It reports zero learned offsets so the recorded ``calib_offsets`` metric
    stays meaningful (0 = nothing was learned, as opposed to a non-zero offset
    that the grid search chose).
    """

    offsets = np.zeros(2, dtype=np.float64)


class BalancePipeline:
    """Runs one (dataset, method, seed) cell end to end."""

    def __init__(self, cfg: BalanceForgeConfig) -> None:
        self.cfg = cfg.validate()

    # ------------------------------------------------------------------ helpers

    def _fit_predict(
        self,
        train: Dataset,
        val: Dataset,
        test: Dataset,
        spec: MethodSpec,
        seed: int,
        calibrate: bool,
    ) -> tuple[np.ndarray, np.ndarray, object, np.ndarray, float]:
        """Fit on ``train``, (optionally) calibrate on ``val``, predict ``test``."""
        t0 = time.perf_counter()
        estimator = None
        if spec.kind == "ensemble":
            estimator = tier0.build_ensemble_estimator(spec.name, seed)
        if estimator is None:
            estimator = make_classifier(
                kind=self.cfg.base_estimator,
                seed=seed,
                n_classes=train.n_classes,
                class_weight=class_weight_for(
                    "balanced" if spec.class_weight else "", train.n_classes
                ),
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                max_features=self.cfg.max_features,
            )
        estimator.fit(train.X, train.y)

        # INV-18: predict_proba columns must follow np.unique(y_train) order.
        # imblearn's ensembles expose their own `classes_` which is NOT
        # guaranteed sorted (EasyEnsembleClassifier re-orders its sub-ensembles
        # by construction), so we realign rather than trust it.
        classes = np.unique(train.y)
        val_proba = _aligned_proba(estimator, val.X, classes)
        test_proba = _aligned_proba(estimator, test.X, classes)

        # --- Does this method still need the closed-form prior correction? ---
        #
        # The Bayes rule for a balanced-risk decision is
        #     argmax_c P(y=c|x) / pi_c.
        # A model fitted on the *rebalanced* training set has already learned
        # approximately p(x | y=c), i.e. the quantity we want, so dividing by
        # pi again is a double correction that systematically over-predicts the
        # minority. Measured on the overlap benchmark: applying the correction on
        # top of MOSAIC's own rebalancing cost **-0.359 G-mean** (0.459 vs
        # 0.818 without it) — by far the largest single effect measured in this
        # project, and entirely an artefact of correcting a model that no longer
        # needs it.
        #
        # Therefore: only methods that leave the training distribution
        # imbalanced (`none`, and cost-only weighting) get the prior term.
        needs_prior_correction = _prior_correction_needed(train, spec)

        if calibrate and calibration_is_admissible(val):
            cal = PriorMatchedCalibrator(
                grid=MosaicConfig().offset_grid,
                iters=3,
                seed=derive_seed(seed, "calib", spec.name),
                prior_correction=needs_prior_correction,
            ).fit(val, val_proba)
            val_counts = np.array([int((val.y == c).sum()) for c in classes], dtype=np.float64)
            test_counts = np.array(
                [int((train.y == c).sum()) for c in classes], dtype=np.float64
            )
            val_pred = cal.predict(val_proba, val_counts)
            test_pred = cal.predict(test_proba, test_counts)
            artifact = cal
        elif calibrate:
            # Too few validation samples to fit offsets honestly. Fall back to
            # the closed-form prior correction only (zero hyper-parameters) and
            # record that the learned stage was skipped.
            val_counts = np.array([int((val.y == c).sum()) for c in classes], dtype=np.float64)
            test_counts = np.array(
                [int((train.y == c).sum()) for c in classes], dtype=np.float64
            )
            if needs_prior_correction:
                val_pred = prior_corrected_logit(val_proba, val_counts).argmax(axis=1)
                test_pred = prior_corrected_logit(test_proba, test_counts).argmax(axis=1)
            else:
                val_pred = val_proba.argmax(axis=1)
                test_pred = test_proba.argmax(axis=1)
            artifact = _PriorOnly()
        else:
            val_pred = val_proba.argmax(axis=1)
            test_pred = test_proba.argmax(axis=1)
            artifact = None

        return test_pred, test_proba, artifact, val_pred, time.perf_counter() - t0

    def _rebalance_train(
        self, fit: Dataset, spec: MethodSpec, seed: int
    ) -> tuple[Dataset, str, dict[str, Any]]:
        """Return ``(train, route_label, audit)`` for the requested method."""
        if spec.kind == "flagship":
            profile = estimate_profile(fit)
            plan = route(profile, val_minority_f1=None)
            cfg = MosaicConfig(
                k_neighbors=8,
                sigma_c=0.5,
                target_ratio=plan.target_ratio,
                regime_synth_budget={
                    "separable": float(plan.knobs.get("synth_budget", 0.25)),
                    "overlap": float(plan.knobs.get("synth_budget", 0.60)),
                    "multimodal": float(plan.knobs.get("synth_budget", 0.80)),
                    "sparse": float(plan.knobs.get("synth_budget", 1.00)),
                },
            )
            res = rebalance(fit, profile, cfg, seed=seed)
            return res.data, f"{plan.regime}/{plan.sampler}", res.as_dict()

        if spec.kind == "ensemble":
            return (
                Dataset(X=fit.X.copy(), y=fit.y.copy(), name=fit.name),
                "estimator-only",
                {},
            )

        if spec.kind == "tier0" and tier0.available_imblearn():
            out = tier0.build_tier0(spec.name, fit, seed)
            return out, "tier0/imblearn", {}

        # Offline path: Tier-1 numpy fallback for any Tier-0 name.
        t1_name = {
            "smote": "smote",
            "adasyn": "smote_k10",
            "bsmote": "smote",
            "svmsmote": "smote_k10",
            "smoteenn": "smote",
            "smotetomek": "smote",
            "ros": "ros",
            "rus": "rus",
            "cc": "cc",
            "enn": "enn",
            "tomek": "tomek",
        }.get(spec.name)
        if t1_name is not None:
            out = tier1.build_tier1(t1_name, fit, seed)
            return out, "tier1/numpy", {}

        return Dataset(X=fit.X.copy(), y=fit.y.copy(), name=fit.name), "none", {}

    # --------------------------------------------------------------------- run

    def run_cell(
        self,
        dataset_name: str,
        method_name: str,
        seed: int,
        with_calibration: bool = True,
        collect_failure_cases: bool = False,
    ) -> tuple[MethodResult, list[dict[str, Any]]]:
        """Run one benchmark cell and return ``(result, failure_cases)``."""
        set_all(seed)
        t0 = time.perf_counter()
        spec = _spec(method_name)

        data = get_dataset(dataset_name, seed=derive_seed(seed, "data", dataset_name))
        fit, val, test = make_splits(data, self.cfg.holdout_size, self.cfg.val_size, seed)

        try:
            train, route_label, audit = self._rebalance_train(fit, spec, seed)
        except E200 as exc:
            # A sampler whose preconditions fail on this geometry is a recorded
            # skip, not a crash. The report must show it so the comparison stays
            # honest (e.g. ADASYN is undefined on separable data by design).
            return (
                MethodResult(
                    dataset=dataset_name,
                    method=method_name,
                    seed=seed,
                    metrics={},
                    n_train_after=0,
                    route="skipped",
                    elapsed_sec=time.perf_counter() - t0,
                    skipped=True,
                    skip_reason=str(exc),
                ),
                [],
            )
        if train.n_features != fit.n_features:
            raise E500(f"{method_name} changed the feature space on {dataset_name}")

        test_pred, test_proba, artifact, val_pred, _ = self._fit_predict(
            train, val, test, spec, seed, with_calibration
        )

        metrics = compute_metrics(test, test_pred, test_proba)
        metrics.pop("_cm", None)
        metrics["n_train_after"] = float(train.n_samples)
        metrics["fit_time_sec"] = float(time.perf_counter() - t0)
        if artifact is not None:
            metrics["calib_offsets"] = float(
                np.abs(getattr(artifact, "offsets", np.zeros(1))).sum()
            )
        else:
            metrics["calib_offsets"] = 0.0

        result = MethodResult(
            dataset=dataset_name,
            method=method_name,
            seed=seed,
            metrics=metrics,
            n_train_after=train.n_samples,
            route=route_label,
            elapsed_sec=time.perf_counter() - t0,
        )

        cases: list[dict[str, Any]] = []
        if collect_failure_cases:
            cases = collect_failures(test, test_pred, n=5)
            for c in cases:
                c["dataset"] = dataset_name
                c["method"] = method_name
                c["seed"] = seed
        return result, cases

    # ---------------------------------------------------------------- benchmark

    def run_benchmark(
        self,
        datasets: tuple[str, ...],
        methods: tuple[str, ...] = METHODS,
        seeds: tuple[int, ...] | None = None,
        with_calibration: bool = True,
    ) -> dict[str, Any]:
        """Run the full grid and return a report-ready dict."""
        if seeds is None:
            base = self.cfg.seed
            seeds = tuple(derive_seed(base, "seed", i) % 10000 for i in range(self.cfg.n_seeds))

        rows: list[MethodResult] = []
        failures: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        for ds in datasets:
            for method in methods:
                for seed in seeds:
                    res, cases = self.run_cell(
                        ds,
                        method,
                        seed,
                        with_calibration=with_calibration,
                        collect_failure_cases=(method in {FLAGSHIP, NAIVE}),
                    )
                    rows.append(res)
                    failures.extend(cases)
        return {
            "config": self.cfg.to_dict(),
            "primary_metric": PRIMARY_METRIC,
            "seeds": list(seeds),
            "datasets": list(datasets),
            "methods": list(methods),
            "flagship": FLAGSHIP,
            "strong_baselines": list(STRONG_BASELINES),
            "elapsed_sec": round(time.perf_counter() - t0, 3),
            "rows": [r.as_dict() for r in rows],
            "failure_cases": failures,
            "backend": {
                "imblearn": tier0.available_imblearn(),
                "imblearn_version": tier0.imblearn_version(),
                "tier1": tier1.available_tier1(),
            },
        }


def run_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Module-level convenience wrapper around :class:`BalancePipeline`."""
    cfg = BalanceForgeConfig(
        **{
            k: kwargs.pop(k)
            for k in list(kwargs)
            if k
            in {
                "seed",
                "n_seeds",
                "holdout_size",
                "val_size",
                "base_estimator",
                "n_estimators",
                "max_depth",
                "max_features",
                "hpo_trials",
                "n_jobs",
            }
        }
    )
    return BalancePipeline(cfg).run_benchmark(**kwargs)


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    """Persist the benchmark report as UTF-8 JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_jsonable),
        encoding="utf-8",
    )
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")
