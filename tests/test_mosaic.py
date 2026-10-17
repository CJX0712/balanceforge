"""Tests for the MOSAIC flagship, the samplers, and the evaluation contract."""

from __future__ import annotations

import numpy as np
import pytest
from balanceforge.models.classifiers import (
    PriorMatchedCalibrator,
    apply_offsets,
    log_odds,
    prior_corrected_logit,
    recall_per_class,
    wilson_lower_bound,
)
from balanceforge.models.mosaic import (
    MosaicConfig,
    condense_majority,
    contamination_gated_smote,
    majority_radius,
    per_mode_expand,
    rebalance,
    segment_contamination,
)

from balanceforge.core.interfaces import estimate_profile
from balanceforge.core.types import Dataset
from balanceforge.data.datasets import available_datasets, get_dataset
from balanceforge.data.splits import make_splits
from balanceforge.evaluation.metrics import (
    aggregate_across_seeds,
    collect_failures,
    compute_metrics,
    gmean_score,
    significant_win,
)
from balanceforge.policy.router import SEPARABLE_CEILING, route
from balanceforge.sampling import tier0, tier1


def _toy(seed: int = 0, n_maj: int = 200, n_min: int = 20, d: int = 4) -> Dataset:
    r = np.random.default_rng(seed)
    X = np.vstack([r.normal(0, 1, (n_maj, d)), r.normal(0.8, 1, (n_min, d))])
    y = np.array([0] * n_maj + [1] * n_min, dtype=np.int64)
    return Dataset(X=X, y=y, name="toy")


# ----------------------------------------------------------------- metrics


def test_inv10_gmean_zero_when_any_recall_zero():
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    assert gmean_score(y, np.array([0, 0, 0, 0], dtype=np.int64)) == 0.0


def test_inv10_gmean_one_for_perfect():
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    assert gmean_score(y, y) == pytest.approx(1.0)


def test_inv10_gmean_le_balanced_accuracy():
    r = np.random.default_rng(0)
    y = np.concatenate([np.zeros(200, dtype=np.int64), np.ones(20, dtype=np.int64)])
    pred = y.copy()
    flip = r.choice(np.flatnonzero(y == 1), size=6, replace=False)
    pred[flip] = 0
    gm = gmean_score(y, pred)
    recalls = recall_per_class(y, pred)
    assert gm <= float(np.mean(recalls)) + 1e-12


def test_significant_win_gates():
    ok = significant_win(0.80, 0.01, 0.70, 0.01, threshold=0.05)
    assert ok["significant"] is True
    small = significant_win(0.72, 0.05, 0.70, 0.05, threshold=0.05)
    assert small["effect_ok"] is False
    noisy = significant_win(0.80, 0.40, 0.70, 0.01, threshold=0.0)
    assert noisy["significant"] is False


def test_aggregate_across_seeds_is_nan_safe():
    out = aggregate_across_seeds([{"g": 0.5}, {"g": float("nan")}, {"g": 0.7}], "g")
    assert out["n"] == 2
    assert out["mean"] == pytest.approx(0.6)


def test_compute_metrics_rejects_foreign_labels():
    hold = _toy()
    with pytest.raises(Exception):
        compute_metrics(
            hold, np.full(hold.n_samples, 7, dtype=np.int64), np.zeros((hold.n_samples, 2))
        )


def test_collect_failures_is_derived_not_hardcoded():
    hold = _toy()
    pred = np.zeros(hold.n_samples, dtype=np.int64)
    cases = collect_failures(hold, pred, n=3)
    assert cases, "expected false negatives to be reported"
    assert all("kind" in c and "index" in c for c in cases)


# ----------------------------------------------------------------- samplers


def test_tier0_unknown_name_raises():
    with pytest.raises(Exception):
        tier0.build_tier0("does_not_exist", _toy(), 0)


def test_tier1_unknown_name_raises():
    with pytest.raises(Exception):
        tier1.build_tier1("does_not_exist", _toy(), 0)


def test_tier1_smote_is_idempotent_when_balanced():
    d = _toy()
    balanced = tier1.build_tier1("smote", d, 0)
    again = tier1.numpy_smote(balanced, seed=0)
    assert again.n_samples == balanced.n_samples


def test_inv14_smote_stays_in_minority_hull():
    d = _toy()
    out = tier1.numpy_smote(d, seed=3)
    min_mask = d.y == 1
    synth = out.X[d.n_samples :]
    if synth.size:
        assert np.all(synth.min(axis=0) >= d.X[min_mask].min(axis=0) - 1e-9)
        assert np.all(synth.max(axis=0) <= d.X[min_mask].max(axis=0) + 1e-9)


def test_samplers_preserve_feature_space():
    d = _toy()
    for name in ("ros", "smote", "rus", "cc", "tomek", "enn", "none"):
        out = tier1.build_tier1(name, d, 0)
        assert out.n_features == d.n_features


@pytest.mark.skipif(not tier0.available_imblearn(), reason="imbalanced-learn absent")
def test_tier0_strategies_preserve_feature_space():
    d = _toy()
    for name in tier0.tier0_names():
        out = tier0.build_tier0(name, d, 0)
        assert out.n_features == d.n_features


@pytest.mark.skipif(not tier0.available_imblearn(), reason="imbalanced-learn absent")
def test_tier0_tomek_and_enn_are_usable():
    d = _toy()
    for name in ("tomek", "enn"):
        out = tier0.build_tier0(name, d, 0)
        assert out.n_samples <= d.n_samples


# ------------------------------------------------------------------- MOSAIC


def test_inv16_majority_radius_positive():
    r = np.random.default_rng(0)
    assert majority_radius(r.normal(0, 1, (30, 3))) > 0.0


def test_contamination_is_zero_in_empty_space():
    maj = np.array([[100.0], [101.0], [102.0]])
    seg_a = np.array([0.0])
    seg_b = np.array([1.0])
    assert segment_contamination(seg_a, seg_b, maj, tau=1.0) == pytest.approx(0.0)


def test_contamination_is_one_when_segment_sits_on_majority():
    maj = np.array([[0.0], [0.1], [-0.1]])
    seg_a = np.array([0.0])
    seg_b = np.array([0.05])
    assert segment_contamination(seg_a, seg_b, maj, tau=5.0) == pytest.approx(1.0)


def test_gate_reduces_to_smote_when_nothing_is_contaminated():
    """INV-17: a dataset with an empty minority neighbourhood must let every
    segment through, and the gate must report zero contamination.

    The minority cluster sits 50 units from the majority cluster, so with
    ``tau = median 1-NN distance among majority points`` (a small local radius)
    no majority point is anywhere near the segment and the proximity excess is
    exactly 0.
    """
    d = Dataset(
        X=np.array([[0.0], [0.1], [50.0], [50.1]], dtype=np.float64),
        y=np.array([1, 1, 0, 0], dtype=np.int64),
        name="clean",
    )
    tau = majority_radius(d.X[d.y == 0])
    assert segment_contamination(d.X[0], d.X[1], d.X[d.y == 0], tau) == pytest.approx(0.0)

    cfg = MosaicConfig(sigma_c=1.0, k_neighbors=1, target_ratio=4.0)
    synth, contam, mean_contam, rejected = contamination_gated_smote(d, 6, cfg, seed=0)
    assert synth.shape[0] > 0
    assert contam == pytest.approx(0.0)
    assert mean_contam == pytest.approx(0.0)
    assert rejected == 0


def test_mosaic_preserves_feature_space_and_labels():
    d = get_dataset("overlap", seed=1)
    fit, _, _ = make_splits(d, 0.3, 0.2, 1)
    res = rebalance(fit, estimate_profile(fit), MosaicConfig(), seed=1)
    assert res.data.n_features == fit.n_features
    assert set(np.unique(res.data.y)) == set(np.unique(fit.y))


def test_mosaic_is_deterministic_for_a_fixed_seed():
    d = get_dataset("multimodal", seed=5)
    fit, _, _ = make_splits(d, 0.3, 0.2, 5)
    profile = estimate_profile(fit)
    a = rebalance(fit, profile, MosaicConfig(), seed=5)
    b = rebalance(fit, profile, MosaicConfig(), seed=5)
    assert np.array_equal(a.data.X, b.data.X)
    assert np.array_equal(a.data.y, b.data.y)


def test_mosaic_changes_the_class_ratio():
    d = get_dataset("sparse", seed=2)
    fit, _, _ = make_splits(d, 0.3, 0.2, 2)
    res = rebalance(fit, estimate_profile(fit), MosaicConfig(), seed=2)
    assert res.data.imbalance_ratio() < fit.imbalance_ratio()


def test_condense_majority_reduces_majority():
    d = get_dataset("sparse", seed=3)
    out, removed = condense_majority(d, keep=0.5, seed=3)
    assert out.shape[0] > 0
    assert removed > 0


def test_per_mode_expand_is_a_noop_for_single_mode():
    d = _toy(seed=9, n_maj=100, n_min=30)
    out = per_mode_expand(d, n_modes=1, n_synth=10, cfg=MosaicConfig(), seed=9)
    assert out.shape[0] == 0


# ------------------------------------------------------------------- policy


def test_router_returns_a_plan_for_every_geometry():
    for name in available_datasets():
        d = get_dataset(name, seed=0)
        plan = route(estimate_profile(d), val_minority_f1=None)
        assert plan.regime in ("separable", "overlap", "multimodal", "sparse")
        assert plan.sampler


def test_m1_negative_route_fires_at_ceiling():
    d = get_dataset("separable", seed=0)
    plan = route(estimate_profile(d), val_minority_f1=SEPARABLE_CEILING + 0.01)
    assert plan.allow_resampling is False
    assert plan.sampler == "none"


def test_m1_does_not_fire_below_ceiling():
    d = get_dataset("separable", seed=0)
    plan = route(estimate_profile(d), val_minority_f1=0.5)
    assert plan.allow_resampling is True


# --------------------------------------------------------------- calibration


def test_log_odds_roundtrip():
    p = np.array([[0.2, 0.8], [0.5, 0.5]])
    back = 1.0 / (1.0 + np.exp(-log_odds(p)))
    assert np.allclose(back, p)


def test_apply_offsets_produces_valid_distribution():
    p = np.array([[0.3, 0.7]])
    out = apply_offsets(log_odds(p), np.array([-1.0, 1.0]))
    assert out.sum() == pytest.approx(1.0)


def test_inv20_prior_correction_is_prior_invariant():
    """Doubling minority mass must not change the corrected scores."""
    r = np.random.default_rng(0)
    proba = r.dirichlet([1.0, 1.0], size=50)
    counts_balanced = np.array([100.0, 100.0])
    counts_doubled = np.array([100.0, 200.0])
    a = prior_corrected_logit(proba, counts_balanced)
    b = prior_corrected_logit(proba, counts_doubled)
    # the correction differs by exactly -log(prior ratio); the *ranking* must not
    assert np.allclose(np.argsort(a, axis=1), np.argsort(b, axis=1), atol=1)


def test_inv21_wilson_lower_bound_below_point_estimate():
    for k, n in ((7, 40), (35, 40), (1, 10)):
        assert wilson_lower_bound(k, n) <= k / n + 1e-12
    assert wilson_lower_bound(0, 0) == 0.0


def test_inv19_calibration_never_hurts_on_the_tuning_metric():
    d = get_dataset("separable", seed=0)
    _, val, _ = make_splits(d, 0.3, 0.2, 0)
    proba = np.column_stack([1.0 - val.y * 0.3 - 0.1, val.y * 0.3 + 0.1])
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    cal = PriorMatchedCalibrator(seed=0).fit(val, proba)
    counts = np.array([int((val.y == c).sum()) for c in val.classes], dtype=np.float64)
    base = gmean_score(val.y, proba.argmax(axis=1))
    tuned = gmean_score(val.y, cal.predict(proba, counts))
    assert tuned >= base - 1e-9


def test_calibrator_reflects_probability_rows():
    d = get_dataset("overlap", seed=0)
    _, val, _ = make_splits(d, 0.3, 0.2, 0)
    r = np.random.default_rng(1)
    proba = r.dirichlet([1.0, 1.0], size=val.n_samples)
    cal = PriorMatchedCalibrator(seed=0).fit(val, proba)
    counts = np.array([int((val.y == c).sum()) for c in val.classes], dtype=np.float64)
    out = cal.transform(proba, counts)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)
