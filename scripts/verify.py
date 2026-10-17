"""Development harness: full matrix, ablations, and gate verification.

Not part of the shipped public API. Run with::

    python scripts/verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from balanceforge.models.classifiers import (  # noqa: E402
    calibration_is_admissible,
)

from balanceforge.core.config import BalanceForgeConfig  # noqa: E402
from balanceforge.core.seed import derive_seed  # noqa: E402
from balanceforge.data.datasets import available_datasets, get_dataset  # noqa: E402
from balanceforge.data.splits import make_splits  # noqa: E402
from balanceforge.pipeline.runner import METHODS, BalancePipeline  # noqa: E402

DATASETS = available_datasets()
SEEDS = tuple(derive_seed(42, "seed", i) % 10000 for i in range(3))


def cell_mean(pipe: BalancePipeline, ds: str, method: str) -> float:
    """Mean G-mean over seeds, or ``nan`` when every seed was skipped."""
    vals = []
    for s in SEEDS:
        res, _ = pipe.run_cell(ds, method, s)
        if res.skipped or "gmean" not in res.metrics:
            continue
        vals.append(res.metrics["gmean"])
    return float(np.mean(vals)) if vals else float("nan")


def std_mean(pipe: BalancePipeline, ds: str, method: str) -> tuple[float, float]:
    vals = []
    for s in SEEDS:
        res, _ = pipe.run_cell(ds, method, s)
        if res.skipped or "gmean" not in res.metrics:
            continue
        vals.append(res.metrics["gmean"])
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def main() -> int:
    pipe = BalancePipeline(BalanceForgeConfig(n_seeds=len(SEEDS), seed=42))

    print("=" * 84)
    print("dataset sanity")
    print("=" * 84)
    for ds in DATASETS:
        d = get_dataset(ds, seed=SEEDS[0])
        _, v, t = make_splits(d, 0.3, 0.2, SEEDS[0])
        print(
            f"  {ds:11s} full={np.bincount(d.y).tolist()} "
            f"val={np.bincount(v.y).tolist()} test={np.bincount(t.y).tolist()} "
            f"calib_admissible={calibration_is_admissible(v)}"
        )
    print()

    print("=" * 84)
    print("G-mean (mean of 3 seeds)")
    print("=" * 84)
    header = ["method"] + list(DATASETS) + ["AGG"]
    print("".join(f"{h:<14}" for h in header))
    print("-" * (14 * len(header)))
    rows = []
    for m in METHODS:
        vals = [cell_mean(pipe, ds, m) for ds in DATASETS]
        numeric = [v for v in vals if np.isfinite(v)]
        agg = float(np.mean(numeric)) if numeric else float("nan")
        rows.append((agg, m, vals, len(numeric)))
    rows.sort(key=lambda r: -r[0] if np.isfinite(r[0]) else 1e9)
    for agg, m, vals, n_ok in rows:
        tag = " <-- flagship" if m == "mosaic" else ""
        suffix = "" if n_ok == len(DATASETS) else f" ({n_ok}/{len(DATASETS)})"
        cells = "".join(f"{v:<14.4f}" if np.isfinite(v) else f"{'n/a':<14}" for v in vals)
        print(f"{m:<14}{cells}{agg:<14.4f}{suffix}{tag}")
    print()

    # ------------------------------------------------------------------ gate
    from balanceforge.cli import GATE_THRESHOLD, gate_verdict
    from balanceforge.pipeline.runner import FLAGSHIP, write_report

    report = pipe.run_benchmark(datasets=DATASETS, methods=METHODS, seeds=SEEDS)
    verdict = gate_verdict(report)

    print("=" * 84)
    print("headline gate")
    print("=" * 84)
    print(
        f"flagship     : {verdict['flagship']}  "
        f"mean={verdict.get('flagship_mean', float('nan')):.4f}  "
        f"std={verdict.get('flagship_std', float('nan')):.4f}"
    )
    print(
        f"strongest    : {verdict['strongest_baseline']}  "
        f"mean={verdict['strongest_baseline_mean']:.4f}  "
        f"std={verdict['strongest_baseline_std']:.4f}"
    )
    print(f"shared ds    : {', '.join(verdict.get('compared_datasets', []))}")
    print(
        f"delta        : {verdict['delta']:+.4f}  "
        f"(threshold {GATE_THRESHOLD:+.4f}, half-sigma {verdict['half_sigma_sum']:.4f})"
    )
    print(f"SIGNIFICANT  : {'PASS' if verdict['significant'] else 'FAIL'}")
    print()
    print("per-baseline (shared datasets only)")
    for m, v in sorted(verdict.get("per_baseline", {}).items(), key=lambda kv: -kv[1]["mean"]):
        print(
            f"  {m:<18} {v['mean']:.4f} +/- {v['std']:.4f}   "
            f"delta={verdict['flagship_mean'] - v['mean']:+.4f}"
        )
    print()

    # ------------------------------------------------------------- ablations
    print("=" * 84)
    print("ablations (flagship, 3 seeds)")
    print("=" * 84)
    from balanceforge.models.classifiers import (
        PriorMatchedCalibrator,
        make_classifier,
        prior_corrected_logit,
    )
    from balanceforge.models.mosaic import MosaicConfig, rebalance

    from balanceforge.core.interfaces import estimate_profile
    from balanceforge.evaluation.metrics import gmean_score
    from balanceforge.pipeline.runner import _prior_correction_needed, _spec

    def ablate(name: str, sigma=2.0, prior=True, learned=True) -> float:
        """Ablate one component of the flagship on the overlap regime.

        The prior term here mirrors the pipeline's own rule
        (``_prior_correction_needed``): MOSAIC's output is already class
        balanced (measured IR = 1.00), so the closed-form correction is *not*
        applied. An earlier version of this script applied it unconditionally
        and reported a spurious 0.36 "gain" from removing it, which was in fact
        a measurement of the double-correction bug, not of the component.
        """
        out = []
        ds = name[1]
        for s in SEEDS:
            d = get_dataset(ds, seed=derive_seed(s, "data", ds))
            fit, val, test = make_splits(d, 0.3, 0.2, s)
            prof = estimate_profile(fit)
            res = rebalance(fit, prof, MosaicConfig(sigma_c=sigma), seed=s)
            clf = make_classifier("rf", s, 2).fit(res.data.X, res.data.y)
            vp = clf.predict_proba(val.X)
            tp = clf.predict_proba(test.X)
            counts = np.array([int((res.data.y == c).sum()) for c in [0, 1]], float)
            apply_prior = bool(prior) and _prior_correction_needed(res.data, _spec(FLAGSHIP))
            if apply_prior and learned:
                cal = PriorMatchedCalibrator(seed=s).fit(val, vp)
                pred = cal.predict(tp, counts)
            elif apply_prior:
                pred = prior_corrected_logit(tp, counts).argmax(1)
            else:
                pred = tp.argmax(1)
            out.append(gmean_score(test.y, pred))
        return float(np.mean(out))

    full = ablate(("full", "overlap"))
    no_gate = ablate(("no_gate", "overlap"), sigma=1e6)
    no_prior = ablate(("no_prior", "overlap"), prior=False, learned=False)
    no_learned = ablate(("no_learned", "overlap"), learned=False)
    print(f"  full          overlap  {full:.4f}")
    print(f"  no_gate       overlap  {no_gate:.4f}   delta {no_gate - full:+.4f}")
    print(f"  no_prior      overlap  {no_prior:.4f}   delta {no_prior - full:+.4f}")
    print(f"  no_learned    overlap  {no_learned:.4f}   delta {no_learned - full:+.4f}")
    print()

    out = write_report(report, REPO_ROOT / "benchmark.json")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
