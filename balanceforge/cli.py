"""Command-line entry point for BalanceForge.

Examples
--------
::

    python -m balanceforge benchmark --out benchmark.json
    python -m balanceforge profile --dataset overlap
    python -m balanceforge rebalance --dataset sparse --seed 7
    python -m balanceforge sweep
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from balanceforge.core.config import load_config
from balanceforge.core.interfaces import estimate_profile
from balanceforge.core.seed import derive_seed, set_all
from balanceforge.data.datasets import available_datasets, get_dataset
from balanceforge.data.splits import make_splits
from balanceforge.models.mosaic import MosaicConfig, rebalance
from balanceforge.pipeline.runner import FLAGSHIP, METHODS, BalancePipeline, write_report

# GBK consoles cannot render the status glyphs; force UTF-8 so the tables print.
if hasattr(sys.stdout, "reconfigure"):
    with contextlib.suppress(Exception):  # pragma: no cover
        sys.stdout.reconfigure(encoding="utf-8")


def _print_table(rows: list[dict[str, Any]], columns: list[str], width: int = 14) -> None:
    """Print a fixed-width table so numbers never drift between rows."""
    header = "".join(f"{c:<{width}}" for c in columns)
    print(header)
    print("-" * (width * len(columns)))
    for r in rows:
        print("".join(f"{str(r.get(c, '')):<{width}}" for c in columns))


def cmd_benchmark(args: argparse.Namespace) -> int:
    cfg = load_config(
        {
            "seed": args.seed,
            "n_seeds": args.n_seeds,
            "base_estimator": args.estimator,
        }
    )
    set_all(cfg.seed)
    datasets = tuple(args.datasets) if args.datasets else available_datasets()
    methods = tuple(args.methods) if args.methods else METHODS

    pipe = BalancePipeline(cfg)
    report = pipe.run_benchmark(datasets=datasets, methods=methods)

    print()
    print("=" * 78)
    print("BalanceForge benchmark")
    print("=" * 78)
    print(f"primary metric : {report['primary_metric']} (geometric mean of per-class recall)")
    print(f"datasets       : {', '.join(datasets)}")
    print(f"methods        : {len(methods)}")
    print(f"seeds          : {report['seeds']}")
    print(
        f"backend        : imblearn={report['backend']['imblearn']} "
        f"({report['backend']['imblearn_version']})"
    )
    print(f"elapsed        : {report['elapsed_sec']:.1f}s")
    print()

    summary = summarise(report)
    _print_table(
        summary,
        ["method", "gmean_mean", "gmean_std", "macro_f1", "mcc", "n_datasets"],
        width=13,
    )

    verdict = gate_verdict(report)
    print()
    print("-" * 78)
    print(f"flagship        : {verdict['flagship']}")
    print(
        f"strongest base  : {verdict['strongest_baseline']} "
        f"({verdict['strongest_baseline_mean']:.4f})"
    )
    print(f"delta           : {verdict['delta']:+.4f}")
    print(
        f"threshold       : {verdict['threshold']:+.4f}  "
        f"(half-sigma-sum: {verdict['half_sigma_sum']:.4f})"
    )
    print(f"SIGNIFICANT     : {'PASS' if verdict['significant'] else 'FAIL'}")
    print("-" * 78)

    out = write_report(report, args.out)
    print(f"report written  : {out}")
    return 0


def summarise(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate per-seed rows into mean/std per (method, dataset) and overall."""
    from balanceforge.evaluation.metrics import aggregate_across_seeds

    buckets: dict[tuple[str, str], list[dict[str, float]]] = {}
    for row in report["rows"]:
        if row.get("skipped"):
            continue
        buckets.setdefault((row["method"], row["dataset"]), []).append(row["metrics"])

    per_method: dict[str, dict[str, float]] = {}
    for (method, _ds), rows in buckets.items():
        agg = per_method.setdefault(
            method,
            {"gmean_sum": 0.0, "f1_sum": 0.0, "mcc_sum": 0.0, "n": 0.0, "gmean_sq": 0.0},
        )
        gm = aggregate_across_seeds(rows, "gmean")["mean"]
        agg["gmean_sum"] += gm
        agg["f1_sum"] += aggregate_across_seeds(rows, "macro_f1")["mean"]
        agg["mcc_sum"] += aggregate_across_seeds(rows, "mcc")["mean"]
        agg["gmean_sq"] += gm * gm
        agg["n"] += 1.0

    out: list[dict[str, Any]] = []
    for method, agg in per_method.items():
        n = max(1.0, agg["n"])
        mean = agg["gmean_sum"] / n
        var = max(0.0, agg["gmean_sq"] / n - mean * mean)
        out.append(
            {
                "method": method,
                "gmean_mean": f"{mean:.4f}",
                "gmean_std": f"{var**0.5:.4f}",
                "macro_f1": f"{agg['f1_sum'] / n:.4f}",
                "mcc": f"{agg['mcc_sum'] / n:.4f}",
                "n_datasets": int(n),
            }
        )
    out.sort(key=lambda r: -float(r["gmean_mean"]))
    return out


#: Effect size the flagship must beat, per the delivery contract.
GATE_THRESHOLD = 0.02


def gate_verdict(report: dict[str, Any]) -> dict[str, Any]:
    """Compare the flagship against the strongest single non-flagship method.

    The comparison is restricted to the datasets where **both** the flagship and
    the baseline actually ran. Without that restriction a baseline that skips
    half the suite (ADASYN is undefined on separable data) would be scored on
    whichever datasets happen to flatter it, which is exactly the kind of
    selection bias the gate is supposed to prevent.
    """
    from balanceforge.evaluation.metrics import aggregate_across_seeds

    buckets: dict[tuple[str, str], list[dict[str, float]]] = {}
    for row in report["rows"]:
        if row.get("skipped"):
            continue
        buckets.setdefault((row["method"], row["dataset"]), []).append(row["metrics"])

    datasets = list(report["datasets"])
    flag_ok = {ds for (m, ds) in buckets if m == FLAGSHIP}

    def method_profile(method: str) -> tuple[float, float]:
        shared = [ds for ds in datasets if ds in flag_ok and (method, ds) in buckets]
        if not shared:
            return float("nan"), float("nan"), []
        means = [
            aggregate_across_seeds(buckets[(method, ds)], "gmean")["mean"] for ds in shared
        ]
        return float(np.mean(means)), float(np.std(means)), shared

    flag_mean, flag_std, _ = method_profile(FLAGSHIP)
    finite: dict[str, tuple[float, float, list[str]]] = {}
    for m in report["strong_baselines"]:
        mean, std, shared = method_profile(m)
        if np.isfinite(mean):
            finite[m] = (mean, std, shared)
    if not finite:
        for m in report["methods"]:
            if m == FLAGSHIP:
                continue
            mean, std, shared = method_profile(m)
            if np.isfinite(mean):
                finite[m] = (mean, std, shared)
    if not finite:
        return {
            "flagship": FLAGSHIP,
            "significant": False,
            "delta": float("nan"),
            "threshold": GATE_THRESHOLD,
            "half_sigma_sum": float("nan"),
            "strongest_baseline": "n/a",
            "note": "no comparable baseline ran on any dataset",
        }

    strongest = max(finite, key=lambda m: finite[m][0])

    from balanceforge.evaluation.metrics import significant_win

    win = significant_win(
        flag_mean, flag_std, finite[strongest][0], finite[strongest][1], GATE_THRESHOLD
    )
    return {
        "flagship": FLAGSHIP,
        "flagship_mean": flag_mean,
        "flagship_std": flag_std,
        "strongest_baseline": strongest,
        "strongest_baseline_mean": finite[strongest][0],
        "strongest_baseline_std": finite[strongest][1],
        "compared_datasets": finite[strongest][2],
        "delta": float(win["diff"]),
        "threshold": GATE_THRESHOLD,
        "half_sigma_sum": float(win["half_sigma_sum"]),
        "significant": bool(win["significant"]),
        "per_baseline": {m: {"mean": v[0], "std": v[1]} for m, v in finite.items()},
    }


def cmd_profile(args: argparse.Namespace) -> int:
    set_all(args.seed)
    for name in [args.dataset] if args.dataset else available_datasets():
        d = get_dataset(name, seed=derive_seed(args.seed, "data", name))
        p = estimate_profile(d)
        print(
            f"{name:12s} n={d.n_samples:6d} d={d.n_features:2d} "
            f"IR={p.imbalance_ratio:7.1f} -> {p.geometric:11s} "
            f"ovl={p.overlap_ratio:.3f} k={p.minority_cluster_count} "
            f"ent={p.knn_entropy:.3f}"
        )
    return 0


def cmd_rebalance(args: argparse.Namespace) -> int:
    set_all(args.seed)
    d = get_dataset(args.dataset, seed=derive_seed(args.seed, "data", args.dataset))
    fit, _, _ = make_splits(d, 0.3, 0.2, args.seed)
    profile = estimate_profile(fit)
    res = rebalance(fit, profile, MosaicConfig(), seed=args.seed)
    print(
        json.dumps(
            {
                "before": {
                    "n": fit.n_samples,
                    "ir": round(fit.imbalance_ratio(), 2),
                },
                "after": res.as_dict(),
            },
            indent=2,
        )
    )
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from balanceforge.hpo import sweep

    out = sweep()
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"best: {json.dumps(out['best'], indent=2)}")
    print(f"elapsed: {out['elapsed_sec']}s -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="balanceforge",
        description="BalanceForge — geometry-routed rebalancing for imbalanced learning",
    )
    p.add_argument("--seed", type=int, default=42, help="master seed")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("benchmark", help="run the full benchmark grid")
    b.add_argument("--out", default="benchmark.json", help="report path")
    b.add_argument("--n-seeds", type=int, default=3)
    b.add_argument("--estimator", default="rf", choices=["rf", "et", "lr", "dt"])
    b.add_argument("--datasets", nargs="*", default=[])
    b.add_argument("--methods", nargs="*", default=[])
    b.set_defaults(func=cmd_benchmark)

    pr = sub.add_parser("profile", help="print imbalance geometry profiles")
    pr.add_argument("--dataset", default=None)
    pr.set_defaults(func=cmd_profile)

    rb = sub.add_parser("rebalance", help="run the flagship on one dataset")
    rb.add_argument("--dataset", default="overlap")
    rb.set_defaults(func=cmd_rebalance)

    sw = sub.add_parser("sweep", help="grid-search the flagship's knobs")
    sw.add_argument("--out", default="sweep.json")
    sw.set_defaults(func=cmd_sweep)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
