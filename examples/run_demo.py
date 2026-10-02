"""End-to-end demonstration: run the benchmark, print the tables, write JSON.

    python examples/run_demo.py

Everything printed here comes from a real run; no number is hard-coded.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from balanceforge.cli import (  # noqa: E402
    GATE_THRESHOLD,
    gate_verdict,
)
from balanceforge.core.config import BalanceForgeConfig  # noqa: E402
from balanceforge.core.seed import set_all  # noqa: E402
from balanceforge.data.datasets import available_datasets  # noqa: E402
from balanceforge.pipeline.runner import (  # noqa: E402
    FLAGSHIP,
    METHODS,
    BalancePipeline,
    write_report,
)

SEEDS = (101, 202, 303)


def _table(rows: list[dict[str, object]], columns: list[str], width: int = 14) -> None:
    print("".join(f"{c:<{width}}" for c in columns))
    print("-" * (width * len(columns)))
    for r in rows:
        print("".join(f"{str(r.get(c, '')):<{width}}" for c in columns))


def main() -> int:
    t0 = time.perf_counter()
    cfg = BalanceForgeConfig(n_seeds=len(SEEDS), seed=42)
    set_all(cfg.seed)

    datasets = available_datasets()
    print("=" * 84)
    print("BalanceForge — geometry-routed rebalancing for imbalanced learning")
    print("author: 晨星 (CJX0712)")
    print("=" * 84)
    print()

    pipe = BalancePipeline(cfg)
    report = pipe.run_benchmark(datasets=datasets, methods=METHODS, seeds=SEEDS)

    print(
        f"primary metric : {report['primary_metric']} "
        f"(geometric mean of per-class recall, higher is better)"
    )
    print(f"datasets       : {', '.join(datasets)}")
    print(f"methods        : {len(METHODS)}")
    print(f"seeds          : {report['seeds']}")
    print(f"tier-0 backend : imbalanced-learn {report['backend']['imblearn_version']}")
    print()

    # ---------------------------------------------------------------- per-dataset
    from balanceforge.evaluation.metrics import aggregate_across_seeds

    buckets: dict[tuple[str, str], list[dict[str, float]]] = {}
    for row in report["rows"]:
        if row.get("skipped"):
            continue
        buckets.setdefault((row["method"], row["dataset"]), []).append(row["metrics"])

    print("-" * 84)
    print("Per-dataset G-mean (mean of 3 seeds)")
    print("-" * 84)
    header = ["method"] + list(datasets) + ["AGG"]
    print("".join(f"{h:<13}" for h in header))
    print("-" * (13 * len(header)))
    agg_rows = []
    for method in METHODS:
        vals = []
        for ds in datasets:
            rows = buckets.get((method, ds))
            vals.append(
                f"{aggregate_across_seeds(rows, 'gmean')['mean']:.4f}" if rows else "  n/a"
            )
        numeric = [
            aggregate_across_seeds(buckets[(method, ds)], "gmean")["mean"]
            for ds in datasets
            if (method, ds) in buckets
        ]
        agg = sum(numeric) / len(numeric) if numeric else float("nan")
        agg_rows.append((agg, method, vals, len(numeric)))
    agg_rows.sort(key=lambda r: -r[0] if r[0] == r[0] else 1e9)
    for agg, method, vals, n_ok in agg_rows:
        star = " <-- flagship" if method == FLAGSHIP else ""
        print(
            f"{method:<13}"
            + "".join(f"{v:<13}" for v in vals)
            + f"{agg:<13.4f}{'' if n_ok == len(datasets) else f'({n_ok}/{len(datasets)})'}{star}"
        )
    print()

    # ------------------------------------------------------------------ gate
    verdict = gate_verdict(report)
    print("-" * 84)
    print("Headline gate (per the delivery contract)")
    print("-" * 84)
    print(
        f"flagship                : {verdict['flagship']}  "
        f"mean={verdict.get('flagship_mean', float('nan')):.4f} "
        f"std={verdict.get('flagship_std', float('nan')):.4f}"
    )
    print(
        f"strongest single baseline: {verdict['strongest_baseline']}  "
        f"mean={verdict['strongest_baseline_mean']:.4f} "
        f"std={verdict['strongest_baseline_std']:.4f}"
    )
    print(f"compared datasets       : {', '.join(verdict.get('compared_datasets', []))}")
    print(
        f"delta                   : {verdict['delta']:+.4f}   "
        f"(threshold {GATE_THRESHOLD:+.4f}, half-sigma-sum {verdict['half_sigma_sum']:.4f})"
    )
    print(f"SIGNIFICANT             : {'PASS' if verdict['significant'] else 'FAIL'}")
    print()

    print("-" * 84)
    print("Per-baseline detail (shared datasets only)")
    print("-" * 84)
    _table(
        [
            {
                "baseline": k,
                "gmean": f"{v['mean']:.4f}",
                "std": f"{v['std']:.4f}",
                "delta": f"{verdict['flagship_mean'] - v['mean']:+.4f}",
            }
            for k, v in sorted(
                verdict.get("per_baseline", {}).items(), key=lambda kv: -kv[1]["mean"]
            )
        ],
        ["baseline", "gmean", "std", "delta"],
        width=18,
    )
    print()

    # ------------------------------------------------------------------ skips
    skips = [r for r in report["rows"] if r.get("skipped")]
    if skips:
        print("-" * 84)
        print(f"Skipped cells ({len(skips)}) — method preconditions not met")
        print("-" * 84)
        seen: set[tuple[str, str]] = set()
        for r in skips:
            key = (r["method"], r["dataset"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  {r['method']:<12} on {r['dataset']:<12} {r['skip_reason'][:70]}")
        print()

    # ------------------------------------------------------------- failures
    print("-" * 84)
    print("Failure cases (derived from the run, flagship only)")
    print("-" * 84)
    ff = [c for c in report["failure_cases"] if c["method"] == FLAGSHIP]
    for c in ff[:8]:
        print(
            f"  {c['dataset']:<12} {c['kind']:<16} idx={c['index']:<6} "
            f"x0={c['x0']:+.3f} x1={c['x1']:+.3f} true={c['true']} pred={c['pred']}"
        )
    print()

    out = write_report(report, REPO_ROOT / "benchmark.json")
    print("=" * 84)
    print(f"total elapsed : {time.perf_counter() - t0:.1f}s")
    print(f"report        : {out}")
    print("=" * 84)

    # Machine-readable copy for CI / downstream tooling.
    summary = {
        "gate": verdict,
        "threshold": GATE_THRESHOLD,
        "aggregate": {
            m: sum(
                aggregate_across_seeds(buckets[(m, ds)], "gmean")["mean"]
                for ds in datasets
                if (m, ds) in buckets
            )
            / max(1, sum(1 for ds in datasets if (m, ds) in buckets))
            for m in METHODS
        },
    }
    (REPO_ROOT / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
