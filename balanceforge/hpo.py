"""Parameter sweep harness for the MOSAIC flagship.

Not shipped as part of the public API — it exists so that every default in
:class:`~balanceforge.models.mosaic.MosaicConfig` is *measured*, not guessed.
Run with ``python -m balanceforge.hpo sweep``.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from balanceforge.core.interfaces import estimate_profile
from balanceforge.core.seed import derive_seed
from balanceforge.data.datasets import available_datasets, get_dataset
from balanceforge.data.splits import make_splits
from balanceforge.evaluation.metrics import gmean_score
from balanceforge.models.classifiers import PriorMatchedCalibrator, make_classifier
from balanceforge.models.mosaic import MosaicConfig, rebalance

SEEDS = (101, 202, 303)


def _eval_mosaic(dataset: str, cfg: MosaicConfig, seeds: tuple[int, ...]) -> float:
    scores = []
    for s in seeds:
        d = get_dataset(dataset, seed=derive_seed(s, "data", dataset))
        fit, val, test = make_splits(d, 0.3, 0.2, s)
        profile = estimate_profile(fit)
        res = rebalance(fit, profile, cfg, seed=s)
        clf = make_classifier("rf", s, 2).fit(res.data.X, res.data.y)
        val_p = clf.predict_proba(val.X)
        test_p = clf.predict_proba(test.X)
        cal = PriorMatchedCalibrator(seed=s).fit(val, val_p)
        counts = np.array([int((val.y == c).sum()) for c in [0, 1]], dtype=np.float64)
        scores.append(gmean_score(test.y, cal.predict(test_p, counts)))
    return float(np.mean(scores))


def sweep() -> dict[str, Any]:
    """Grid-search the flagship's knobs on the aggregate mean G-mean."""
    grid = {
        "sigma_c": [0.35, 0.7, 1.2, 2.0],
        "k_neighbors": [3, 5, 8],
        "condense_keep": [0.4, 0.6, 0.8],
    }
    datasets = available_datasets()
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for sigma, k, keep in itertools.product(
        grid["sigma_c"], grid["k_neighbors"], grid["condense_keep"]
    ):
        cfg = MosaicConfig(sigma_c=sigma, k_neighbors=k, condense_keep=keep)
        per_ds = {ds: _eval_mosaic(ds, cfg, SEEDS) for ds in datasets}
        agg = float(np.mean(list(per_ds.values())))
        rows.append(
            {
                "sigma_c": sigma,
                "k_neighbors": k,
                "condense_keep": keep,
                "aggregate": round(agg, 5),
                **{ds: round(v, 5) for ds, v in per_ds.items()},
            }
        )
    rows.sort(key=lambda r: -r["aggregate"])
    return {
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "seeds": list(SEEDS),
        "best": rows[0],
        "top10": rows[:10],
        "all": rows,
    }


if __name__ == "__main__":  # pragma: no cover
    out = sweep()
    Path("sweep.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(out["top10"], indent=2))
