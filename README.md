<div align="center">

# BalanceForge

**Geometry-routed rebalancing for imbalanced and long-tailed learning**

[![CI](https://github.com/CJX0712/balanceforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/balanceforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/Release-v0.1.0-2ea44f)](https://github.com/CJX0712/balanceforge/releases)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776ab)](https://www.python.org)
[![Quality](https://img.shields.io/badge/Quality-C%20%28user-authorised%20publish%29-informational?style=flat-square)](BLOCKED.md)

> **⚠️ Quality grade: C — the flagship does not win.**
> MOSAIC reaches aggregate G-mean **0.8751 ± 0.0866** against the strongest
> single baseline `BalancedRandomForestClassifier` at **0.9027 ± 0.0745** — a
> delta of **−0.0275**, short of the +0.0200 threshold in the delivery
> contract §5. That is a **C grade**, and this repository was published at the
> user's explicit instruction rather than under the default rule (which would
> block it). The measured gap is reported, not smoothed over.
>
> Everything else passes: **55 tests**, **240 benchmark rows bit-identical
> across processes** plus 186 identical failure cases, offline fallback,
> full docs, CI on 3 OS × 2 Python versions.
> See **[`BLOCKED.md`](BLOCKED.md)** for the DoD table, the six bugs found and
> fixed, the three benchmark pathologies corrected, and the recommended next
> steps.

</div>

---

## The problem with "just use SMOTE"

Class imbalance is **not one problem**. It is four different geometric
situations that happen to share a symptom, and the standard toolkit offers one
blunt instrument per situation:

| regime | what it looks like | what actually helps | what actively hurts |
|---|---|---|---|
| `separable` | minority spatially detached | almost nothing — naive is already at ceiling | **any** resampling (pure variance injection) |
| `overlap` | minority inside the majority hull | cleaning + border-only synthesis | global SMOTE (interpolates through the boundary) |
| `multimodal` | `k` detached minority islands | **per-mode** equal-budget expansion | global SMote (largest mode eats the budget) |
| `sparse` | very few minority samples | aggressive expansion | under-sampling (discards the needed evidence) |

## MOSAIC

Three stages, each independently switchable:

**1. Geometry routing** — an MST plateau-width cluster detector plus overlap,
entropy and IR statistics classify the dataset, then pick a matching remedy.
Includes a **negative route**: on `separable` data at ceiling, resampling is
forbidden outright (justified by the fact that `b*_BA ≡ 0` for every majority
point in the linearly separable case, so the sampling-variance term is provably
zero, not merely small).

**2. Contamination-gated SMOTE** — each candidate k-NN synthesis segment is
scored by its *proximity excess* to the majority set and admitted with
probability `exp(−c²/2σ_c²)`. This is the core contribution: plain SMOTE
synthesises straight through class boundaries, and no mainstream sampler guards
against it. MOSAIC degenerates to plain SMOTE when nothing is contaminated.

**3. Prior-matched calibration** — a **zero-hyper-parameter** closed-form prior
correction `s_c = log q_c − log π_c`, optionally followed by a validation-only
coordinate ascent with a Wilson floor on the objective.

## Install

```bash
pip install -e ".[test]"
```

or

```bash
git clone https://github.com/CJX0712/balanceforge && cd balanceforge
make install
```

Pure CPU. No GPU, no network at run time.

## Quick start

```bash
python examples/run_demo.py              # full benchmark, writes benchmark.json
python -m balanceforge.cli profile       # geometry profile of every dataset
python -m balanceforge.cli rebalance --dataset sparse
make test                                # unit tests
```

Library use:

```python
from balanceforge import BalancePipeline, BalanceForgeConfig
from balanceforge.core.seed import set_all

set_all(42)
pipe = BalancePipeline(BalanceForgeConfig(n_seeds=3))
result, failure_cases = pipe.run_cell("overlap", "mosaic", seed=101)
print(result.metrics["gmean"], result.route)
```

## Measured results

3 seeds × 5 geometries, G-mean (geometric mean of per-class recall), from
[`benchmark.json`](benchmark.json):

```
method        separable     overlap       multimodal    sparse        manifold      AGG
--------------------------------------------------------------------------------------------------
balanced_rf   0.9906        0.8315        0.9780        0.8066        0.9067        0.9027
smoteenn      0.9618        0.8374        0.9678        0.7853        0.9106        0.8926
cc            0.9649        0.8219        0.9684        0.8314        0.8437        0.8861
rus           0.9766        0.8147        0.9784        0.8000        0.8568        0.8853
mosaic        0.9619        0.8158        0.9603        0.7386        0.8992        0.8751  <-- flagship
smote         0.9568        0.8132        0.9696        0.7022        0.9103        0.8704
adasyn        0.9568        0.8070        n/a           0.7124        0.9185        0.8487
ros           0.9569        0.6554        0.9756        0.5019        0.8574        0.7894
easy_ensemble 0.9907        0.4522        0.9467        0.7877        0.4341        0.7223
class_weight  0.5840        0.3644        0.8513        0.5619        0.7212        0.6166
none          0.5711        0.4522        0.8317        0.5940        0.5412        0.5980
```

The flagship ranks **5th of 16**, 0.0275 behind `balanced_rf` — statistically
indistinguishable (½σ sum = 0.0805) but directionally behind, so it does not
clear the +0.0200 bar. `adasyn` shows `n/a` on `multimodal` because it raises
by design when no minority neighbour belongs to the majority class; that is
recorded as a legitimate skip, not a crash.

## What this repository is honest about

* **The flagship does not win.** Aggregate G-mean 0.8751 vs 0.9027 for
  `BalancedRandomForest`. The contamination gate's ablation delta is
  **+0.0069** — the gate is mathematically present and behaviourally almost
  inert, because most of these geometries have no majority intrusion into the
  minority's interpolation paths. The gate can only *reject* synthesis, never
  *create* it. See [`BLOCKED.md`](BLOCKED.md) §4.
* **Six real bugs were found and fixed during the build**, each now covered by
  a test. The subtlest: a per-column z-score on two-class logits *negates* them
  (the columns are exact negatives of each other), so "standardising the scale"
  silently reversed every prediction. The most expensive: applying the prior
  correction to an already-rebalanced model cost **−0.359 G-mean**. Both are
  documented in `docs/architecture.md` because both look correct on paper.
* **Three benchmark pathologies were corrected**, and the first hyper-parameter
  sweep had to be thrown out and redone because it ran on the broken data. A
  sweep is only as valid as the data it ran on — see
  `docs/architecture.md` §10.1.
* **G-mean alone is not a sufficient summary.** Heavy under-sampling can post a
  high G-mean with a poor MCC, because discarding majority evidence inflates
  minority recall at the cost of precision. MCC, macro-F1 and per-class recalls
  are all reported for exactly that reason.
* **Bit-identity with imbalanced-learn is not the invariant.** The two consume
  the RNG stream in different order, so the draws differ. The invariant that is
  enforced is geometric: every synthesised point is a convex combination
  `(1−λ)x_i + λx_j` of two genuine k-NN minority samples, verified by search.

## Architecture

```
cli → pipeline → {data, policy, models, sampling, evaluation} → core
```

Strictly acyclic, single responsibility per module. Full map, every invariant
(`INV-1` … `INV-21`), and the measured default for each shipped parameter are
in [`docs/architecture.md`](docs/architecture.md).

## Reproducing every number

```bash
python examples/run_demo.py
```

Two runs with the same seed produce bit-identical metric values; only wall-clock
timings differ.

## License

MIT — see [`LICENSE`](LICENSE). All runtime dependencies (numpy, scipy,
scikit-learn, imbalanced-learn, optuna) are permissively licensed and MIT
compatible. No copyleft dependency is used.
