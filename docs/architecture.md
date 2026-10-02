# BalanceForge — Architecture

Author: 晨星 (CJX0712)

---

## 1. Problem statement

Class imbalance is not one problem. It is **four different geometric situations**
that happen to share a symptom (low minority recall), and the standard toolkit
offers one blunt instrument per situation. That mismatch is the reason
"just use SMOTE" is unreliable advice:

| regime | what it looks like | what actually helps | what actively hurts |
|---|---|---|---|
| `separable` | minority is spatially detached | almost nothing (naive already at ceiling) | any resampling — pure variance injection |
| `overlap` | minority sits inside the majority hull | cleaning + border-only synthesis | global SMOTE — it interpolates straight through the boundary |
| `multimodal` | `k` well-separated minority islands | **per-mode** equal-budget expansion | global SMOTE — the largest mode eats the whole budget |
| `sparse` | very few minority samples, large majority | aggressive expansion + majority condensation | under-sampling — it discards the very evidence you need |

BalanceForge's contribution is (a) a **cheap geometry classifier** that tells
these four apart from training data alone, (b) a **contamination-gated
sampler** that refuses to synthesise across class boundaries, and (c) a
**zero-hyper-parameter prior correction** that fixes the decision threshold.

---

## 2. Module map (strictly acyclic)

```
cli ──▶ pipeline ──┬──▶ data      (datasets, splits)
                   ├──▶ policy    (router: geometry → regime plan)
                   ├──▶ models    (mosaic, classifiers/calibration)
                   ├──▶ sampling  (tier0 imblearn, tier1 numpy)
                   └──▶ evaluation(metrics, significance gate)
                          │
                          ▼
                        core      (types, errors, config, interfaces, seed)
```

`cli → pipeline → {data, policy, models, sampling, evaluation} → core`.
No module imports upward. The pipeline is the only component that knows about
all the others, which is what makes the benchmark fair: every method travels
the *same* fit / calibrate / predict path and only the rebalancing step differs.

```
balanceforge/
  core/
    types.py         Dataset · SplitData · ImbalanceProfile · MethodResult · Protocols
    errors.py        E100 input · E200 sampling · E300 model · E400 eval · E500 pipeline
    config.py        dataclass + BALANCEFORGE_* env overrides + validation
    interfaces.py    geometry profiling: MST cluster count, k-NN entropy, overlap
    seed.py          set_all() — the single determinism entry point
  data/
    datasets.py      5 generators, one per geometry, each with a difficulty dial
    splits.py        stratified holdout / validation, no-leakage by construction
  policy/
    router.py        profile → RegimePlan, incl. the M1 negative route
  models/
    mosaic.py        the flagship: gate + per-mode expand + condensation
    classifiers.py   estimator factory, prior correction, threshold calibration
  sampling/
    tier0.py         imbalanced-learn wrappers (SMOTE/ADASYN/ENN/EE/BRF/…)
    tier1.py         pure-numpy offline fallbacks for every Tier-0 name
  evaluation/
    metrics.py       G-mean, macro-F1, MCC, AUPRC + the ½σ significance gate
  pipeline/
    runner.py        BalancePipeline.run_cell / run_benchmark
  hpo.py             grid search used to measure every shipped default
  cli.py             benchmark · profile · rebalance · sweep
```

---

## 3. Stage 1 — geometry profiling

`balanceforge/core/interfaces.py` estimates an `ImbalanceProfile` from a
**training split only** (the holdout is never touched here).

### 3.1 Minority mode count via MST plateau width

Build a minimum spanning tree over the minority subset (Prim, `O(n²)`, pure
numpy). Then sweep a threshold `t` geometrically over the edge spectrum and
record how many components the tree has at each `t`.

* One mode → the sequence is `1, 1, 1, …, 1` with at most a razor-thin spike.
* `k` islands → the sequence is `1 → k (wide) → 1`, so the **penultimate
  plateau** has height `k`.

The estimate is the penultimate plateau's height, provided that plateau spans
at least `MST_PLATEAU_FRAC = 12%` of the log range.

**Two rejected formulations, both measured** (documented because the naive
versions look right):

| formulation | isotropic Gaussian blob (200×4) | verdict |
|---|---|---|
| count edges above `K × median` | returns 2–3 | Gaussian MST tails alone trip it |
| largest relative jump in the sorted spectrum | `k=3` islands → 1 | equidistant islands give *identical* inter-mode edges, so no value cut separates them |

The plateau criterion passes both: **15/15** blob configurations return 1, and
2/3/4 islands return 2/3/4 exactly.

### 3.2 Other statistics

* `overlap_ratio` — fraction of minority points whose 1-NN is a majority point.
* `minority_noise_fraction` — fraction of minority points whose own 1-NN is also
  minority (label-noise proxy).
* `knn_entropy` — mean neighbourhood label entropy, normalised to `[0, 1]`.
* `imbalance_ratio`, `minority_fraction`.

### 3.3 Routing decision

Decision order is most-specific-first:

```
IR >= 25          -> sparse
modes >= 2        -> multimodal
overlap >= 0.30   -> overlap
otherwise         -> separable
```

### 3.4 The M1 negative route

If the regime is `separable` **and** the measured validation minority F1 is
already at ceiling (`>= 0.95`), the router **forbids resampling** and returns
`none`.

The justification is a theorem rather than a heuristic. For a linearly
separable pair, the bias-variance decomposition term for every majority point
is `b*_BA ≡ 0` — there is no misclassification cost to trade, so
`B_BA = B_BD` identically. The sampling-variance term is *provably* zero, not
merely small, so resampling can only add noise.

---

## 4. Stage 2 — contamination-gated SMOTE (the core contribution)

### 4.1 The problem it solves

SMOTE synthesises `s = (1-λ)x_i + λx_j` between k-NN minority neighbours. When
the minority has no genuine local structure — a single diffuse mode inside the
majority hull — a large fraction of those segments pass through majority
territory. The synthetic points then teach the classifier that majority space
is minority space. This is the mechanism behind the well-documented "SMOTE
hurts on overlapping data" result, and no mainstream sampler guards against it.

### 4.2 The gate

Let `M` be the majority set and let

```
τ  = median 1-NN distance among majority samples        (the local majority radius)
```

For a candidate segment `AB` define its contamination as a **proximity excess**:

```
contam(A, B) = clip(1 - min_{m ∈ M} dist(m, segment AB) / τ,  0, 1)
```

`0` means the nearest majority point is a full local radius away (clean);
`1` means the segment is closer than that (contaminated). The segment is then
admitted with probability

```
p_AB = exp( - contam(A, B)² / (2 σ_c²) )
```

and synthesis draws segments with probability proportional to `p_AB`.

**Two earlier formulations were measured and rejected** — both are documented in
the source because they look correct on paper:

| formulation | p50 / p90 across the 5 benchmark datasets | verdict |
|---|---|---|
| `hits / |M|` | 0.0000 / 0.0000 | everything is numerically ~0 |
| `hits / (|M| / n_min)` | 0.00 / 0.00 (max 0.37) | still no separation |

The separating signal is the **distance**, not the count. Measured p50
contamination: `multimodal 0.00`, `separable 0.00`, `overlap 0.00`, `sparse
0.22`, `manifold 0.47` — monotone in the actual geometric overlap.

### 4.3 When MOSAIC is exactly SMOTE

When every segment has zero contamination, `p_AB = 1` for all `AB` and the
gate is a no-op: MOSAIC degenerates to SMOTE with the same `k` (INV-17). It is
a strict generalisation, not a replacement.

### 4.4 Per-mode expansion

For `multimodal`, a global SMOTE budget is captured by whichever mode has the
most samples. MOSAIC allocates `n_synth / k_modes` per mode via internal
KMeans, so every island ends with equal support.

### 4.5 Majority condensation

For `sparse`, k-means centroid compression of the majority class to
`condense_keep × |M|`. This runs **before** the synthesis budget is computed,
so the budget is measured against the post-condensation majority — otherwise the
class ratio is never actually repaired.

### 4.6 Complexity

`O(n_min² )` for the Prim sweep, `O(n_min · k · n_maj)` for the contamination
scoring (vectorised per segment), and `O(n_synth · d)` for the synthesis
itself. `n_synth` is capped so the end-to-end demo stays inside the 60 s
budget.

---

## 5. Stage 3 — prior-matched threshold calibration

### 5.1 The closed-form part (zero hyper-parameters)

For a class-conditional decision the Bayes rule is

```
ŷ = argmax_c  P(y=c | x) / π_c  =  argmax_c  p(x | y=c)
```

so the logit must be de-biased by the training class prior:
`s_c = log q_c − log π_c`. This alone beat most SMOTE configurations in our
measurements (e.g. on the original `overlap` setting, `class_weight` went from
G-mean 0.000 to 0.708 with this correction and no other change).

### 5.2 Three guards, each added after a measured failure

1. **Scale standardisation.** A per-column z-score over the log-odds.
   Without it the correction is silently wrong for models whose probability
   scale differs wildly.

2. **Anti-symmetry detection.** `EasyEnsembleClassifier` emits *exactly*
   anti-symmetric logits — measured column means `+0.456 / −0.456` with
   identical stds, so after standardisation column 1 is literally `−column 0`.
   Adding `log(π_1) ≈ +2.44` then makes the minority column win on **every**
   row: 396/396 predictions collapsed to the minority and G-mean went to
   **0.000**. Such a model has already performed the balancing internally, so
   the prior term must be skipped. `detect_already_balanced()` requires two
   independent signals to agree (anti-symmetric logits **and** an argmax
   minority rate far above the prior).

3. **Magnitude clamp.** `|log π|` is clipped to `MAX_PRIOR_SHIFT = 2.0` nats, so
   no single class can be promoted by more than that regardless of residual
   scale mismatch.

### 5.3 The learned part

A coordinate ascent over a fixed offset grid `(-1, -0.5, -0.25, 0, 0.25, 0.5, 1)`
on the **validation** split, maximising the validation G-mean (the primary
metric — the objective is aligned with evaluation by construction). One pass
per class per iteration, 3 iterations, early exit when no improvement.

The all-zero offset is always evaluated first, so the calibrated model is never
worse than the raw one *on the tuning metric* (INV-19).

### 5.4 Sample-size admissibility gate

If any class has fewer than `MIN_CLASS_SAMPLES_FOR_CALIBRATION = 25` validation
samples, the learned stage is **skipped entirely** and only the closed-form
prior correction is applied.

Measured justification: the `sparse` regime at `imratio=120` left **4** minority
validation samples. The coordinate ascent pushed the minority offset to its
grid maximum, which sent 924 of 1209 test rows to the minority and *lowered*
G-mean from SMOTE's 0.511 to 0.169. With too few samples the honest answer is
"do not calibrate".

### 5.5 Wilson floor on the objective

Per-class recall is floored at its Wilson score lower bound. With `n = 40` the
raw recall has a standard error of ≈ 0.072, so optimising a threshold directly
against it is optimising noise. The floor makes the effective gradient vanish
exactly when a class is too small to justify a shift.

---

## 6. Evaluation contract

* **Primary metric**: `gmean` — geometric mean of per-class recall (Kubat &
  Holte). Maximised. Guarantees: `gmean = 0` if any class recall is 0;
  `gmean = 1` for a perfect classifier; `gmean <= balanced_accuracy` always.
* **Secondary**: `macro_f1`, `balanced_accuracy`, `mcc`, `minority_recall`,
  `minority_precision`, `au_prc` (binary only), and the per-class recall vector.
* **`predict_proba` column order is a hard contract** (INV-18): columns must
  follow `np.unique(y_train)`. This is not hypothetical —
  `EasyEnsembleClassifier` reports a different `classes_` order on the
  benchmark data, which silently inverted every imbalance metric until
  `_aligned_proba` was added.

### 6.1 The significance gate

A win requires **all three**:

1. `mean_method − mean_base > threshold` (declared effect size, `0.02` for
   G-mean);
2. `mean_method − mean_base > ½(σ_method + σ_base)`;
3. both σ finite.

The comparison is restricted to the datasets where **both** methods actually
ran. Without that restriction, a baseline that skips half the suite (ADASYN is
undefined on separable data — it raises *"Not any neigbours belong to the
majority class"* by design) would be scored on whichever datasets flatter it.

---

## 7. Leakage control

Three disjoint splits, carved in this order:

```
full ──▶ holdout (untouched)  ──▶ fit ──▶ validation
```

* The holdout is carved **first** and is never passed to any `fit`.
* The validation split is the only data the calibrator and the router may see.
* Splits are stratified, so the class ratio is preserved to within one sample
  per class (INV-9).
* `estimate_profile` and `rebalance` receive the **fit** split only.

---

## 8. Offline fallback (Tier-1)

Every Tier-0 name has a pure numpy/scikit-learn twin in
`balanceforge/sampling/tier1.py`: ROS, SMOTE (k=5 and k=10), RUS,
ClusterCentroids, Tomek links, ENN.

Bit-identity with `imbalanced-learn` is **not** the invariant — the two consume
the RNG stream in a different order, so the draws differ. The invariant that
*is* enforced is geometric:

> **INV-13**: every synthesised point is a convex combination
> `(1-λ)·x_i + λ·x_j` of two genuine k-NN minority samples.

The test suite verifies this by searching all `(i, j)` neighbour pairs for a
reconstruction within `1e-9`. Additionally:

* **INV-14**: synthesised points stay inside the minority convex hull.
* **INV-15**: `numpy_smote` is idempotent once the target ratio is met.

A cell whose Tier-0 sampler raises (e.g. ADASYN on separable data) is recorded
as a **legitimate skip** with its reason, not silently dropped and not a crash.

---

## 9. Reproducibility

`core.seed.set_all(seed)` is the single entry point. Every sampler, splitter and
estimator derives its own child seed via `derive_seed(base, *tags)`, an FNV-1a
hash — deterministic across processes, unlike Python's salted `hash()`.

`n_jobs` is pinned to 1 everywhere: the sandbox's joblib IPC is unreliable on
Windows and parallel workers would break the determinism guarantee.

Two consecutive runs of `examples/run_demo.py` with the same seed produce
bit-identical metric values; only wall-clock timings differ.

---

## 10. Measured defaults

Every shipped default was swept on the **final** dataset set (`scripts/verify.py`
plus the grid in `balanceforge/hpo.py`, 3 seeds × 5 datasets). The values below
are the ones that measurement supports; where an earlier sweep disagreed, the
disagreement is recorded rather than quietly dropped.

| parameter | swept values | chosen | basis |
|---|---|---|---|
| `sigma_c` | 0.3 / 0.5 / 0.8 / 1.2 / 2.0 | see note | **re-swept**; the first sweep ran on a superseded dataset set |
| `k_neighbors` | 3 / 5 / 8 | 5 | aggregate G-mean, ties at 3 seeds |
| `condense_keep` | 0.4 / 0.6 / 0.8 / 1.0 | **1.0** (no condensation) | measured on the sparse regime alone: 1.0 → 0.388, 0.8 → 0.359, 0.6 → 0.345 |
| `sparse` `imratio` | 400 / 120 / 20 | **20** | 400 → 3 test minority samples; 120 → 4 validation samples |
| `sparse` `shift` | 0.6 / 2.2 | **2.2** | at 0.6 the train/holdout G-mean gap was **+1.000** (no learnable signal) |
| `overlap` `imratio` | 20 / 8 | **8** | 20 left 18 test minority samples (cross-method std 0.28 = noise) |
| `overlap` `sep` | 0.35 / 0.9 / 1.2 | **0.9** | 0.35 gave gap +0.65 (memorisation); 1.2 saturates every method at 0.985 |
| `manifold` `noise` | 0.10 / 0.05 | **0.05** | 0.10 gave gap +0.58 |
| `SPARSE_IR` gate | 100 / 25 / 18 | **18** | must fire on `sparse` (IR 20) and not on `multimodal` (IR 15) |
| `MIN_CLASS_SAMPLES_FOR_CALIBRATION` | 25 / 15 / 10 | **10** | 25 disabled calibration on 4/5 datasets; 4 samples is the catastrophic case |

### 10.1 The first sweep was run on the wrong data — and that mattered

`sigma_c` was originally set to `2.0` from a 36-point grid search. That search
ran against the **first** dataset set, whose `sparse` and `overlap` generators
had train/holdout G-mean gaps of `+1.000` and `+0.654` — i.e. they were
measuring memorisation, not imbalance handling. Once those generators were
fixed, the `sigma_c = 2.0` choice turned out to make the contamination gate
**inert**: with `σ_c = 2.0` the rejection count is `0 / 0 / 0` on
overlap / sparse / manifold, and the ablation delta is `+0.0003` — the gate is
mathematically present and behaviourally absent.

At `σ_c = 0.5` the gate rejects 14 / 1 / 22 segments on the same three
datasets, and the accepted-segment mean contamination drops from 0.133 to
0.109. That is the difference between a component that exists and a component
that does something, so `σ_c` is re-swept on the corrected data rather than
carried over.

**The general lesson, recorded because it applies to every entry in this
table**: a hyper-parameter sweep is only as valid as the data it ran on. Two
of the "measured defaults" recorded before the DGP fix were artefacts of
broken generators, and they would have shipped silently.

---

## 11. Known limitations

* **Binary only.** The metric, gate and routing all assume two classes. The
  estimator factory and the data contract generalise to `n` classes, but the
  contamination gate's `|M|/n_min` normalisation is written for the binary case.
* **`multimodal` is detected, not exploited.** Per-mode expansion is
  implemented and active, but with `k=3` islands a global budget already covers
  each mode, so the measured gain over plain gated-SMOTE is inside seed noise.
* **The plateau-width cluster detector is `O(n_min²)`.** Fine at the shipped
  sizes (≤ 1000 minority samples); it would need a k-NN graph approximation for
  the extreme-IR regime beyond `imratio ≈ 1000`.
* **MCC vs G-mean can disagree.** Under heavy under-sampling, `cc`/`rus` can
  post a high G-mean with a poor MCC, because discarding majority evidence
  inflates minority recall at the cost of precision. Both metrics are reported
  for exactly this reason; G-mean alone is not a sufficient summary.
