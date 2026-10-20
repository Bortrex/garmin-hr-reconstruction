# Temporal-window gradient boosting for HR residuals

**Recommendation: B. Promote the window-enhanced model as the next version, subject to user approval.** The established version-1 reconstruction and documentation remain unchanged. This is a single fixed many-to-one feature experiment, not a new sequence model or an anchor generator.

## Controlled design

Reuse `evaluate_hr_ml_residual.py` for FIT preparation, non-HR preprocessing, anchor simulation, interpolation, five-sample residual averaging, `4f(1−f)` anchor taper and all error/peak/roughness metrics. Two outdoor roads are the independent leave-one-ride-out units; phases are dependent simulations. One scalar HR residual is predicted at each supported time; add it to the same linear baseline.

Version-1 fitted estimators and held-out curves were not saved, so the minimum controlled refits train ML-B (20 inputs) and Window (50 inputs) on identical rows for each fold and the pooled-road MTB check. All 26 recomputed linear/ML-B rows match the saved benchmark's numeric metrics and coverage within 1e−10 tolerance. No baseline retuning or ML-A/virtual experiment is performed.

Runtime: Python 3.8.8, scikit-learn 1.1.3, NumPy 1.20.1, pandas 1.2.4, SciPy 1.7.3, Matplotlib 3.3.4. No new dependency.

Unchanged estimator settings:

```text
{'learning_rate': 0.05, 'max_iter': 120, 'max_leaf_nodes': 7, 'max_depth': 3, 'min_samples_leaf': 100, 'l2_regularization': 10.0, 'early_stopping': False, 'random_state': 265}
```

No scaling, parameter/window search, automatic early-stopping split, or held-out tuning. Train on point observations only; evaluate point plus complete 60-second centered-average sensitivity without refitting. Training retains one timestamp every five elapsed seconds, cyclically assigned to one of phases 0/20/40s, excluding observations and unsupported baseline points. No duplicated targets across phases.

| Training partition | Training examples | Benchmark inputs | Window inputs |
|---|---:|---:|---:|
| road_ride_2 | 2131 | 20 | 50 |
| road_ride_1 | 2509 | 20 | 50 |
| road_ride_1 + road_ride_2 | 4640 | 20 | 50 |

## Exact temporal features

Keep all 20 existing anchor-context, movement/stop/restart and terrain inputs. Add 30 interpretable inputs:

- Four **half-open 30-second regions** relative to t: `[-60,-30)`, `[-30,0)`, `[0,30)`, `[30,60)`. At 1Hz these include offsets −60…−31, −30…−1, 0…29, and 30…59 respectively; t+60 is excluded.
- Seven outputs per region: mean audited `speed5` (m/s), movement fraction (`speed_raw>0.3m/s`), stationary-evidence fraction, mean positive/negative quality-masked `climb30` magnitudes (m/s), record coverage and terrain coverage (fractions of 30 expected seconds). This gives 28 features.
- Two transitions: future-near minus past-near mean speed, and the same contrast in signed mean climbing rate (`up−down`). Units are m/s; these are differences of regional means, not acceleration estimates.

The existing five-sample speed and centered terrain preprocessing retain their own numeric support; the nominal regions describe where these audited summaries are sampled, not a strict limit on their underlying raw support. Future **non-HR** observations are available for offline reconstruction. No full/reference-HR value, HR rolling statistic, activity identity, power, cadence, absolute position or HR-derived load enters a window.

### Validity and gaps

Query actual timezone-aware UTC timestamps inside the target's continuous record block. A region on the far side of a gap is unsupported even if it contains 30 later observations. Require all 30 recorded/finite values for each mean/fraction; require all 30 quality-supported terrain rates for up/down means. Partial regions remain NaN with coverage retained; no filling or averaging across an invalid altitude interval. HGB handles missing inputs natively, so no prediction is dropped solely because windows are incomplete.

Stops do not invalidate speed/movement summaries: a full stopped region has stationary fraction 1, movement fraction 0, and typically zero speed. Its unsupported terrain stays NaN, never a fabricated flat/climbing measurement. Existing stop/restart ages remain in the feature set; no HR state is reset at stops.

## Outdoor holdout results

BPM; equal-weight phase means [minimum, maximum]. Bias equals average-HR error on matching timestamps. All three models and both observation conditions share the same hidden evaluation timestamps, excluding anchor times, and retain exactly the benchmark coverage.

| Held-out ride | Model | MAE | RMSE | Bias | High-HR MAE | Coverage % |
|---|---|---:|---:|---:|---:|---:|
| road_ride_1 | linear | 4.13 [4.09, 4.17] | 6.20 [6.10, 6.36] | -0.10 | 1.92 | 84.94 [84.27, 85.60] |
| road_ride_1 | ML-B | 3.77 [3.76, 3.78] | 5.52 [5.37, 5.62] | 0.28 | 1.72 | 84.94 [84.27, 85.60] |
| road_ride_1 | Window | 3.50 [3.44, 3.55] | 5.14 [4.90, 5.30] | 0.23 | 1.62 | 84.94 [84.27, 85.60] |
| road_ride_2 | linear | 3.92 [3.89, 3.96] | 5.99 [5.96, 6.04] | 0.22 | 1.44 | 82.42 [81.40, 83.31] |
| road_ride_2 | ML-B | 3.64 [3.59, 3.74] | 5.37 [5.26, 5.45] | 0.18 | 1.32 | 82.42 [81.40, 83.31] |
| road_ride_2 | Window | 3.34 [3.30, 3.37] | 4.97 [4.90, 5.00] | -0.03 | 1.35 | 82.42 [81.40, 83.31] |

| Held-out ride | Phase s | Linear MAE/RMSE | ML-B MAE/RMSE | Window MAE/RMSE |
|---|---:|---:|---:|---:|
| road_ride_1 | 0 | 4.12/6.15 | 3.76/5.37 | 3.44/4.90 |
| road_ride_1 | 20 | 4.17/6.36 | 3.76/5.62 | 3.50/5.21 |
| road_ride_1 | 40 | 4.09/6.10 | 3.78/5.57 | 3.55/5.30 |
| road_ride_2 | 0 | 3.89/6.04 | 3.59/5.26 | 3.30/4.90 |
| road_ride_2 | 20 | 3.90/5.97 | 3.59/5.39 | 3.34/5.00 |
| road_ride_2 | 40 | 3.96/5.96 | 3.74/5.45 | 3.37/5.00 |

### Dynamics and stability

Use the existing score-only definitions: high HR = reference upper decile; rise/recovery = ≥+5/≤−5 BPM over 30 observed consecutive seconds. Peaks have prominence≥10 BPM, separation≥120s; fully scored ±15s windows compare local maxima and earliest maximum timing (bounded/censored at ±15s). Unmatched predicted peaks have no reference peak within ±30s. These diagnostic labels never enter features. Roughness includes anchor neighborhoods and requires adjacent supported seconds, not differences across recording gaps.

| Ride | Model | Rise MAE | Recovery MAE | Peak amplitude abs error | Peak timing abs s | Unmatched peaks | P99 / max 1s jump |
|---|---|---:|---:|---:|---:|---:|---|
| road_ride_1 | linear | 5.08 | 5.11 | 9.12 | 15.00 | 8.3 | 0.689 / 0.883 |
| road_ride_1 | ML-B | 4.41 | 4.82 | 7.10 | 10.24 | 7.0 | 0.947 / 2.292 |
| road_ride_1 | Window | 4.03 | 4.47 | 5.87 | 8.85 | 6.3 | 1.130 / 2.955 |
| road_ride_2 | linear | 4.48 | 5.48 | 8.09 | 15.00 | 5.3 | 0.728 / 0.967 |
| road_ride_2 | ML-B | 4.15 | 4.74 | 6.73 | 10.31 | 3.7 | 0.983 / 2.468 |
| road_ride_2 | Window | 3.80 | 4.44 | 6.02 | 8.30 | 5.0 | 1.093 / 2.217 |

### Averaged-observation sensitivity

Complete [t−30,t+29] 60-second means are pseudo-point constraints for this sensitivity only. They are not instantaneous HR or verified Garmin wellness behavior; the point-trained models are frozen.

| Ride | Model | MAE | RMSE |
|---|---|---:|---:|
| road_ride_1 | linear | 4.13 [4.05, 4.22] | 5.70 [5.58, 5.80] |
| road_ride_1 | ML-B | 3.78 [3.72, 3.83] | 5.13 [5.06, 5.19] |
| road_ride_1 | Window | 3.42 [3.37, 3.45] | 4.64 [4.58, 4.75] |
| road_ride_2 | linear | 3.89 [3.81, 3.95] | 5.53 [5.40, 5.64] |
| road_ride_2 | ML-B | 3.65 [3.60, 3.69] | 5.03 [4.97, 5.09] |
| road_ride_2 | Window | 3.28 [3.24, 3.31] | 4.54 [4.48, 4.58] |

## Actual MTB overlap

The pooled-road models use actual wellness summaries and quality-masked Strava terrain. Evaluate only where original Edge HR exists; never initialize with hidden Edge HR or add the later final-handoff boundary anchor to this diagnostic. Future non-HR activity can extend beyond the Edge overlap. Watch/Edge path/averaging differences confound the absolute errors; this previously examined overlap is not a pristine independent test.

| Model | n / coverage % | MAE | RMSE | Bias | High-HR | Recovery | Peak amplitude / timing s | Unmatched peaks | P99 / max 1s jump |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| linear | 1420 / 93.11 | 12.98 | 17.39 | -1.08 | 25.36 | 14.64 | 17.61 / 15.00 | 5 | 0.633 / 0.633 |
| ML-B | 1420 / 93.11 | 12.57 | 16.68 | -1.01 | 22.38 | 14.32 | 15.62 / 12.20 | 5 | 0.900 / 1.307 |
| Window | 1420 / 93.11 | 12.50 | 16.67 | -0.89 | 21.99 | 14.27 | 14.14 / 11.80 | 5 | 0.932 / 1.580 |

### Stationary and restart context

Stationary evidence includes inferred short pauses; restart is the first 30s after a supported restart. Road stationary counts are tiny; a missing (nan) error is undefined, not perfect. Terrain cannot identify a ravito.

| Activity | Model | Stationary n / MAE | Restart n / MAE | Uphill / downhill MAE |
|---|---|---|---|---|
| road_ride_1 | ML-B | 1–2 / 5.66 | 32–41 / 4.20 | 3.57 / 4.90 |
| road_ride_1 | Window | 1–2 / 1.52 | 32–41 / 1.34 | 3.32 / 4.60 |
| road_ride_2 | ML-B | 1–1 / 9.19 | 29–30 / 8.09 | 3.00 / 4.77 |
| road_ride_2 | Window | 1–1 / 8.91 | 29–30 / 8.13 | 2.60 / 4.56 |
| mtb_edge | ML-B | 584–584 / 12.03 | 178–178 / 21.65 | 17.87 / 10.14 |
| mtb_edge | Window | 584–584 / 11.86 | 178–178 / 21.51 | 17.41 / 10.15 |

## Decision

MAE/RMSE and rise/recovery improve consistently, but the benefit is not uniform across all diagnostics. Road ride 2 has slightly worse high-HR MAE and more unmatched peaks; 1s P99 roughness increases on both roads. The MTB MAE improvement is small, so the main evidence is the outdoor comparison, not strong target validation.

Fixed promotion screens compare **Window with ML-B**: mean MAE improves ≥0.20 BPM on each outdoor ride, mean RMSE never worsens, and high/rise/recovery MAE improves ≥0.15 BPM in at least one subset per ride. High-HR MAE must not worsen >0.15 BPM, peak-amplitude error >0.25 BPM or peak timing >1s. Each outdoor phase: MAE degradation≤0.15 BPM, unchanged coverage, P99 jump increase≤0.25 BPM, >5 BPM/s frequency increase≤0.1 percentage points, ≤2 extra unmatched peaks, valid numeric HR range. For stationary/restart subsets with ≥20 scored points, allow no MAE degradation >1 BPM. MTB applies the same stability/coverage/state screens and MAE/RMSE/high-HR tolerances of 0.5 BPM. These are prespecified descriptive safeguards, not significance tests.

Passed 64/64 screens. Window lowers MAE in 6/6 outdoor phases. Mean MAE improvement (ML-B minus Window): road_ride_1 +0.273 BPM; road_ride_2 +0.303 BPM.

**B. Promote the window-enhanced model as the next version, subject to user approval.**

The experiment supports another window-based candidate, but promotion is pending user approval. Only a separate candidate was produced; version 1 remains current.

Separate candidate generated: **True**. Candidate CSV: `data/processed/hr_ml_window_candidate_2026-10-16.csv`; figure: `docs/images/hr_ml_window_candidate_2026-10-16.png`. Recorded Edge HR and original watch values remain exact; output is explicitly an unapproved estimate, with no extrapolation.

## Limitations and reproduction

Two outdoor rides cannot establish generalization to technical MTB. Three sampling phases are not three rides. Only a short, previously examined MTB overlap is labelled; later sustained stops remain unvalidated. Road training has little stopping support, optical HR is imperfect, wellness averaging is unresolved, terrain is quality-screened rather than surveyed, and power/cadence are absent outdoors. A future activity window can help offline prediction without revealing true missing HR fluctuations. Strict complete-region means may reduce feature availability near gaps/stops, but they never reduce baseline prediction coverage.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_hr_ml_windows.py
```

Outputs: this report; `docs/images/hr_ml_window_evaluation.png`; `data/processed/hr_ml_window_metrics.csv`; script `scripts/evaluate_hr_ml_windows.py`. The plot repeats the benchmark's phase-0 central 20 minutes of the longest continuous reference block, chosen without model errors. No large intermediates or serialized models.

Executed checks:

Candidate-only checks: supported mean 120.96 BPM, peak 172.00 BPM, 10359/10409 supported rows; maximum estimated/handoff 1s change 1.797 BPM. The last 50s stay missing. These are estimated-sequence statistics, not recovered physiological truth. All original Edge/watch values remain unchanged; v1 remains current.

- Explicit ramp/impulse checks: four half-open 30s boundaries, future +30 included/+60 excluded, gap isolation, stop fractions, invalid-terrain coverage
- road_ride_1: hidden-HR substitution leaves all inputs/predictions identical; exact anchors and support retained
- road_ride_2: hidden-HR substitution leaves all inputs/predictions identical; exact anchors and support retained
- Repeated first Window fit gives bit-identical training predictions; unchanged hyperparameters/sampling/postprocessing
- mtb_edge: hidden-HR substitution leaves all inputs/predictions identical; exact anchors and support retained
- All 26 saved benchmark rows reproduced with identical support and numeric metrics within 1e-10
- All 162 protected source/prior-output hashes unchanged, including the v1 CSV, reports and figures.
- Numeric score/coverage invariants, mean-HR error identity, finite-range summaries, non-empty outputs. Figures are inspected after generation. No new dependencies, large test suite, commits, pushes or uploads.
