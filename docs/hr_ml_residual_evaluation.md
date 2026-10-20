# Gradient-boosted HR residual evaluation

**Final selection: ML-B.** The correction algorithm/configuration and linear base policy were fixed before held-out scoring. Model selection below applies the predefined gates once; no redesign or parameter tuning follows the scores.

## Data and method

Two outdoor FIT rides provide leave-one-ride-out evaluation (17,735 and 15,455 HR records). The 5,115-record virtual ride is a separate transfer check, never pooled into fitting. The target has 10,409 activity timestamps, 1,525 Edge HR observations, 158 watch summaries and 20 probable sustained stops. Reuse the existing official FIT decoder, UTC timestamps, sparse observation definitions, interpolation, and terrain/movement masks. No raw data or earlier output is changed.

Base policy: **linear interpolation**, fixed for its simple bounded segments and conservative behavior. PCHIP remains an independent comparator, not a base chosen using the new held-out results. Maximum anchor gap 180s, no extrapolation, split at every unsupported reference/recording gap.

Installed scikit-learn: **1.1.3**. [The version-matched official API](https://scikit-learn.org/1.1/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) documents native NaN routing. Missing terrain stays NaN, with a support indicator; no gradient is silently filled with zero. The existing installation is used; no dependency installation was necessary.

`estimated_HR = linear_HR + correction`; the supervised target is `recorded_HR − linear_HR`. One HistGradientBoostingRegressor per ablation configuration, with the same fixed settings:

```text
{'learning_rate': 0.05, 'max_iter': 120, 'max_leaf_nodes': 7, 'max_depth': 3, 'min_samples_leaf': 100, 'l2_regularization': 10.0, 'early_stopping': False, 'random_state': 265}
```

Automatic early stopping is disabled, avoiding a random validation split of neighboring seconds. No parameter search, scaler or imputation is fitted on evaluation rides. Tree bins/NaN routes are learned on training data only. The script limits thread pools to two for predictable runtime.

### Compact feature set

- Anchor context (7): linear HR, neighboring-anchor HR difference, seconds since/until anchors, gap duration, interval fraction and linear slope (BPM/s). Both neighboring **observed** anchors are permitted in this offline task.
- Movement (8): trailing speed5/speed30 (m/s), 10s change in speed5 (m/s²), stationary evidence, time in a stationary run, time since an observed restart (ages capped at 300s), moving fraction over 60s, uncertainty flag. Restart age is missing before a supported restart; observed run age is not a claim about unobserved motion. Stationary evidence includes supported short pauses; sustained-stop support is scarce in the road training data.
- Terrain (5, ML-B only): positive/negative 30-sample climbing-rate magnitudes (m/s), their trailing 60s means, and current terrain support. Reuse centered least-squares rate after the 5-sample median, movement/gap splits and >2m/1s step caution masks. Full 60 valid samples are required for histories. Offline terrain is not delayed by the ODE experiment's causal lag. Prefer Strava elevation on MTB; roads use native altitude.
- ML-A uses context+movement (15 inputs); ML-B adds terrain (20). Gradient is omitted because it is redundant with speed/climbing rate and has unvalidated MTB local slopes. No absolute position, activity identity, hidden HR, rolling dense HR, power, cadence, calories or load summaries enter inputs.

### Training, postprocessing and evaluation

Replay the exact 142×60s/15×120s interval sequence at phases 0/20/40s in elapsed UTC time. Point observations are primary. Averaged sensitivity uses complete observed [t−30,t+29] 60-second windows, without refitting the point-trained models. Its averages are explicitly treated as pseudo-point interpolation constraints for this sensitivity assumption, not established instantaneous measurements or a verified Garmin mechanism.

Training retains only timestamps whose elapsed seconds are divisible by five, assigning each selected timestamp to exactly one phase cyclically. Thus the same hidden HR target is never triplicated across phases. Known anchors and unsupported rows are excluded. Targets are supervised training labels only; no hidden evaluation HR affects features or normalization. Fitted sample counts: road_ride_2: 2131; road_ride_1: 2509; road_ride_1+road_ride_2: 4640.

Fixed stability rule: average predicted residuals over five centered samples **within each supported anchor interval**, then multiply by `4f(1−f)` (f=interval fraction). This tapers corrections to zero at observations; no settings are chosen from held-out curves, no residual clipping or extra ML algorithm is used. Point anchors are exact. For averaged/actual wellness summaries, passing through them is an explicit pseudo-point assumption, not recovery of true instantaneous HR.

Every ride/phase uses the same intersection of point/averaged support and all four models, with all anchor times excluded. Coverage is relative to valid recorded reference HR, not elapsed gaps. High HR = reference upper decile; rise/recovery = ≥+5/≤−5 BPM over 30 consecutive observed seconds. Peaks: prominence≥10 BPM, separation≥120s; eligible ±15s windows must be completely scored. Peak amplitude is estimated local maximum minus reference peak; timing uses the earliest local maximum and is bounded/censored by ±15s. Unmatched predicted peaks have no reference peak within ±30s. These are score-only diagnostics, not model inputs. One-second HR records and phases are correlated.

## Outdoor holdouts

Equal-weight phase means [minimum, maximum], BPM. Bias also equals average-HR error on the same timestamps.

| Held-out ride | Model | MAE | RMSE | Bias | High-HR MAE | Coverage % |
|---|---|---:|---:|---:|---:|---:|
| road_ride_1 | linear | 4.13 [4.09, 4.17] | 6.20 [6.10, 6.36] | -0.10 | 1.92 | 84.94 [84.27, 85.60] |
| road_ride_1 | pchip | 4.04 [3.99, 4.11] | 6.24 [6.09, 6.41] | -0.08 | 1.57 | 84.94 [84.27, 85.60] |
| road_ride_1 | ML-A | 4.15 [4.11, 4.19] | 6.11 [6.01, 6.29] | 0.16 | 2.01 | 84.94 [84.27, 85.60] |
| road_ride_1 | ML-B | 3.77 [3.76, 3.78] | 5.52 [5.37, 5.62] | 0.28 | 1.72 | 84.94 [84.27, 85.60] |
| road_ride_2 | linear | 3.92 [3.89, 3.96] | 5.99 [5.96, 6.04] | 0.22 | 1.44 | 82.42 [81.40, 83.31] |
| road_ride_2 | pchip | 3.88 [3.84, 3.90] | 6.04 [5.96, 6.10] | 0.18 | 1.37 | 82.42 [81.40, 83.31] |
| road_ride_2 | ML-A | 3.90 [3.83, 4.03] | 5.89 [5.82, 5.99] | 0.14 | 1.56 | 82.42 [81.40, 83.31] |
| road_ride_2 | ML-B | 3.64 [3.59, 3.74] | 5.37 [5.26, 5.45] | 0.18 | 1.32 | 82.42 [81.40, 83.31] |

| Ride | Phase s | Linear MAE/RMSE | PCHIP MAE/RMSE | ML-A MAE/RMSE | ML-B MAE/RMSE |
|---|---:|---:|---:|---:|---:|
| road_ride_1 | 0 | 4.12/6.15 | 3.99/6.09 | 4.15/6.01 | 3.76/5.37 |
| road_ride_1 | 20 | 4.17/6.36 | 4.11/6.41 | 4.19/6.29 | 3.76/5.62 |
| road_ride_1 | 40 | 4.09/6.10 | 4.01/6.21 | 4.11/6.03 | 3.78/5.57 |
| road_ride_2 | 0 | 3.89/6.04 | 3.84/6.06 | 3.83/5.82 | 3.59/5.26 |
| road_ride_2 | 20 | 3.90/5.97 | 3.90/6.10 | 3.84/5.85 | 3.59/5.39 |
| road_ride_2 | 40 | 3.96/5.96 | 3.89/5.96 | 4.03/5.99 | 3.74/5.45 |

### Short fluctuations, recovery, peaks and stability

Roughness includes anchor neighborhoods and requires consecutive supported seconds; it never bridges a recording gap. Large changes are inspection flags, not physiologically validated maximum rates.

| Ride | Model | Rise MAE | Recovery MAE | Peak amplitude abs error | Peak timing abs s | P99 abs 1s step | Unmatched peaks (phase mean) |
|---|---|---:|---:|---:|---:|---:|---:|
| road_ride_1 | linear | 5.08 | 5.11 | 9.12 | 15.00 | 0.689 | 8.3 |
| road_ride_1 | pchip | 4.95 | 5.09 | 8.54 | 15.00 | 0.836 | 8.3 |
| road_ride_1 | ML-A | 4.98 | 5.22 | 8.58 | 13.02 | 0.813 | 8.3 |
| road_ride_1 | ML-B | 4.41 | 4.82 | 7.10 | 10.24 | 0.947 | 7.0 |
| road_ride_2 | linear | 4.48 | 5.48 | 8.09 | 15.00 | 0.728 | 5.3 |
| road_ride_2 | pchip | 4.44 | 5.42 | 7.52 | 15.00 | 0.888 | 5.3 |
| road_ride_2 | ML-A | 4.51 | 5.27 | 7.91 | 13.07 | 0.836 | 5.7 |
| road_ride_2 | ML-B | 4.15 | 4.74 | 6.73 | 10.31 | 0.983 | 3.7 |

| Ride | Model | Averaged-anchor MAE | Averaged-anchor RMSE |
|---|---|---:|---:|
| road_ride_1 | linear | 4.13 [4.05, 4.22] | 5.70 [5.58, 5.80] |
| road_ride_1 | pchip | 3.88 [3.82, 3.94] | 5.47 [5.45, 5.51] |
| road_ride_1 | ML-A | 4.21 [4.18, 4.26] | 5.72 [5.69, 5.75] |
| road_ride_1 | ML-B | 3.78 [3.72, 3.83] | 5.13 [5.06, 5.19] |
| road_ride_2 | linear | 3.89 [3.81, 3.95] | 5.53 [5.40, 5.64] |
| road_ride_2 | pchip | 3.69 [3.61, 3.75] | 5.36 [5.26, 5.44] |
| road_ride_2 | ML-A | 3.88 [3.79, 3.97] | 5.47 [5.35, 5.59] |
| road_ride_2 | ML-B | 3.65 [3.60, 3.69] | 5.03 [4.97, 5.09] |

### Movement and terrain contexts

Mean point-condition errors by audited context; n is the per-phase count range. Stationary means inferred stationary evidence (including short pauses), not proven rest. Restart means the first 30s after an observed restart; uphill/downhill requires supported climbing rate >+0.05/<−0.05m/s. These subsets overlap and do not identify pedaling effort. Undefined (nan) means no supported observations, not zero error.

| Activity | Model | Stationary n / MAE | Restart n / MAE | Uphill MAE | Downhill MAE |
|---|---|---|---|---:|---:|
| road_ride_1 | linear | 1–2 / 2.34 | 32–41 / 1.57 | 3.87 | 5.54 |
| road_ride_1 | ML-B | 1–2 / 5.66 | 32–41 / 4.20 | 3.57 | 4.90 |
| road_ride_2 | linear | 1–1 / 11.78 | 29–30 / 9.43 | 3.06 | 5.39 |
| road_ride_2 | ML-B | 1–1 / 9.19 | 29–30 / 8.09 | 3.00 | 4.77 |
| mtb_edge | linear | 584–584 / 11.90 | 178–178 / 21.94 | 19.97 | 10.80 |
| mtb_edge | ML-B | 584–584 / 12.03 | 178–178 / 21.65 | 17.87 | 10.14 |

Climb/descent errors improve on both outdoor rides, but stop/restart benefit is not consistent: road ride 1 restart error increases on its very small supported subset, and MTB stationary error is slightly worse. Overall acceptance does not establish accurate reconstruction of the long unobserved MTB stops.

## Separate transfer checks

Both-road fits are frozen before MTB scoring. The MTB overlap uses real watch observations only, without an Edge boundary anchor or any hidden Edge input. It ends before the 20:34 altitude anomaly; watch/Edge path differences confound interpretation and this previously examined interval is not a pristine independent test. Virtual speed/terrain are simulated and the virtual ride never trains a model.

| Domain / observation | Model | MAE | RMSE | Bias | High-HR MAE | Scored n | P99 1s step |
|---|---|---:|---:|---:|---:|---|---:|
| virtual_transfer / point | linear | 1.82 [1.74, 1.87] | 3.07 [2.96, 3.21] | -0.09 | 0.79 | 4904–4963 | 0.389 |
| virtual_transfer / point | pchip | 1.81 [1.71, 1.88] | 3.10 [2.90, 3.28] | -0.05 | 0.77 | 4904–4963 | 0.411 |
| virtual_transfer / point | ML-A | 1.89 [1.84, 1.97] | 3.06 [3.01, 3.14] | 0.08 | 0.69 | 4904–4963 | 0.485 |
| virtual_transfer / point | ML-B | 1.81 [1.78, 1.87] | 2.94 [2.84, 3.01] | 0.40 | 0.78 | 4904–4963 | 0.586 |
| virtual_transfer / averaged | linear | 1.77 [1.72, 1.86] | 2.73 [2.58, 2.93] | -0.18 | 0.70 | 4904–4963 | 0.336 |
| virtual_transfer / averaged | pchip | 1.69 [1.64, 1.75] | 2.66 [2.49, 2.86] | -0.15 | 0.64 | 4904–4963 | 0.341 |
| virtual_transfer / averaged | ML-A | 1.84 [1.78, 1.90] | 2.77 [2.72, 2.81] | -0.01 | 0.64 | 4904–4963 | 0.455 |
| virtual_transfer / averaged | ML-B | 1.74 [1.69, 1.79] | 2.62 [2.57, 2.71] | 0.31 | 0.69 | 4904–4963 | 0.566 |
| mtb_overlap / actual_watch | linear | 12.98 [12.98, 12.98] | 17.39 [17.39, 17.39] | -1.08 | 25.36 | 1420–1420 | 0.633 |
| mtb_overlap / actual_watch | pchip | 13.37 [13.37, 13.37] | 18.01 [18.01, 18.01] | -0.89 | 25.65 | 1420–1420 | 0.854 |
| mtb_overlap / actual_watch | ML-A | 13.03 [13.03, 13.03] | 17.30 [17.30, 17.30] | 0.54 | 23.78 | 1420–1420 | 0.733 |
| mtb_overlap / actual_watch | ML-B | 12.57 [12.57, 12.57] | 16.68 [16.68, 16.68] | -1.01 | 22.38 | 1420–1420 | 0.900 |

## Conservative selection

Predefined gates: on **each** outdoor ride, ≥0.25 BPM mean MAE improvement over linear, no mean RMSE worsening, ≥0.25 BPM improvement in high-HR, rise or recovery MAE, and peak-amplitude error no worse by >0.5 BPM. Each phase must avoid MAE degradation >0.25 BPM, P99 1s-step increase >0.5 BPM, >5 BPM/s step-frequency increase >0.1 percentage points, >2 extra unmatched peaks, or estimates outside 0–255 BPM. MTB veto: no MAE/RMSE degradation >0.5 BPM, high-HR degradation >1 BPM, and the same stability/peak/range checks. These are operational conservative screens, not significance tests. Among configurations passing every gate, choose lower mean outdoor MAE; otherwise retain the predefined linear baseline.

| Configuration | Gates passed | Accepted |
|---|---:|---|
| ML-A | 41/45 | False |
| ML-B | 45/45 | True |

ML-B improves MAE over ML-A in 6/6 outdoor phases. ML-A improves on linear in 2/6; ML-B in 6/6. These counts alone do not justify deployment. **Selected final method: ML-B.** The selected correction passes the predefined descriptive screens, but two roads and one short target overlap cannot establish generalization or recover the unknown true MTB fluctuations.

## Limitations and reproducibility

Road rides have few short stationary examples and no sustained stops, whereas MTB has about 56 minutes of stops. Native NaN routing and state-age features do not solve that domain mismatch. Optical sensor artifacts, uncertain wellness averaging, sparse-anchor timing, imperfect terrain, absent power/cadence and only two independent outdoor rides constrain interpretation. A smaller overall error cannot establish accurate unseen short peaks. No hyperparameters or smoothing settings were changed after held-out scoring.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_hr_ml_residual.py
```

The comparison figure uses phase-0 point observations and the central 20 minutes of each ride's longest continuous HR block. Shading marks 30s after the window maximum; window selection never uses model error. All phase/subset counts and signed peak errors are in `data/processed/hr_ml_residual_metrics.csv`.

Checks executed:

- road_ride_1: hidden-HR feature/prediction invariance; original anchors exact; injected missing-record gap remains unsupported
- road_ride_2: hidden-HR feature/prediction invariance; original anchors exact; injected missing-record gap remains unsupported
- virtual_road_ride_1: hidden-HR feature/prediction invariance; original anchors exact; injected missing-record gap remains unsupported
- mtb_edge: hidden-HR feature/prediction invariance; original anchors exact; injected missing-record gap remains unsupported
- Training-only binning, fixed parameters/linear base, no early-stopping split; phase examples deduplicated by timestamp
- All methods/conditions share hidden score times per phase; averaging windows complete; bias equals mean-HR error
- Exact Edge grid/HR and all 158 watch summaries preserved; real boundary handoff; no unsupported extrapolation
- All 155 protected source/prior-analysis hashes unchanged; no original FIT/GPX/HR output overwritten.
- Non-empty requested artifacts; matched timestamp metrics, bias/mean-HR identity, and finite-range checks. Visual readability is inspected after generation. No separate test suite, commits, pushes or uploads.
