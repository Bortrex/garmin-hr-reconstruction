# Physiological HR predictor feasibility — Milestone 2

## Design and interpretation

This is an exploratory forward-prediction experiment, not a final MTB reconstruction. All settings below are fixed before held-out scoring. Two outdoor rides supply the principal leave-one-ride-out comparison; phase replicates are not independent rides. A descriptive material terrain benefit was predefined as ≥0.5 BPM mean MAE reduction on **each** outdoor ride, without worsening mean RMSE. This is an operational criterion, not a validated physiological or statistical significance threshold.

| Source | Valid HR n | HR span (min) | Recorded gaps | Supported lagged terrain % | Stationary-evidence rows |
|---|---:|---:|---:|---:|---:|
| road_ride_1 | 17735 | 329.02 | 20 | 95.48 | 42 |
| road_ride_2 | 15455 | 294.27 | 22 | 94.76 | 36 |
| virtual_road_ride_1 | 5115 | 85.23 | 0 | 95.39 | 0 |
| mtb_edge | 1525 | 25.40 | 0 | 53.44 | 3736 |

Inputs: `data/road_ride_1.fit`, `road_ride_2.fit`, `virtual_road_ride_1.fit`, `mtb_edge.fit`, `VTT_Apero.gpx`, and `data/processed/watch_hr_ride_2026-10-16.csv`. Official FIT decoding and UTC handling are reused. Display times use Europe/Brussels. Road HR sensor metadata does not establish independent physiological truth. MyWhoosh is a separate recording/kinematic domain. The MTB HR span is its initial observed portion; all activity predictors span 173.47 minutes.

## Model and exact predictors

```text
dh/dt = (h_equilibrium(t) - h) / tau
h(t+dt) = exp(-dt/tau)*h(t) + (1-exp(-dt/tau))*h_equilibrium(t)
M0 equilibrium = b0
M1 equilibrium = b0 + b_speed*speed5 + b_stop*stop + b_uncertain*uncertain
M2 equilibrium = M1 equilibrium + b_up*max(climb30_lag,0) + b_down*max(-climb30_lag,0)
```

- `speed5`: trailing mean of up to five recorded speed values in m/s, split at every recording gap. Missing speed uses the **training** speed median and an uncertainty flag; no held-out normalization or HR-derived feature is used.
- `stop`: prior audit's stationary evidence, including supported short pauses as well as probable stops. `uncertain`: missing speed or speed≤0.3m/s without stationary evidence. Movement context uses existing distance/GPS criteria; absolute location is not a predictor. HR state continues to evolve at stops and is reset only by permitted sparse observations. The roads contain no probable sustained stops, so long MTB stop recovery is weakly identified.
- Terrain: reuse the 5-sample median and centered 30-sample altitude/time least-squares climbing rate (29s endpoint span), with gap, movement and >2m/1s step masks and the prior 30s caution margin. Delay it by 17 recorded seconds **inside each continuous moving run** so even the median's numeric support precedes the current input time. Uphill and downhill magnitudes are in m/s. No shifting crosses stops or gaps. Existing retrospective quality classification/masks remain offline: this is HR-causal prediction conditioned on an audited activity, not a fully real-time detector.
- The audited 50m distance-gradient is retained in preprocessing but omitted as a model input: it is correlated with climbing rate/speed, and MTB local slope accuracy is unverified. This is a prespecified parsimonious choice, not a search using held-out performance.
- On unsupported terrain, M2's instantaneous **workload correction is zero**, explicitly falling back to M1 equilibrium, not asserting flat terrain. A previous terrain effect may remain in HR state and decay normally. M2 freezes M1 coefficients and tau, fitting only two residual terrain coefficients; fallback is identical in training and evaluation.
- No power, cadence, calories, load summaries, absolute location or hidden evaluation HR enters the feature matrix. Physiological state supplies the recent workload history. No endpoint correction, drift calibration, state-space smoothing or offline assimilation is applied.

### Literature and parameter interpretation

The first-order differential equation follows the structure in [Schnack et al. (2026)](https://doi.org/10.1186/s13102-026-01727-4). Its running-specific energy-expenditure equation, evolutionary optimizer and endpoint drift correction are not transferred here. The [official repository](https://github.com/tjorvenschnack/HR_Model) overview was reviewed; implementation-file fetching was unavailable, so this script implements the stated equation directly rather than copying or claiming reproduction of that code.

[Lefever et al.](https://pubmed.ncbi.nlm.nih.gov/24444235/) supports compact cycling HR dynamics but used measured power, unlike this experiment. [Hunt et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6705825/) reports a cycling time constant of 68.7±21.5s under controlled conditions. The fixed grid **30, 60, 90s** is a broad, physiologically motivated feasibility range, not a subject-specific confidence interval. Tau is in seconds; constant-input HR closes about 63% of the equilibrium gap in one tau. Equilibrium is a fitted statistical asymptote, not measured resting/maximal HR or cycling power. Linear speed coefficients can be negative because faster coasting does not imply greater effort.

### Fitting

Unroll the entire multi-second trajectory from the left point anchor; for fixed tau this is linear in equilibrium coefficients. Fit mean squared trajectory error plus 0.1 times the sum of squared standardized coefficients (intercept unpenalized). Speed is centered/scaled using training-only mean/std; terrain magnitudes are scaled but not centered, preserving exact fallback. Tau minimizes this regularized training objective on phase-0 point trajectories. M0 and M1 each use three candidates; M2 inherits M1 tau. No held-out or MTB result chooses settings. After both outdoor folds, the same fixed procedure fits both roads for the separate virtual and MTB checks. No virtual or MTB HR trains a coefficient.

Coefficients below are converted back to native units. Speed gain: BPM/(m/s); uphill/downhill gain: BPM/(m/s vertical); flags: BPM; intercept: BPM at zero speed, not resting HR. Small stop/uncertainty coefficients reflect little training support and regularization.

| Training | Model | Tau s | Intercept | Speed | Stop | Uncertain | Uphill | Downhill | Training RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| road_ride_2 | M0 | 90 | 138.629 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 9.93 |
| road_ride_2 | M1 | 90 | 149.863 | -1.674 | -0.001 | 0.000 | 0.000 | 0.000 | 9.64 |
| road_ride_2 | M2 | 90 | 149.863 | -1.674 | -0.001 | 0.000 | 54.665 | -23.750 | 8.90 |
| road_ride_1 | M0 | 90 | 131.283 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 10.43 |
| road_ride_1 | M1 | 90 | 148.875 | -2.539 | -0.004 | -0.001 | 0.000 | 0.000 | 10.07 |
| road_ride_1 | M2 | 90 | 148.875 | -2.539 | -0.004 | -0.001 | 73.518 | -38.694 | 8.91 |
| both roads (transfer only) | M0 | 90 | 134.655 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 10.27 |
| both roads (transfer only) | M1 | 90 | 148.695 | -2.056 | -0.003 | -0.001 | 0.000 | 0.000 | 9.95 |
| both roads (transfer only) | M2 | 90 | 148.695 | -2.056 | -0.003 | -0.001 | 65.426 | -29.938 | 8.99 |

A selected boundary tau signals limited identification/range sensitivity; the grid is not expanded after observing held-out scores. No physiological plausibility clipping hides prediction errors. Training RMSE is descriptive fit, not independent validation.

## Sparse-interval evaluation

Replay the existing 142×60s/15×120s sequence in elapsed UTC time, cyclically for longer rides, at phases 0/20/40s. Point anchors use exact HR; averaged sensitivity anchors use all 60 observed samples [t−30,t+29]. Incomplete windows are dropped. Averaged anchors are an alternative observation assumption and contain up to 29s look-ahead; they are not instantaneous measurements or a verified Garmin mechanism.

An interval needs two anchors ≤180s apart in the same continuous reference/predictor block. Only the left anchor **value** initializes forward prediction; the right time bounds comparison support but its value never updates preceding states. Hidden preceding-second HR is never supplied. Stop timestamps remain in trajectories; unsupported recording gaps terminate them and require another sparse initialization. No prediction beyond the final supported anchor is scored. Linear/PCHIP reuse the existing offline implementation and can use future anchors, so they have an information advantage.

Within each ride/phase, all five methods and both observation scenarios use the identical intersection of supported timestamps, excluding every scheduled anchor timestamp. Coverage denominator is all valid recorded HR rows. This deliberately differs from the earlier interpolation report, which included anchors. Bias is estimate minus reference; it also equals average-HR error on matching timestamps. Quality masks determine fallback, not a selective removal of difficult terrain from evaluation.

### Outdoor results

Equal-weight phase means [minimum, maximum]; BPM except coverage. Full phase/subset counts and errors are in the compact CSV.

| Held-out ride | Method | MAE | RMSE | Bias / average-HR error | High-HR MAE | Coverage % |
|---|---|---:|---:|---:|---:|---:|
| road_ride_1 | M0 | 8.20 [7.95, 8.33] | 10.84 [10.55, 11.13] | 2.15 | 9.61 | 84.94 [84.27, 85.60] |
| road_ride_1 | M1 | 8.04 [7.79, 8.19] | 10.58 [10.30, 10.86] | 2.05 | 8.96 | 84.94 [84.27, 85.60] |
| road_ride_1 | M2 | 7.35 [7.10, 7.58] | 9.85 [9.62, 10.19] | 2.54 | 5.70 | 84.94 [84.27, 85.60] |
| road_ride_1 | linear | 4.13 [4.09, 4.17] | 6.20 [6.10, 6.36] | -0.10 | 1.92 | 84.94 [84.27, 85.60] |
| road_ride_1 | pchip | 4.04 [3.99, 4.11] | 6.24 [6.09, 6.41] | -0.08 | 1.57 | 84.94 [84.27, 85.60] |
| road_ride_2 | M0 | 7.90 [7.50, 8.18] | 10.63 [10.07, 10.92] | -2.41 | 13.84 | 82.42 [81.40, 83.31] |
| road_ride_2 | M1 | 7.54 [7.13, 7.84] | 10.19 [9.64, 10.50] | -2.28 | 12.38 | 82.42 [81.40, 83.31] |
| road_ride_2 | M2 | 6.67 [6.38, 6.86] | 9.17 [8.80, 9.36] | -1.38 | 7.48 | 82.42 [81.40, 83.31] |
| road_ride_2 | linear | 3.92 [3.89, 3.96] | 5.99 [5.96, 6.04] | 0.22 | 1.44 | 82.42 [81.40, 83.31] |
| road_ride_2 | pchip | 3.88 [3.84, 3.90] | 6.04 [5.96, 6.10] | 0.18 | 1.37 | 82.42 [81.40, 83.31] |

| Held-out ride | Phase s | M0 MAE/RMSE | M1 MAE/RMSE | M2 MAE/RMSE | Linear MAE/RMSE | PCHIP MAE/RMSE |
|---|---:|---:|---:|---:|---:|---:|
| road_ride_1 | 0 | 8.33/10.85 | 8.15/10.57 | 7.38/9.74 | 4.12/6.15 | 3.99/6.09 |
| road_ride_1 | 20 | 7.95/10.55 | 7.79/10.30 | 7.10/9.62 | 4.17/6.36 | 4.11/6.41 |
| road_ride_1 | 40 | 8.33/11.13 | 8.19/10.86 | 7.58/10.19 | 4.09/6.10 | 4.01/6.21 |
| road_ride_2 | 0 | 7.50/10.07 | 7.13/9.64 | 6.38/8.80 | 3.89/6.04 | 3.84/6.06 |
| road_ride_2 | 20 | 8.03/10.90 | 7.65/10.43 | 6.77/9.35 | 3.90/5.97 | 3.90/6.10 |
| road_ride_2 | 40 | 8.18/10.92 | 7.84/10.50 | 6.86/9.36 | 3.96/5.96 | 3.89/5.96 |

### Peaks, increases, recovery and terrain transitions

Prespecified score-only subsets: high HR = held-out reference's upper decile; rise/recovery = ≥+5/≤−5 BPM change over 30 fully observed consecutive seconds. These diagnostic labels never enter fitting/prediction. Peaks use SciPy prominence≥10 BPM and separation≥120s within continuous reference blocks. Keep a peak only if its entire ±15s window is commonly scored; compare local estimated maximum amplitude and its earliest maximum time with the true peak. Timing is bounded by the ±15s search and can be censored at its boundary; this is local peak discrepancy, not independently matched event detection.

Terrain transition = a supported ≥10s rate>+0.05m/s followed within 60s by ≥10s rate<−0.05m/s, events ≥120s apart. Score ±30s around the end of the first qualifying descending run. Events use the original centered audit trend, not HR; they are approximate terrain labels. Zero eligible events/peaks give undefined (nan) errors, never zero error.

| Held-out ride | Method | Rise MAE | Recovery MAE | Transition MAE | Peak abs amplitude error | Peak timing abs s | Eligible peaks per phase |
|---|---|---:|---:|---:|---:|---:|---|
| road_ride_1 | M0 | 7.39 | 11.89 | 10.38 | 7.98 | 15.00 | 34/40/29 |
| road_ride_1 | M1 | 7.33 | 11.64 | 10.43 | 7.95 | 14.81 | 34/40/29 |
| road_ride_1 | M2 | 6.46 | 11.45 | 10.18 | 6.82 | 14.35 | 34/40/29 |
| road_ride_1 | linear | 5.08 | 5.11 | 6.93 | 9.12 | 15.00 | 34/40/29 |
| road_ride_1 | pchip | 4.95 | 5.09 | 6.92 | 8.54 | 15.00 | 34/40/29 |
| road_ride_2 | M0 | 9.56 | 8.29 | 9.20 | 10.36 | 15.00 | 22/26/23 |
| road_ride_2 | M1 | 9.38 | 8.11 | 9.32 | 10.46 | 14.25 | 22/26/23 |
| road_ride_2 | M2 | 8.50 | 7.86 | 8.32 | 8.68 | 13.34 | 22/26/23 |
| road_ride_2 | linear | 4.48 | 5.48 | 5.48 | 8.09 | 15.00 | 22/26/23 |
| road_ride_2 | pchip | 4.44 | 5.42 | 5.42 | 7.52 | 15.00 | 22/26/23 |

### Averaged-anchor sensitivity

Same frozen point-trained parameters and common timestamps; no refitting to averaged anchors.

| Held-out ride | Method | Point MAE | Averaged MAE | Averaged RMSE |
|---|---|---:|---:|---:|
| road_ride_1 | M0 | 8.20 | 8.05 | 10.34 |
| road_ride_1 | M1 | 8.04 | 7.89 | 10.07 |
| road_ride_1 | M2 | 7.35 | 7.16 | 9.29 |
| road_ride_1 | linear | 4.13 | 4.13 | 5.70 |
| road_ride_1 | pchip | 4.04 | 3.88 | 5.47 |
| road_ride_2 | M0 | 7.90 | 7.68 | 10.18 |
| road_ride_2 | M1 | 7.54 | 7.29 | 9.71 |
| road_ride_2 | M2 | 6.67 | 6.36 | 8.59 |
| road_ride_2 | linear | 3.92 | 3.89 | 5.53 |
| road_ride_2 | pchip | 3.88 | 3.69 | 5.36 |

## Separate transfer diagnostics

Virtual rows use the final two-road fit, never pooled virtual training. Virtual speed and terrain do not measure outdoor kinematics; power/cadence are excluded. These results are a domain-shift warning rather than outdoor validation.

The MTB diagnostic initializes only from actual FR265 wellness observations during the initial Edge-HR overlap, not hidden Edge values. Parameters and scaling are fixed from the two roads. No unavailable MTB HR is generated or exported. Compare the same M2 coefficients with Garmin and Strava terrain, alongside M1 and offline interpolation. The actual wellness observation mechanism and watch/Edge discrepancies remain confounds; this short, previously examined overlap is not independent evidence for the remaining ride. It ends at 19:57:46 Brussels, before the 20:34 discontinuity and later large altitude drift; it cannot test HR benefit in the main corrected regions.

| Domain | Anchors | Method | MAE | RMSE | Bias | High-HR MAE | Scored n / % |
|---|---|---|---:|---:|---:|---:|---|
| virtual_transfer | point | M0 | 6.15 [6.05, 6.21] | 7.49 [7.41, 7.55] | -4.72 | 8.94 | 4904–4963 / 96.26 |
| virtual_transfer | point | M1 | 4.32 [4.24, 4.43] | 5.34 [5.25, 5.50] | -2.66 | 6.06 | 4904–4963 / 96.26 |
| virtual_transfer | point | M2 | 2.60 [2.38, 2.87] | 4.32 [3.95, 4.76] | 1.21 | 1.14 | 4904–4963 / 96.26 |
| virtual_transfer | point | linear | 1.82 [1.74, 1.87] | 3.07 [2.96, 3.21] | -0.09 | 0.79 | 4904–4963 / 96.26 |
| virtual_transfer | point | pchip | 1.81 [1.71, 1.88] | 3.10 [2.90, 3.28] | -0.05 | 0.77 | 4904–4963 / 96.26 |
| virtual_transfer | averaged | M0 | 6.00 [6.00, 6.01] | 7.30 [7.28, 7.32] | -4.78 | 8.89 | 4904–4963 / 96.26 |
| virtual_transfer | averaged | M1 | 4.16 [4.10, 4.20] | 5.06 [4.98, 5.11] | -2.71 | 6.01 | 4904–4963 / 96.26 |
| virtual_transfer | averaged | M2 | 2.45 [2.30, 2.57] | 3.94 [3.66, 4.12] | 1.16 | 1.07 | 4904–4963 / 96.26 |
| virtual_transfer | averaged | linear | 1.77 [1.72, 1.86] | 2.73 [2.58, 2.93] | -0.18 | 0.70 | 4904–4963 / 96.26 |
| virtual_transfer | averaged | pchip | 1.69 [1.64, 1.75] | 2.66 [2.49, 2.86] | -0.15 | 0.64 | 4904–4963 / 96.26 |
| mtb_overlap | actual_watch | M1 | 23.42 [23.42, 23.42] | 28.60 [28.60, 28.60] | 15.76 | 21.12 | 1420–1420 / 93.11 |
| mtb_overlap | actual_watch | M2_Garmin | 22.58 [22.58, 22.58] | 27.97 [27.97, 27.97] | 15.03 | 20.26 | 1420–1420 / 93.11 |
| mtb_overlap | actual_watch | M2_Strava | 22.58 [22.58, 22.58] | 27.98 [27.98, 27.98] | 14.97 | 20.23 | 1420–1420 / 93.11 |
| mtb_overlap | actual_watch | linear | 12.98 [12.98, 12.98] | 17.39 [17.39, 17.39] | -1.08 | 25.36 | 1420–1420 / 93.11 |
| mtb_overlap | actual_watch | pchip | 13.37 [13.37, 13.37] | 18.01 [18.01, 18.01] | -0.89 | 25.65 | 1420–1420 / 93.11 |

## Decisions and limitations

1. **Movement over M0:** M1 lowers MAE in 6/6 outdoor phases; mean changes by ride (M0−M1, positive=improvement): road_ride_1 +0.16 BPM, road_ride_2 +0.36 BPM.
2. **Terrain over M1:** M2 lowers MAE in 6/6 phases; road_ride_1 +0.69 BPM, road_ride_2 +0.87 BPM. The prespecified material-benefit criterion is met.
3. **Consistency:** Both outdoor mean MAEs improve with terrain: True. Per-phase tables show whether the sign persists; six phases are not six independent rides.
4. **Peak/dynamics benefit (M1−M2 error, positive=improvement):** peak amplitude road_ride_1 +1.13; road_ride_2 +1.79 BPM; timing road_ride_1 +0.46; road_ride_2 +0.91s; rise road_ride_1 +0.87; road_ride_2 +0.88 BPM; recovery road_ride_1 +0.20; road_ride_2 +0.24 BPM. A better average score is not proof of recovering unseen short peaks.
5. **Anchor timing sensitivity:** M2 point MAE ranges are road_ride_1 7.35 [7.10, 7.58]; road_ride_2 6.67 [6.38, 6.86]. Other model/averaged results above show phase and observation-definition sensitivity.
6. **MTB Strava usefulness:** M1 MAE 23.42, Garmin-terrain M2 22.58, Strava-terrain M2 22.58 BPM. Strava's MAE advantage over Garmin is only 0.004 BPM: no meaningful source-specific HR benefit is demonstrated in this overlap. Transfer is poor relative to linear interpolation (12.98 BPM MAE), and Strava-M2 bias is +14.97 BPM. This does not validate predictions on the rest of the MTB ride.
7. **Offline assimilation next?** The consistent material outdoor terrain benefit supports a small separately validated assimilation feasibility experiment, while retaining M1 and interpolation controls. It does not justify deploying this predictor for the final MTB reconstruction: poor target transfer and weak identification of stop/recovery dynamics remain unresolved. No assimilation or final reconstruction is implemented here.

The final two-road fit has a zero-speed/stopped equilibrium of approximately 149 BPM and near-zero stop coefficient. This is not a plausible calibrated resting equilibrium: it exposes a central failure of transferring road speed/coasting associations to long MTB stops. Neither clipping HR nor imposing an arbitrary resting value was used to conceal it.

Only two outdoor rides, optical/held HR records, few stationary training samples, fixed tau, correlated seconds, sensor errors and terrain uncertainty limit interpretation. Measured pedaling power/cadence are absent outdoors. Speed is affected by coasting, wind, braking and technical riding; climbing rate is not metabolic workload. Model coefficients are descriptive, not calibrated physiological parameters. No population generalization or confidence interval follows from this experiment. Strava fixes gross elevation inconsistency but has unvalidated local gradients and some stop artifacts.

## Reproduction and verification

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_hr_physiology.py
```

Figure selection: phase-0 point observations, central 20 minutes of the longest continuous reference block in each held-out outdoor ride; no selection by HR shape or error. The figure includes all point-supported curves; metrics use the stricter shared point/average mask excluding anchors. No MTB prediction series is saved.

Outputs: `docs/hr_physiological_feasibility.md`, `docs/images/hr_physiological_feasibility.png`, `data/processed/hr_physiological_feasibility_metrics.csv`, `scripts/evaluate_hr_physiology.py`.

Executed checks:

- Analytic constant-input response, seconds/BPM units and no HR reset at an equilibrium/stop change
- road_ride_1: hidden-HR/future-anchor invariance, M1 fallback, gap splitting
- road_ride_2: hidden-HR/future-anchor invariance, M1 fallback, gap splitting
- virtual_road_ride_1: hidden-HR/future-anchor invariance, M1 fallback, gap splitting
- mtb_edge: hidden-HR/future-anchor invariance, M1 fallback, gap splitting
- Training/held-out identity separation, training-only scaling, frozen two-road transfer fit; no MTB fitting
- All 151 protected source and prior-output SHA-256 hashes unchanged; outputs exist and are non-empty. Real-file execution, FIT integrity and UTC parsing passed.
- All outdoor/virtual method/scenario score counts match within each ride/phase; MAE and bias checked against matching numeric pairs. Anchors excluded; peak windows fully supported. No installed dependencies, commits, pushes or data uploads.
- Figure readability is inspected after generation; no separate test suite is created.
