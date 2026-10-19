# HR interpolation evaluation — three cycling rides

## Datasets

UTC is used internally; displayed start/end times are Europe/Brussels. Durations below span the first/last HR record. A one-second record is not necessarily a fresh independent physiological observation.

| Ride | Date, local HR start–end | Span | Valid HR | Coverage of elapsed seconds | Recording gaps / longest | Repeated adjacent BPM |
|---|---|---:|---:|---:|---:|---:|
| road_ride_1 | 2026-10-04T09:20:00+02:00–14:49:01 | 329.02 min | 17735 | 89.83% | 20 / 1219s | 58.63% |
| road_ride_2 | 2026-09-26T09:49:36+02:00–14:43:52 | 294.27 min | 15455 | 87.53% | 22 / 1147s | 59.50% |
| virtual_road_ride_1 | 2026-10-12T16:27:52+02:00–17:53:06 | 85.23 min | 5115 | 100.00% | 0 / 0s | 70.12% |

Outdoor activities identify an Edge 840 recorder and an ANT+ heart-rate device labeled `OHR`; the metadata does not establish its exact optical sensor model or physiological accuracy. Their timer pauses and recording gaps prevent continuous reference evaluation over the whole elapsed ride. HR is stored at 1s intervals except those discontinuities: 17,714 adjacent 1s intervals in road ride 1, 15,432 in road ride 2 and 5,114 in the virtual ride. All recorded HR values pass 0<BPM<255 validity checks; no physiological outliers are removed. Each raw inter-record gap is treated as an unsupported boundary.

The virtual activity is the confirmed FR265 optical sensor → Bluetooth → MyWhoosh path. Its developer metadata identifies MyWhoosh despite an Edge product ID in the FIT header. It has no recording gaps. Held values, latency and optical errors remain possible in all references. The road and virtual references therefore have different recording paths.

## Simulation and comparison

Replay the exact October 16 watch interval sequence from `data/processed/watch_hr_ride_2026-10-16.csv`: 158 anchors, 142×60s and 15×120s intervals (172-minute span). Start at each activity's first recorded HR time +0/+20/+40 seconds; cycle the sequence for longer rides and truncate for shorter ones. This deterministic schedule uses elapsed UTC time through pauses, never row indices. Longer/shorter rides accordingly have different anchor counts; prefix proportions need not equal the full-pattern proportions exactly.

A: point anchors equal the exact recorded HR at scheduled timestamps. B: averaged anchors are means of all 60 observed integer-second values [t−30s, t+29s]. If any required record is absent/invalid, drop that anchor; never fill reference HR. Averaging is an alternative simulation assumption, not an established Garmin storage rule. Its 60 source samples are intentionally used to define an anchor, then only that scalar anchor is supplied to reconstruction.

Reuse `interpolate_watch` and its verification from the October 16 baseline: linear/PCHIP, maximum 180s anchor gap, no extrapolation. Additionally break at every missing reference second to avoid interpolating through known recording gaps. This structural use of gap locations does not expose hidden HR values. No filtering, calibration or time shifts are used.

Within each ride/phase, all four scenario/method combinations use the identical intersection of supported timestamps with valid reference HR. Counts/percentages use all valid HR rows in that ride as denominator. All eligible errors are retained, including anchor timestamps. Point-sampled anchor accuracy is therefore partly guaranteed by design. Estimates are offline and use anchors on both sides. No independent holdout is needed to fit these fixed interpolation rules; the reference is used only to construct declared simulated anchors and score.

| Ride | Phase | Scheduled anchors (60s/120s intervals) | Point / averaged valid anchors | Evaluated n (%) |
|---|---:|---:|---:|---:|
| road_ride_1 | 0s | 301 (271/29) | 271 / 253 | 15198 (85.69%) |
| road_ride_1 | 20s | 300 (270/29) | 270 / 256 | 15438 (87.05%) |
| road_ride_1 | 40s | 300 (270/29) | 272 / 253 | 15318 (86.37%) |
| road_ride_2 | 0s | 269 (242/26) | 237 / 217 | 12798 (82.81%) |
| road_ride_2 | 20s | 268 (241/26) | 236 / 221 | 13097 (84.74%) |
| road_ride_2 | 40s | 268 (241/26) | 239 / 220 | 12978 (83.97%) |
| virtual_road_ride_1 | 0s | 79 (71/7) | 79 / 77 | 4981 (97.38%) |
| virtual_road_ride_1 | 20s | 78 (70/7) | 78 / 77 | 4981 (97.38%) |
| virtual_road_ride_1 | 40s | 78 (70/7) | 78 / 78 | 5041 (98.55%) |

## Accuracy and phase variability

Values are the equal-weight mean across all three phases [minimum, maximum]; phases are not independent rides. Bias is estimate minus reference. The average-HR error equals bias on these same evaluation timestamps, rather than comparing a supported estimate with a full-ride reference mean.

| Ride | Anchors | Method | MAE (BPM) | RMSE (BPM) | Mean-HR error / bias (BPM) |
|---|---|---|---:|---:|---:|
| road_ride_1 | point | linear | 4.06 [4.02, 4.10] | 6.15 [6.05, 6.31] | -0.10 [-0.22, 0.13] |
| road_ride_1 | point | pchip | 3.97 [3.92, 4.04] | 6.19 [6.04, 6.36] | -0.08 [-0.21, 0.15] |
| road_ride_1 | averaged | linear | 4.11 [4.03, 4.19] | 5.67 [5.56, 5.77] | -0.05 [-0.09, 0.02] |
| road_ride_1 | averaged | pchip | 3.86 [3.81, 3.92] | 5.45 [5.43, 5.48] | -0.02 [-0.07, 0.05] |
| road_ride_2 | point | linear | 3.85 [3.82, 3.89] | 5.94 [5.91, 5.99] | 0.22 [-0.32, 0.57] |
| road_ride_2 | point | pchip | 3.81 [3.78, 3.83] | 5.99 [5.91, 6.05] | 0.18 [-0.34, 0.53] |
| road_ride_2 | averaged | linear | 3.87 [3.79, 3.93] | 5.51 [5.37, 5.61] | 0.11 [-0.15, 0.38] |
| road_ride_2 | averaged | pchip | 3.68 [3.60, 3.73] | 5.33 [5.24, 5.42] | 0.12 [-0.10, 0.37] |
| virtual_road_ride_1 | point | linear | 1.80 [1.71, 1.84] | 3.05 [2.93, 3.18] | -0.09 [-0.27, 0.20] |
| virtual_road_ride_1 | point | pchip | 1.78 [1.68, 1.85] | 3.07 [2.88, 3.25] | -0.04 [-0.23, 0.25] |
| virtual_road_ride_1 | averaged | linear | 1.76 [1.71, 1.85] | 2.72 [2.57, 2.92] | -0.17 [-0.20, -0.14] |
| virtual_road_ride_1 | averaged | pchip | 1.69 [1.63, 1.74] | 2.65 [2.48, 2.85] | -0.14 [-0.20, -0.10] |

PCHIP has lower MAE in 17/18 ride/scenario/phase combinations; linear wins 1/18. This descriptive count does not establish superiority or statistical significance. Full phase-level metrics, paired means and evaluation counts are in the compact CSV. Across phases, point-sampled PCHIP has slightly lower mean MAE but higher mean RMSE than linear on each ride. With averaged anchors, PCHIP has lower mean MAE and RMSE on each ride. Averaging reduces RMSE for both methods on all three rides; linear MAE changes little, while PCHIP MAE improves.

## High HR and rapid changes

High HR is the ride's upper decile of recorded BPM (ties included). Rapid change is an absolute ≥2 BPM difference between consecutive observed seconds, never across a gap. These diagnostic subsets do not replace the primary scores.

| Ride | Anchors | Method | High-HR bias, mean over phases | Rapid-change MAE, mean over phases |
|---|---|---|---:|---:|
| road_ride_1 | point | linear | -1.40 | 6.11 |
| road_ride_1 | point | pchip | -0.85 | 6.41 |
| road_ride_1 | averaged | linear | -1.87 | 5.25 |
| road_ride_1 | averaged | pchip | -1.27 | 5.16 |
| road_ride_2 | point | linear | -1.08 | 6.01 |
| road_ride_2 | point | pchip | -0.83 | 6.37 |
| road_ride_2 | averaged | linear | -1.45 | 5.23 |
| road_ride_2 | averaged | pchip | -1.24 | 5.19 |
| virtual_road_ride_1 | point | linear | -0.53 | 4.51 |
| virtual_road_ride_1 | point | pchip | -0.45 | 4.77 |
| virtual_road_ride_1 | averaged | linear | -0.48 | 3.55 |
| virtual_road_ride_1 | averaged | pchip | -0.39 | 3.37 |

Examples at phase 0, point scenario: each method's largest underestimate within the scored region. These deliberately show worst cases rather than typical errors; all timestamps remain in the primary metrics.

| Ride | Method | Local timestamp | Recorded | Estimate |
|---|---|---|---:|---:|
| road_ride_1 | linear | 2026-10-04T11:41:30+02:00 | 148 | 116.00 |
| road_ride_1 | pchip | 2026-10-04T11:41:30+02:00 | 148 | 116.00 |
| road_ride_2 | linear | 2026-09-26T12:33:13+02:00 | 147 | 122.23 |
| road_ride_2 | pchip | 2026-09-26T12:33:14+02:00 | 146 | 117.67 |
| virtual_road_ride_1 | linear | 2026-10-12T17:10:34+02:00 | 147 | 131.40 |
| virtual_road_ride_1 | pchip | 2026-10-12T17:10:34+02:00 | 147 | 130.12 |

## Implications and reproducibility

Compare the per-ride table before drawing outdoor/virtual conclusions: steadier virtual HR and its recording path can make interpolation easier, while outdoor changes, pauses and optical artifacts can increase error. Averaging anchors can change both peak errors and mean HR without restoring missing rapid fluctuations. Low average-HR bias does not imply low instantaneous error. Point anchors assume perfect agreement with the reference at sampled times, unlike the actual October 16 watch/Edge recording paths.

These three rides evaluate the information lost by sparse sampling under two assumptions. They do not identify Garmin's wellness algorithm, calibrate the MTB recording paths, or prove generalization to MTB. Keep the original October 16 12.78/17.18 BPM linear MAE/RMSE as separate evidence. Temporal correlation and only three rides preclude strong population claims. Synthetic anchors are simulated observations; curves remain estimates. No ML or new reconstruction method is used.

The figure uses point anchors at phase 0 and the central 20 minutes of each ride's longest uninterrupted HR block. This rule uses timestamp availability, not errors or HR shape.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_hr_interpolation.py
```

- Figure: `docs/images/hr_reconstruction_validation_3rides.png`.
- Report: `docs/hr_reconstruction_validation_3rides.md`.
- All 36 metrics rows: `data/processed/hr_reconstruction_validation_3rides_metrics.csv`.

The real-file run verifies intended anchor definitions, fully observed windows, exact anchors, no extrapolation, gap/value bounds and matching evaluation timestamps. FIT and existing October 16 output hashes are checked unchanged. No dependencies, test suite, raw-data copies, commits or uploads are created.
