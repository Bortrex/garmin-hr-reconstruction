# FR265 wellness anchor diagnostic — 2026-10-12

**No unique wellness averaging/timestamp mechanism is established.** A short centered mean improves the training fit, but the ranking changes later. All results are exploratory evidence from one ride.

## Recording paths and input inspection

The user confirms one FR265 optical sensor: (A) normal watch wellness storage and (B) Bluetooth to a Windows PC approximately 50 cm away, recorded by MyWhoosh. No Edge was involved in this ride. MyWhoosh is a same-sensor reference path, not raw optical data or independent physiological ground truth. Proximity does not exclude latency, smoothing, buffering or retransmission loss.

Inspected 21 files in `data/2026-10-12/`: 20 identify `fr265_large`; one is Edge wellness with no valid HR and is excluded. All FIT integrity and decode checks passed. `monitoring` (55), field `heart_rate` (27), provides the watch observations. The existing audit's `monitoring_times` function resolves `timestamp_16` (26) against full monitoring timestamps, with rollover. No new FIT parser was implemented and no source timestamps were adjusted.

Recovered **1222** valid watch observations across 2026-10-12T00:01:00+02:00–2026-10-13T00:00:00+02:00, including the next midnight endpoint. **74** occur during the virtual ride: 2026-10-12T16:28:00+02:00–2026-10-12T17:53:00+02:00, all in `492952012911_WELLNESS.fit`. No invalid watch rows, duplicate anchors or conflicting timestamps were found. Exact UTC times, BPM and source filenames are in the observations CSV.

Ride spacing (seconds: count): **{'60.0': 62, '120.0': 10, '180.0': 1}**. Median 60s; mean 69.86s. Whole-export spacing: {'60.0': 1057, '120.0': 129, '180.0': 24, '240.0': 6, '300.0': 4, '480.0': 1}.

Other watch HR-related records: `monitoring_hr_data` (17 resting-HR summaries), `hrv_value` (76), `hrv_status_summary` (1). OHR settings, stress, respiration, SpO2 and monitoring fields are inventoried, not counted as additional BPM observations. No standard array `hr` messages occur. Unknown numeric message types are inventoried; their proprietary meanings remain unknown.

Automatic activity detections: {'sedentary': 133, 'generic': 134, 'walking': 2}; **no cycling classification** is present. Generic/sedentary detections during a stationary ride do not establish a cycling Move IQ event. Event reporting times are distinct from stored start/duration.

## MyWhoosh activity and clock checks

`data/virtual_road_ride_1.fit` contains **5115 valid HR records**, from 2026-10-12T16:27:52+02:00 to 2026-10-12T17:53:06+02:00. Session start is 16:27:52 and lap/session stop is 17:53:07 local: 85m15s elapsed and timer time. The summary timestamp repeats the start and is not used as an end. All 5,114 inter-record intervals are 1s; no missing/zero/invalid HR, duplicate records, recording gaps or explicit pause events were found. This cannot rule out held values masking absent Bluetooth updates.

**70.12%** of consecutive record pairs have identical BPM. The longest constant run is 49 records (48 seconds between endpoints). Thus 1 Hz storage does not prove 1 Hz independent optical observations. Largest one-second change: 8 BPM at 2026-10-12T17:10:35+02:00; no >=10 BPM jumps. Changes are retained, not removed as outliers.

The FIT header says Garmin/product 4062 (`edge_840`), which conflicts with the confirmed setup. Conversely, the developer application ID decodes to **`mywhoosh.whooshg`**, and the session sub-sport is `virtual_activity`, corroborating MyWhoosh. No `device_info` sensor metadata establishes the Bluetooth device. The header is therefore not evidence of an actual Edge in the recording chain. User confirmation is retained.

UTC record times align with the wellness date and are converted using Europe/Brussels (UTC+02:00), not naive local timestamps or the export's `local_timestamp`. Both have integer-second times; wellness anchors fall on minute boundaries. There is no independent clock synchronization log to attribute any offset.

## Experiment and evaluation protocol

Thirteen predefined zero-offset candidates: same-time recorded HR plus 15/30/60/120-second trailing, centered and leading means. For integer-second anchor t and width W: trailing uses [t-W+1, t]; leading [t+1, t+W]; centered [t-floor(W/2), t+ceil(W/2)-1]. Endpoints are inclusive, giving exactly W observations. Even centered windows have a -0.5s discrete midpoint; odd ones are exactly centered. Every second must contain an actual valid record (100% coverage); no interpolation, carry-forward or partial-window means.

Select the best unshifted averaging shape by TRAIN MAE. Then search offsets -60…+60s in 5s steps for that shape and for instantaneous records only (61 unique candidates total, including zero offsets). Positive offset reads later MyWhoosh data; negative reads earlier. The final candidate minimizes TRAIN MAE across the fixed candidates and these two selected offset variants. No held-out score sets any parameter. No exponential smoothing search was added: this small dataset already cannot resolve competing window/offset explanations.

Chronological split at **2026-10-12T17:19:00.400000+02:00**, 60% of the recorded time span. All candidates use the same anchor sets. A fully observed ±180s envelope covers every candidate support; it must also lie wholly inside its split. This intentionally purges boundary anchors so no reference window crosses training into evaluation. No anchor is excluded based on error or HR value.

Common TRAIN set: **40**, 2026-10-12T16:31:00+02:00–2026-10-12T17:16:00+02:00. Common held-out set: **22**, 2026-10-12T17:24:00+02:00–2026-10-12T17:49:00+02:00. Excluded by common support/split guard: **12** of 74 anchors. The original overlap has 74 same-time matches (MAE 3.338, RMSE 5.963); that larger set is not used to rank candidates.

## Candidate results

Errors below are **wellness minus reference representation**, in BPM. Correlations are descriptive. Displaying held-out scores for predefined alternatives does not authorize selecting a new winner from that portion.

| Candidate | TRAIN MAE | Held-out MAE | RMSE | Bias | r | Max absolute error |
|---|---:|---:|---:|---:|---:|---:|
| Same-time record | 2.650 | 2.591 | 4.528 | -0.500 | 0.638 | 17.000 |
| Trailing 15s | 2.433 | 2.570 | 4.334 | -0.491 | 0.673 | 16.400 |
| Trailing 30s | 2.170 | 2.241 | 3.630 | -0.511 | 0.743 | 13.100 |
| Trailing 60s | 2.239 | 1.465 | 2.238 | -0.617 | 0.895 | 6.333 |
| Trailing 120s | 3.341 | 0.871 | 1.150 | -0.556 | 0.977 | 2.975 |
| Centered 15s | 2.688 | 2.706 | 4.429 | -0.524 | 0.652 | 16.400 |
| Centered 30s | 2.397 | 2.686 | 4.287 | -0.547 | 0.661 | 15.633 |
| Centered 60s | 1.803 | 2.519 | 3.773 | -0.667 | 0.710 | 12.750 |
| Centered 120s | 1.906 | 2.123 | 2.930 | -0.688 | 0.812 | 8.008 |
| Leading 15s | 2.937 | 2.821 | 4.281 | -0.645 | 0.643 | 14.667 |
| Leading 30s | 3.084 | 2.850 | 4.092 | -0.832 | 0.655 | 12.733 |
| Leading 60s | 3.230 | 3.008 | 4.028 | -0.760 | 0.663 | 9.867 |
| Leading 120s | 3.855 | 2.894 | 3.984 | -0.823 | 0.653 | 9.858 |
| Point record (-25s) | 2.350 | 1.955 | 2.931 | -0.500 | 0.811 | 9.000 |
| **Centered 60s (-5s) — TRAIN choice** | 1.721 | 2.367 | 3.551 | -0.645 | 0.738 | 11.867 |

The TRAIN choice is centered 60s at -5s: it averages reference records from **t-35 through t+24**. Its held-out MAE/RMSE is **2.367/3.551 BPM**, versus 2.591/4.528 for same-time readings. The zero-offset centered 60s mean gives 2.519/3.773. Displacement alone selects -25s on TRAIN, but this differs from the averaged variant's -5s.

Trailing 120s happens to score best on the held-out portion (0.871 BPM MAE), despite much worse TRAIN MAE (3.341). It is **not promoted as the selected mechanism**. This rank reversal and the changing HR regime argue against claiming a single fixed rule was identified. A shifted centered window can also be described as another alignment of the same samples: window placement and recording delay are intrinsically confounded.

## HR regimes, discrepancies and temporal dependence

Regimes use original reference HR at t+30 minus t-30: increasing >5 BPM, decreasing <-5 BPM, stable otherwise. High HR is >=153 BPM, the TRAIN same-time 75th percentile, fixed before evaluation; it overlaps other regimes.

| Held-out regime | n | Same-time MAE | Selected MAE | Selected RMSE | Selected bias |
|---|---:|---:|---:|---:|---:|
| increasing | 1 | 5.000 | 4.450 | 4.450 | -4.450 |
| decreasing | 1 | 17.000 | 11.867 | 11.867 | 11.867 |
| stable | 20 | 1.750 | 1.788 | 2.417 | -1.081 |
| high_hr | 19 | 1.684 | 1.746 | 2.408 | -1.273 |

Only one held-out increasing and one decreasing anchor are available; neither subgroup supports reliable generalization. Full regime metrics and correlations are saved in the regimes CSV.

Selected-candidate held-out MAE varies from 1.057 to 3.921 BPM across 6 contiguous five-minute blocks (1–5 anchors per block). Neighboring values, averaging windows and errors are temporally correlated. We do not treat 22 anchors as independent replicates, report IID confidence intervals, or claim statistical significance for the modest improvement.

Largest selected-candidate held-out discrepancies:

| Local timestamp | Watch | Reference representation | Signed error |
|---|---:|---:|---:|
| 2026-10-12T17:29:00+02:00 | 155 | 143.133 | +11.867 |
| 2026-10-12T17:31:00+02:00 | 147 | 154.683 | -7.683 |
| 2026-10-12T17:30:00+02:00 | 145 | 149.450 | -4.450 |

## Interpretation and October 16 comparison

The results are compatible with averaging or filtering of a common optical signal, but they do not establish that wellness values are instantaneous, exactly 60-second means, or use a particular causal window. Modest TRAIN-selected displacement improves some agreement; its small benefit and dependence on window choice do not prove a clock error or a Bluetooth delay. Garmin storage conventions, optical filtering, Bluetooth updates and MyWhoosh buffering/recording remain confounded. The two paths originate from the same sensor.

For October 16, the selected window SHAPE (centered 60s) was tested with **zero offset** against original Edge HR. The MyWhoosh-fitted -5s delay was not transferred. On the identical 21 fully supported MTB anchor timestamps, same-time MAE/RMSE is 13.905/18.444 BPM; centered-mean MAE/RMSE is 13.106/17.948 BPM. This small improvement does not explain most MTB disagreement.

These are anchor-level comparisons, **not replacements** for the original linear baseline's 1,487 per-second pairs and 12.78 BPM MAE. The two scores target different quantities. Smoothing and sparse storage can miss rapid changes, but the October 12 recording path is not a calibration of the Edge path. Keep original instantaneous-reference baseline metrics; add window-level diagnostics as sensitivity analyses rather than redefine success to lower errors.

Next step: collect additional same-sensor paired recordings with repeated controlled rises, falls and recoveries, preserve clock/synchronization evidence, and test these fixed hypotheses on whole unseen rides. Include a separate watch-to-Edge session before transferring timing assumptions to MTB. No new reconstruction or machine-learning model is justified here.

## Outputs and reproduction

- Figure: `docs/images/watch_anchor_diagnostic_2026-10-12.png` (3200×2600).
- Report: `docs/watch_anchor_diagnostic_2026-10-12.md`.
- Processed tables: `data/processed/watch_anchor_diagnostic_2026-10-12_*.csv`: all observed anchors with provenance, candidate values, full metrics, TRAIN-only offset search, regimes, five-minute blocks, file inventory and separate MTB check. Metadata and summary JSON files accompany them.

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_watch_anchors.py
```

Uses existing NumPy, pandas, Matplotlib and Garmin SDK; no new dependencies. UTC is retained in CSVs; plot labels use Brussels. Checks independently recompute every main candidate's windows from original rows, enforce full coverage and split separation, and verify exact same-time matches. Source FITs and existing October 16 artifacts are hash-checked unchanged. No commits, uploads or modifications to original activities were performed.
