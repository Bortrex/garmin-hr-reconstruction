# Cycling predictor quality and MTB stops

**MTB: 20 probable sustained stops, 3381s (56.35 min), 32.48% of recorded elapsed duration.** These are inferred stationary intervals, not identified ravitos. Five ravitos and additional waits are known from the rider's account; their identities and boundaries cannot be recovered from these channels alone.

## Data and recording quality

Existing official Garmin SDK decoder reused; UTC internally, Europe/Brussels (UTC+02:00) for displays. Source FITs and existing HR artifacts remain unchanged. No HR model or HR estimates are produced. FIT scaling is applied by the SDK: speed m/s, distance/elevation metres, cadence rpm, power watts. GPS semicircles convert with π/2³¹ to radians for great-circle metres.

| Activity | Records | Elapsed min | Missing s / gaps / coverage % | Speed zero % | Probable observed stop seconds | Altitude flags | Distance-speed MAE / P95 (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mtb_edge | 10409 | 173.47 | 0 / 0 / 100.00 | 35.25 | 3381 | 8 | 0.270 / 1.052 |
| road_ride_1 | 17735 | 329.02 | 2007 / 20 / 89.83 | 0.17 | 0 | 0 | 0.193 / 0.525 |
| road_ride_2 | 15455 | 294.27 | 2202 / 22 / 87.53 | 0.19 | 0 | 0 | 0.238 / 0.646 |
| virtual_road_ride_1 | 5115 | 85.23 | 0 / 0 / 100.00 | 0.00 | not physical | 96 | 0.046 / 0.158 |

- mtb_edge: 2026-10-16T19:32:22+02:00–2026-10-16T22:25:50+02:00.
- road_ride_1: 2026-10-04T09:20:00+02:00–2026-10-04T14:49:01+02:00.
- road_ride_2: 2026-09-26T09:49:36+02:00–2026-09-26T14:43:52+02:00.
- virtual_road_ride_1: 2026-10-12T16:27:52+02:00–2026-10-12T17:53:06+02:00.

Missing seconds are absent integer-second timestamps between the first/last record; they are not labelled observed stops. All recorded fields are nominally 1Hz. Outdoor timer pauses explain much of the discontinuous recording but do not observe physical motion during the gaps. MTB has one timer start and final stop only: genuine physical stops occurred while recording continued.

| Activity | GPS / elevation / speed / distance valid rows | Cadence / power rows | Resumed timer-pause intervals |
|---|---|---|---:|
| mtb_edge | 10409 / 10409 / 10409 / 10409 | 0 / 0 | 0 |
| road_ride_1 | 17732 / 17735 / 17732 / 17735 | 0 / 0 | 20 |
| road_ride_2 | 15455 / 15455 / 15455 / 15455 | 0 / 0 | 20 |
| virtual_road_ride_1 | 5115 / 5115 / 5115 / 5115 | 5115 / 5115 | 0 |

Recording gaps intersecting explicit resumed timer pauses: road_ride_1 20/20; road_ride_2 20/22. Other gaps are missing records of unknown cause, not automatically observed stops.

Outdoor Edge records include an ANT+ HR device labelled OHR and barometer metadata, but do not establish that the exported altitude is error-free. Cadence/power are absent from all outdoor files. MyWhoosh coordinates, altitude, distance and speed describe platform terrain/motion, not physical GPS or vertical travel. Its cadence and power may represent physical effort, but the FIT metadata does not identify their hardware source. Virtual terrain statistics below are numerical diagnostics only, not transferable outdoor workload validation.

## Stop criteria and chronological MTB candidates

A candidate is a consecutive observed run with speed ≤0.3m/s. A stationary classification additionally requires complete position/distance, cumulative-distance range ≤3m, and maximum GPS radius ≤8m around the median position. A run ≥30 elapsed seconds is a **probable sustained stop**; a shorter supported run is a short pause/transition. No runs are merged through evidence of movement. The table retains candidates ≥15s; shorter runs remain represented internally.

Very slow intervals with appreciable travel or GPS spread are slow/ambiguous movement, not stops. Riding at 0.3–1.5m/s is labelled slow movement without claiming it is a technical climb. Thresholds are conservative operational definitions, not externally validated classifiers. Distance, speed and GPS may share firmware processing; frozen coordinates are consistent with stopping but are not three independent proofs.

Duration is end timestamp minus start timestamp, never row count. Stop boundaries have at least sample-level uncertainty; slow manoeuvres may split one real-world visit into several candidate stops. The recording cannot establish their purposes.

Confidence is moderate when all export channels agree, because they may be coupled. There is no independent stop ground truth.

| Local start–end | Elapsed s | Classification | Confidence | Distance range m | Max GPS radius m | Altitude change / range m |
|---|---:|---|---|---:|---:|---:|
| 19:32:53–19:33:13 | 20 | short_pause | moderate | 0.00 | 0.01 | -0.60 / 0.60 |
| 19:33:48–19:39:22 | 334 | probable_stop | moderate | 0.00 | 0.00 | +0.20 / 0.40 |
| 19:40:19–19:40:43 | 24 | short_pause | moderate | 0.00 | 0.01 | +0.00 / 0.20 |
| 19:40:45–19:41:08 | 23 | short_pause | moderate | 0.00 | 0.00 | -0.20 / 0.20 |
| 19:41:10–19:41:37 | 27 | short_pause | moderate | 0.00 | 0.00 | +0.00 / 0.00 |
| 19:42:56–19:43:35 | 39 | probable_stop | moderate | 0.00 | 0.00 | +0.00 / 0.00 |
| 19:43:41–19:44:22 | 41 | probable_stop | moderate | 0.00 | 0.01 | +0.00 / 0.20 |
| 19:55:51–19:56:08 | 17 | short_pause | moderate | 0.00 | 0.01 | +0.00 / 0.20 |
| 19:56:14–20:01:05 | 291 | probable_stop | moderate | 0.00 | 0.00 | +0.40 / 0.60 |
| 20:01:11–20:01:30 | 19 | short_pause | moderate | 0.00 | 0.00 | +0.00 / 0.00 |
| 20:12:08–20:12:54 | 46 | probable_stop | moderate | 0.00 | 0.01 | +0.00 / 0.00 |
| 20:14:09–20:21:24 | 435 | probable_stop | moderate | 0.00 | 0.00 | -3.80 / 3.80 |
| 20:21:27–20:22:01 | 34 | probable_stop | moderate | 0.00 | 1.54 | +0.00 / 0.00 |
| 20:35:51–20:37:09 | 78 | probable_stop | moderate | 0.00 | 0.00 | -0.60 / 0.60 |
| 20:37:11–20:39:46 | 155 | probable_stop | moderate | 0.00 | 0.00 | -0.20 / 0.20 |
| 20:55:30–21:01:35 | 365 | probable_stop | moderate | 0.89 | 0.60 | +3.80 / 3.80 |
| 21:09:21–21:09:36 | 15 | short_pause | moderate | 0.90 | 0.44 | -0.60 / 0.60 |
| 21:09:38–21:12:31 | 173 | probable_stop | moderate | 0.00 | 0.00 | -4.20 / 4.40 |
| 21:20:59–21:27:50 | 411 | probable_stop | moderate | 0.00 | 0.00 | +3.80 / 7.20 |
| 21:27:56–21:28:24 | 28 | short_pause | moderate | 0.00 | 0.00 | +0.60 / 0.60 |
| 21:31:26–21:32:06 | 40 | probable_stop | moderate | 0.00 | 0.01 | +0.40 / 0.40 |
| 21:39:16–21:40:56 | 100 | probable_stop | moderate | 0.00 | 0.00 | -0.20 / 0.60 |
| 21:40:58–21:44:04 | 186 | probable_stop | moderate | 0.00 | 0.00 | -1.40 / 1.60 |
| 21:46:38–21:51:42 | 304 | probable_stop | moderate | 0.00 | 0.84 | +0.20 / 0.20 |
| 21:51:45–21:52:32 | 47 | probable_stop | moderate | 0.00 | 0.00 | +0.00 / 0.00 |
| 21:52:35–21:53:22 | 47 | probable_stop | moderate | 0.00 | 0.00 | +0.00 / 0.00 |
| 21:53:24–21:54:21 | 57 | probable_stop | moderate | 0.00 | 0.00 | +0.00 / 0.00 |
| 22:08:22–22:11:40 | 198 | probable_stop | moderate | 0.00 | 1.13 | +8.20 / 8.20 |

Zero-speed records: **3669 (35.25% of records)**. Of these, **3376 (92.01%)** lie in probable sustained stops. This record-count fraction is distinct from elapsed stationary duration. Supported short pauses account for another 293 zero-speed records. Durations should not be interpreted as exact visit boundaries.

For threshold sensitivity, requiring at least 60s instead of 30s retains 12 intervals totalling 3030s. This is a sensitivity check, not an alternative selected to match the number of known ravitos.

## Elevation findings

MTB raw altitude starts at 198.6m and ends at 304.0m (difference +105.4m), while endpoints are 35.9m apart horizontally. The discrepancy supports caution about altitude calibration or drift; it does not identify their cause or justify forcing loop closure with an arbitrary correction.

The previously stated 13.4m jump needs a boundary clarification: altitude is 135.2m at **20:34:13** and 148.6m at **20:34:16**, a 13.4m increase over 10.14m recorded travel. From the 20:34:14 sample itself to 20:34:16 the increase is 9.0m. Speed is ~3.4m/s, GPS/distance advance, there is no timestamp gap or pause, and it is not classified as a stop. The implied slope is highly suspicious; actual terrain versus altitude offset/noise cannot be established without independent elevation. The diagnostic flags it instead of correcting it.

During the probable stop beginning 22:08:22, altitude changes +8.2m over 198s while distance range is 0.00m and GPS radius 1.13m. This is evidence consistent with measurement/calibration drift during likely stationarity, but not proof of a particular sensor cause. Stable/frozen horizontal GPS alone does not guarantee numerically constant altitude.

Abrupt altitude behaviour also occurs within probable stops: 21:20:59–21:27:50, largest adjacent step 2.4m. The net change alone would hide this short-timescale variation.

The altitude flag is an adjacent observed change >2m in 1s; it is a screening threshold, not proof that every flagged change is false or every unflagged point is accurate. A centered five-sample median is shown as diagnostic filtered altitude. It suppresses isolated noise, not persistent offsets or slow drift. Source altitude remains unchanged.

Only MTB has >2m adjacent altitude steps among the outdoor records. MyWhoosh's 96 threshold exceedances describe virtual terrain/export behaviour and must not be read as physical altimeter errors. Even unflagged MTB altitude can remain biased.

## Feature definitions, stability and usable coverage

Speed candidates are centered 5/10-sample means. Climbing rates are least-squares altitude-versus-time trends over 15/30 consecutive samples (14/29s endpoint spans). Gradient uses local altitude-versus-cumulative-distance regression, bracketing approximately ±10/±25m; actual support spans at least 20/50m because of discrete records. Distance windows taking >120s are unavailable. No near-zero denominator is used.

This is rise divided by recorded path distance, an approximate gradient; the export does not certify a horizontal-distance denominator. Small road grades are less sensitive to that distinction than extreme slopes.

All smoothing/trend windows split at recording gaps and near-zero movement transitions; full window coverage is required. Terrain features are missing during stationary evidence and speed ≤0.3m/s, not calculated from altitude drift. Flagged altitude steps have an additional 30s caution margin plus the feature's support window. Legitimate steep climbs are not clipped or flattened. Elapsed stop timestamps stay on the modelling timeline with movement/quality labels.

Coverage is the percentage of recorded rows with a finite candidate, not a certification of terrain truth. P95 step is the 95th percentile absolute consecutive change. Units: m/s for speed/climbing rate; percentage points for gradient.

| Activity | Candidate | Coverage all / moving % | P95 absolute value | P95 1s change | Raw-v-filtered MAE |
|---|---|---:|---:|---:|---:|
| mtb_edge | speed_mean5 | 96.30 / 97.35 | 6.634 | 0.311 | 0.08647 |
| mtb_edge | speed_mean10 | 93.22 / 95.32 | 6.599 | 0.219 | 0.14456 |
| mtb_edge | climb15 | 56.97 / 88.87 | 0.272 | 0.019 | 0.00029 |
| mtb_edge | climb30 | 53.61 / 83.62 | 0.265 | 0.011 | 0.00012 |
| mtb_edge | gradient20 | 57.89 / 90.30 | 7.945 | 0.842 | 0.01767 |
| mtb_edge | gradient50 | 55.70 / 86.89 | 8.044 | 0.441 | 0.00732 |
| road_ride_1 | speed_mean5 | 99.18 / 99.43 | 9.802 | 0.285 | 0.06375 |
| road_ride_1 | speed_mean10 | 98.49 / 98.75 | 9.760 | 0.220 | 0.12133 |
| road_ride_1 | climb15 | 97.34 / 97.59 | 0.365 | 0.024 | 0.00045 |
| road_ride_1 | climb30 | 95.60 / 95.84 | 0.346 | 0.016 | 0.00020 |
| road_ride_1 | gradient20 | 98.22 / 98.47 | 6.937 | 0.871 | 0.02251 |
| road_ride_1 | gradient50 | 97.29 / 97.54 | 6.855 | 0.480 | 0.01145 |
| road_ride_2 | speed_mean5 | 99.17 / 99.40 | 11.270 | 0.244 | 0.06553 |
| road_ride_2 | speed_mean10 | 98.43 / 98.66 | 11.277 | 0.192 | 0.11450 |
| road_ride_2 | climb15 | 97.09 / 97.32 | 0.459 | 0.024 | 0.00031 |
| road_ride_2 | climb30 | 94.89 / 95.12 | 0.452 | 0.015 | 0.00013 |
| road_ride_2 | gradient20 | 98.10 / 98.33 | 6.618 | 0.875 | 0.01567 |
| road_ride_2 | gradient50 | 97.18 / 97.41 | 6.515 | 0.447 | 0.00788 |
| virtual_road_ride_1 | speed_mean5 | 99.73 / 99.77 | 9.684 | 0.153 | 0.02258 |
| virtual_road_ride_1 | speed_mean10 | 99.43 / 99.47 | 9.664 | 0.113 | 0.04827 |
| virtual_road_ride_1 | climb15 | 96.32 / 96.36 | 0.271 | 0.011 | 0.00002 |
| virtual_road_ride_1 | climb30 | 95.43 / 95.46 | 0.267 | 0.006 | 0.00001 |
| virtual_road_ride_1 | gradient20 | 96.52 / 96.56 | 15.437 | 0.525 | 0.00097 |
| virtual_road_ride_1 | gradient50 | 95.91 / 95.95 | 15.207 | 0.253 | 0.00057 |

Prefer 30s climbing rate and 50m gradient as the conservative candidates if their lower variability and usable coverage are acceptable; retain 15s/20m as sensitivity checks for sharper terrain changes. Longer windows naturally suppress real variation as well as noise, so numerical smoothness is not evidence of superior accuracy. No independent terrain reference establishes an optimal window. Use the 5s speed summary initially; 10s trades extra smoothness for slower start/stop response. All are candidates, not calibrated settings.

Moving coverage uses only recorded speed >0.3m/s; overall coverage includes genuine stops, where terrain features are deliberately unavailable. The near-zero raw/median trend difference shows that this filter does not resolve the underlying terrain uncertainty. Large unflagged slopes still require caution; no maximum slope was imposed to improve appearance or flatten real terrain.

## Speed/distance consistency by movement context

Derived speed is Δ recorded cumulative distance / actual Δt; only consecutive 1s observations enter comparisons. Gaps are not averaged into observed speeds. Bias is derived minus recorded. The channels may be coupled internally, so close agreement does not independently prove physical accuracy. Rows with n=0 have undefined (nan) metrics, not zero error.

| Activity | Context | n | MAE m/s | P95 absolute error m/s | Bias m/s |
|---|---|---:|---:|---:|---:|
| mtb_edge | riding (>1.5m/s) | 6057 | 0.410 | 1.219 | 0.296 |
| mtb_edge | slow/technical candidate | 615 | 0.481 | 1.212 | 0.243 |
| mtb_edge | probable stop | 3401 | 0.003 | 0.000 | 0.000 |
| mtb_edge | start/stop +/-5s | 816 | 0.238 | 0.932 | 0.051 |
| road_ride_1 | riding (>1.5m/s) | 17564 | 0.192 | 0.523 | 0.076 |
| road_ride_1 | slow/technical candidate | 106 | 0.340 | 1.100 | 0.215 |
| road_ride_1 | probable stop | 0 | nan | nan | nan |
| road_ride_1 | start/stop +/-5s | 269 | 0.336 | 0.964 | 0.194 |
| road_ride_2 | riding (>1.5m/s) | 15333 | 0.236 | 0.640 | 0.106 |
| road_ride_2 | slow/technical candidate | 64 | 0.472 | 1.121 | 0.342 |
| road_ride_2 | probable stop | 0 | nan | nan | nan |
| road_ride_2 | start/stop +/-5s | 246 | 0.400 | 1.296 | 0.221 |
| virtual_road_ride_1 | riding (>1.5m/s) | 4990 | 0.042 | 0.147 | -0.002 |
| virtual_road_ride_1 | slow/technical candidate | 122 | 0.188 | 0.572 | 0.010 |
| virtual_road_ride_1 | probable stop | 0 | nan | nan | nan |
| virtual_road_ride_1 | start/stop +/-5s | 5 | 0.206 | 0.322 | -0.206 |

Outdoor GPS step-speed versus recorded speed (additional, noisier position-based diagnostic): mtb_edge MAE 0.248m/s, P95 0.952m/s; road_ride_1 MAE 0.152m/s, P95 0.431m/s; road_ride_2 MAE 0.189m/s, P95 0.534m/s. GPS scatter, quantization, smoothing and frozen positions limit interpretation. Virtual coordinates are not used as independent physical evidence.

## Recommendation and limitations

Use elapsed time, raw/5s speed, movement state and explicit quality flags as the small common outdoor predictor set. Add positive/negative 30s climbing rate only on supported segments; use 50m gradient cautiously as a sensitivity input rather than assume it measures workload. Do not require power/cadence. Treat MyWhoosh terrain as a different domain.

Retain states for riding, slow movement, probable stop, short pause, uncertain movement and recording gaps. Do not equate a stationary state with zero cardiovascular effort or reset physiological history. HR recovery may continue through ravitos and waits; no HR is estimated here.

The speed/movement predictors support a small next modelling experiment. Terrain predictors require an ablation with and without them; they do not yet support strong accuracy claims or confident short-peak recovery. Independent altitude/route evidence and confirmation of stationary intervals would help distinguish drift from actual terrain. Five ravitos cannot be identified by ranking the five longest stops.

## Reproduction and checks

```powershell
.\.venv\Scripts\python.exe scripts/audit_cycling_predictors.py
```

Figure: `docs/images/cycling_predictor_quality.png`. Optional compact CSV: `data/processed/cycling_predictor_quality_summary.csv` (MTB candidate stops and feature summary statistics only). No dense intermediates.

Verified real-file decoding, UTC/local conversions, elapsed stop durations, gap splitting, full smoothing support, stationary feature masking, minimum gradient distance and unchanged original FIT/HR output hashes. No reconstruction, model fitting, new dependencies, commits or uploads.
