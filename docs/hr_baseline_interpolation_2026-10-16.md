# HR baseline interpolation — 2026-10-16

## Inputs and method

Watch anchors: `data/processed/watch_hr_ride_2026-10-16.csv`, previously extracted from `494532015728_WELLNESS.fit`. 158 valid anchors, from 2026-10-16T19:33:00+02:00 to 2026-10-16T22:25:00+02:00. Cleaning removed 0 invalid rows and 0 identical duplicate rows. Conflicting duplicates or naive timestamps cause an error rather than averaging.

Original Edge: `data/mtb_edge.fit`, read using the existing official Garmin SDK dependency, with HR merging disabled. 1525 valid recorded HR values. All original FIT SHA-256 hashes were checked unchanged after the run.

Both curves use only watch observations at their actual times. NumPy linear interpolation and SciPy PCHIP are applied separately to segments split at gaps >180 seconds. No extrapolation, rounding of BPM, Edge calibration, training, or curve tuning. Every valid watch anchor is preserved exactly. PCHIP is [shape-preserving](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html); it does not create extrema outside neighboring anchor bounds. Installed SciPy version: 1.7.3.

Output: 10409 one-second rows spanning 2026-10-16T19:32:22+02:00–2026-10-16T22:25:50+02:00. 10321 rows have reconstructed values, including 158 unchanged anchors and 10163 estimated non-anchor rows. 88 boundary/gap rows remain missing. Gaps exceeding 180 seconds: 0.

## Validation

Edge HR runs 2026-10-16T19:32:22+02:00–2026-10-16T19:57:46+02:00. First missing record: 2026-10-16T19:57:47+02:00. Evaluation includes all 1487 finite, exact UTC timestamp pairs from 2026-10-16T19:33:00+02:00 through 2026-10-16T19:57:46+02:00. 38 Edge samples lack supported estimates (boundaries or excluded gaps). Difficult intervals are not selectively removed. The same pairs are used for both methods.

| Method | Pairs | MAE | RMSE | Bias | Mean estimate | Mean Edge | Mean HR difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear | 1487 | 12.7797 | 17.1767 | -0.9523 | 95.0786 | 96.0309 | -0.9523 |
| pchip | 1487 | 13.1135 | 17.7500 | -0.8361 | 95.1949 | 96.0309 | -0.8361 |

All HR/error values are BPM. Bias is estimate minus Edge; mean HR difference is the difference of means on the same pairs and therefore equals bias.

Both inputs have timezone-aware UTC times. Watch anchors are on minute boundaries; Edge samples are on integer seconds. There is no timestamp evidence establishing a clock offset, so no shift is applied. Apparent timing differences in fluctuations may reflect clock offset, storage averaging or filtering; these cannot be separated from this overlap alone. This is an offline reconstruction: interpolation can use watch anchors later than each evaluation timestamp, including the 19:58 anchor for the final overlap seconds. It is not a causal, real-time prediction experiment.

## Peaks and rapid changes

Upper-decile Edge observations (>= 118 BPM; 163 pairs) have mean signed errors of linear: -25.23 BPM, pchip: -25.48 BPM.

At the Edge maximum (146 BPM), mean estimated HR across tied maximum timestamps is linear: 103.80 BPM, pchip: 102.79 BPM.

Rapid-change diagnostic: 109 paired seconds where successive Edge readings change by >=2 BPM. MAE: linear: 14.03 BPM, pchip: 14.65 BPM. All remain in the primary metrics.

Maximum absolute one-second change on the paired window: Edge 22.00 BPM; linear 0.63 BPM; pchip 0.90 BPM. Sparse-anchor interpolation smooths rapid fluctuations; negative high-HR errors demonstrate peak underestimation in this overlap, not necessarily every individual peak.

## Mean reconstructed HR and recommendation

Arithmetic mean over the 10321 valid one-second rows (anchors included): linear **121.3781 BPM**; PCHIP **121.3734 BPM**. PCHIP minus linear: -0.0046 BPM. This is the watch-covered interval, not a filled whole-session average.

Linear has the lower full-overlap MAE. Linear is the preferred first baseline here because it also has lower RMSE and is simpler; the small difference does not establish a general advantage on other rides. Use the measured error table, not visual smoothness, to choose between these baselines. Neither can restore unobserved second-by-second physiology. Both are useful reference baselines before evaluating additional models.

Validation covers only the initial approximately 25 minutes. Later ride accuracy is unknown, including at higher exertion. Wellness HR may be stored averages or filtered measurements; its averaging window is unknown. It is not confirmed instantaneous ground truth. One-second output resolution does not imply one-second measurement information. No external activities or machine learning were used.

## Outputs and verification

- Figure: `docs/images/hr_baseline_interpolation_2026-10-16.png`.
- CSV: `data/processed/hr_baseline_interpolation_2026-10-16.csv`.
- Script: `scripts/reconstruct_hr_baseline.py`.

The CSV stores UTC timestamps, floating-point linear/PCHIP estimates, exact watch/Edge observations, anchor/interpolation flags and a support status. Blank values are unavailable. No values are rounded for CSV display. Estimated timestamps are explicitly flagged; observed anchors retain their stored values without asserting an instantaneous measurement window.

The script checks one-second spacing, exact anchor preservation, missing unsupported intervals, local neighboring-value bounds, valid pairing and unchanged original FIT hashes. It also round-trips the CSV to check values and missingness.

```powershell
.\.venv\Scripts\python.exe scripts/reconstruct_hr_baseline.py
```
