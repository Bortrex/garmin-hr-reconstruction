# Final experimental HR reconstruction: soft wellness averages

**Reconstruction CSV, model and figure completed; FIT export blocked safely.** The installed official Encoder cannot preserve this original file's unsupported messages/fields. No reconstructed FIT or upload-ready claim is produced, and nothing was uploaded.

## Model and observation assumption

One fresh trial-10 TCN fit on five road rides plus MTB rides 1 and 2; no MyWhoosh or October 16 training. Input 128×28; central positions 56–71 predict 16 residuals; 16 channels, kernel 3, two same-padded noncausal Conv1D layers per residual block, dilations 1/2/4/8, ReLU/skip connections, linear projection, 6,753 parameters, RF 61s. Adam 0.0003, batch 64, 30 epochs, seed 265, float32; training/inference strides 16/8. GPU matmul and all-output RF gradients checked on the RTX 3080 Laptop GPU in unchanged Conda deep.

**Unverified observation assumption:** a watch value at t approximates mean HR over [t−30,t+29], not instantaneous HR at t. Firmware averaging/filtering remains unknown. Synthetic means require 60 valid reference seconds in one recorded block, using the exact 142×60s/15×120s interval sequence and phases 0/20/40. Only averaged examples are trained. The 28 existing channels retain their meanings, with sparse HR and baseline now representing summaries. Normalization uses these seven reference rides only. True instantaneous residual labels include observation-center seconds, because a mean does not disclose their instantaneous HR. Masks exclude missing reference HR/unsupported baseline; temporal pairs stay in one 16s output/block.

Target context uses the original Edge speed, movement and UTC timeline, plus the existing quality-masked Strava climbing features from `data/VTT_Apero.gpx`. These are predictor features only; no original route, altitude, time, distance or other sensor value is rewritten.

Loss remains masked Huber(delta 5) + 0.5×10s absolute-HR-change MAE. Inference averages overlapping residual predictions and retains the existing centered five-sample residual mean inside each supported block. Both the 4f(1−f) taper and observation-time overwrite are absent. The interpolated average baseline is context, never proof of HR at a dot. No support is extended beyond the observations, across a missing record, or across >180s gaps.

## Training

Python 3.9.7, TensorFlow 2.6.1, Keras 2.6.0; actual fit 27.94s. Final total/Huber/temporal losses: 9.30947 / 7.90351 / 2.83453.

Keras epoch total loss and component metrics aggregate the final partial batch differently; the displayed epoch components need not add exactly to the displayed total. Each batch uses the exact fixed combined loss above.

| ride | windows | loss_seconds | temporal_pairs | supervised_average_centers |
|---|---|---|---|---|
| road_ride_1 | 954 | 15099 | 5629 | 236 |
| road_ride_2 | 801 | 12678 | 4729 | 199 |
| road_ride_3 | 1052 | 16645 | 6183 | 268 |
| road_ride_4 | 424 | 6650 | 2460 | 118 |
| road_ride_5 | 934 | 14854 | 5533 | 245 |
| mtb_ride_1 | 714 | 11412 | 4274 | 171 |
| mtb_ride_2 | 1606 | 25673 | 9620 | 386 |


## Soft consistency, fixed before outcome inspection

Let h0 be the unadjusted TCN with actual Edge samples substituted only after the withheld overlap diagnostic. Solve for adjustment δ at supported estimated seconds; δ=0 at actual Edge samples. With A the complete-window 60-s mean and w the watch summaries:

`J = sum(δ²) + 60 sum((A(h0+δ)−w)²) + 25 sum((Dδ−b)²)`

All residuals/differences are BPM on the one-second grid; the objective is BPM² with fixed dimensionless weights 1/60/25. The factor 60 gives a minute-level mean penalty comparable to 60 per-second fidelity terms; these are conservative engineering weights, not calibrated measurement precisions. D connects adjacent supported estimated samples only, with b=0, discouraging rough adjustments. At a measured/estimated seam it instead includes the available one-step boundary with b=−(h0_right−h0_left), encouraging continuity with the fixed real HR. No operator crosses unsupported time. The sparse SPD normal equations are solved once using SciPy. Finite weights do not impose exact dot or window equality. Frozen-only windows contribute constant disagreement and cannot change real Edge HR.

Solver/objective audit: `{'adjustable_seconds': 8834, 'adjustable_windows': 136, 'objective_before': 73147.40027858384, 'objective_after': 38110.31867104776, 'normal_equation_relative_error': 3.3210519262758865e-15}`.

## Original Edge overlap (model only, before substitution or consistency)

| n | bias | mae | rmse | p95_abs | max_abs | high_hr_n | high_hr_mae | reference_n | coverage_percent | watch_window_n | watch_mean_bias | watch_mean_mae |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1487 | -1.11999 | 12.59427 | 16.72888 | 36.29922 | 53.42202 | 163 | 22.78663 | 1525 | 97.5082 | 20 | -0.2147 | 2.96726 |


Actual Edge HR was absent from prediction features, baseline, normalization and training. Future non-HR measurements and actual watch summaries are permitted offline. Watch-center timestamps remain scored since averages are not instantaneous HR observations. These masks/full-ride observation support differ from the earlier 1,420-second hard-anchor diagnostic, so their errors are not a controlled head-to-head improvement claim. This previously examined initial overlap is not independent validation; the new average-trained model has no new held-out ride evaluation or tuned consistency weights.

## Final reconstruction and agreement

| recorded | estimated | missing | supported | coverage_percent | mean | maximum |
|---|---|---|---|---|---|---|
| 1525 | 8834 | 50 | 10359 | 99.51965 | 121.63801 | 176.23298 |


Mean/max describe estimates on supported seconds, not recovered physiological measurements. Every original timestamp, all original Edge values and all 158 separate watch summaries remain. Unsupported final time stays missing, not zero. Maximum/P99 adjacent supported final change: 22.0000 / 2.0000 BPM.

For estimated seconds, including the measured/estimated handoff, maximum/P99 change is 1.5110 / 0.9443 BPM. The largest full-ride jump occurs in the `recorded` portion and is preserved because original HR must remain exact.

Watch-window residual = reconstructed mean minus watch summary (BPM):

| series | n | bias | mae | rmse | p95_abs | max_abs |
|---|---|---|---|---|---|---|
| Model-only (Edge withheld) | 156 | 0.42293 | 2.37717 | 3.06994 | 5.74037 | 10.42268 |
| Combined before soft adjustment | 157 | 0.7852 | 3.74289 | 7.12583 | 12.29667 | 43.85 |
| Final, all complete windows | 157 | 0.56303 | 2.79662 | 6.72309 | 12.29667 | 43.85 |
| Final, adjustable windows | 136 | 0.26701 | 1.2048 | 1.56187 | 3.03033 | 5.33885 |
| Final, frozen Edge-only windows | 21 | 2.48016 | 13.10556 | 17.94785 | 34.96667 | 43.85 |


Incomplete reasons: `{'unsupported_HR_inside_60s_window': 1}`. Frozen-only complete windows: 21. A summary cannot override conflicting measured Edge HR.

Adjustment at estimated seconds (signed mean, MAE, RMS, P95/max magnitude in BPM): `{'n': 8834, 'bias': -0.2399752843590032, 'mae': 1.033041028754125, 'rmse': 1.3949890623858843, 'p95_abs': 2.853595739299561, 'max_abs': 5.331786215526631}`. Instantaneous final HR minus orange dots at estimated observation centers: `{'n': 137, 'bias': 0.035334932781248274, 'mae': 1.6322933861685491, 'rmse': 2.2301063906481575, 'p95_abs': 4.575961699715729, 'max_abs': 8.252966204542915, 'exact_dot_matches': 0}`.

The CSV stores every observation's raw/combined/final 60-second mean and residual, complete-window flag, exclusion reason and adjustable sample count, along with full-precision one-second HR, measured/estimated/missing provenance and terrain/support indicators.

## Difference from earlier hard-anchored results

| previous | previous_mean | previous_max | n | bias | mae | rmse | p95_abs | max_abs |
|---|---|---|---|---|---|---|---|---|
| hr_seven_ride_candidate_2026-10-16.csv | 121.60222 | 172.0 | 10359 | 0.03579 | 1.46463 | 2.11686 | 4.46921 | 9.93957 |
| hr_tcn_tensorflow_candidate_2026-10-16.csv | 121.66399 | 173.5451 | 10359 | -0.02598 | 1.80758 | 2.84609 | 5.77925 | 20.12153 |


Earlier final candidates preserve the same measured Edge prefix. The new fit, average-based training, removed taper/overwrite and soft adjustment change together; differences cannot be attributed to one component, and neither a smoother curve nor closer means establish true missing HR. The seven-ride gradient-boosting benchmark remains accessible and is not replaced.

![Final HR, fixed zoom and mean consistency](images/hr_tcn_soft_average_2026-10-16.png)

Zoom rule: central 20min of supported post-disconnection time, fixed independently of errors.

## FIT audit: no uploadable export

Original source: `data/mtb_edge.fit`, 10409 chronological unique records, 1525 recorded HR seconds. Official SDK 21.218.0; original header/CRC valid; file_id/activity/session/lap/record/timer messages present. The Encoder was tested in memory on original ordered unexpanded messages. It fails with: `Could not write Message: mesg_num: 140 could not be found in the Profile`.

Unsupported message types and counts:

| message | count | exported |
|---|---|---|
| 140 | 1 | not exported |
| 288 | 7 | not exported |
| 327 | 3 | not exported |
| 326 | 18 | not exported |
| 22 | 6 | not exported |
| 141 | 1 | not exported |
| 394 | 2 | not exported |
| 147 | 5 | not exported |
| 79 | 1 | not exported |
| 534 | 10409 | not exported |
| 325 | 562 | not exported |
| 526 | 1 | not exported |


Additional unknown fields include record 107 (10,409 messages) and record 137/138/144 (1,525 each), plus session/lap/device/sport fields. Full inventory is in `results/final/fit_export_audit.json`. Garmin's Encoder ignores unrecognized fields; silently stripping these messages/fields is unsafe. No new binary encoder or unverified proprietary-field conversion is attempted. [Official Python SDK](https://github.com/garmin/fit-python-sdk), [Activity requirements](https://developer.garmin.com/fit/file-types/activity), [encoding cookbook](https://developer.garmin.com/fit/articles/cookbook/encoding_activity_files.html).

| Check | Original | Export |
|---|---|---|
| Header/CRC and required Activity messages | Pass | Not run: no export |
| Record chronology / duplicates | Unique, chronological | Not run: no export |
| Route, timestamps, distance, speed, altitude, sensor/event/lap data | Original bytes unchanged | Not run: no export |
| HR records, quantization and HR summaries | Original unchanged; CSV verified separately | Not run: no export |
| Unknown messages/fields preserved | Present | Incompatible with installed Encoder |

The requested `results/final/hr_reconstructed_2026-10-16.fit` is intentionally absent. No session/lap summaries, training effect, HRV, calories or proprietary metrics were recomputed or fabricated. A separate future lossless encoder capable of this exact Garmin payload would be needed, followed by every specified round-trip comparison. No file is ready for upload.

## Manual import, only if a separately validated FIT later exists

Do not upload this reconstruction CSV as an Activity FIT. For a future validated file: Strava web → plus icon → Upload Activity → File → Choose files ([official instructions](https://support.strava.com/en-us/articles/15402173-how-do-i-bulk-upload-activities-to-strava)); Garmin Connect web → upload/cloud icon → Import Data → Browse → Import ([official instructions](https://support.garmin.com/en-US/?faq=Ht3ZP52Kju075uKvqTqu99)). Inspect provenance/privacy before importing. Existing activity timestamps are unchanged: platforms may reject a duplicate or create another activity; import does not reliably update the existing one. Do not delete the original or shift its timestamps to bypass detection. A valid decode still cannot guarantee platform acceptance. No account was connected and no upload, publication, commit or push occurred.

## Verification, reproduction and limitations

Pre-fit deterministic checks passed for the exact centered mean, missing-window exclusion, no taper/overwrite, a known soft solution, missing-block preservation and frozen measurements. Actual GPU execution/RF, input/output shapes, all-seven-only normalization, hidden-HR substitution, finite HR, model reload, solver normal equations, exact measured HR/watch retention, original grid/support, CSV read-back and nonempty artifacts are checked in the script. 294 prior files remain byte-identical, except the explicitly updated README/.gitignore. The generated figure is visually inspected after execution. FIT export checks above remain unperformed because encoding compatibility failed before file creation.

One seed/fit, GPU nondeterminism, unknown wellness firmware filtering/latency, optical/reference sensor errors, uncertain terrain/effort relationships and absent target power/cadence remain. Physically plausible predictions and approximate local-mean agreement do not recover an unknown measured signal. This is the final experimental reconstruction, not a claim of model superiority. Implementation/documentation were AI-assisted; all numerical results come from local script execution.

```powershell
conda run --no-capture-output -n deep python scripts/reconstruct_hr_soft_average.py
```

The normal command refuses existing model/metadata to avoid another fit. `--reuse-trained` regenerates outputs using only this final model, without training. Model/normalization/audit are local under `results/final/`; the sensitive CSV and that folder are Git-ignored.
