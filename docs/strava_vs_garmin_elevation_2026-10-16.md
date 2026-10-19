# Strava versus Garmin elevation — October 16, 2026

## Decision

The GPX contains a materially different elevation profile, consistent with the user's Strava correction/export workflow. It is not a copy or constant offset of Garmin altitude. It resolves two strong internal inconsistencies: route closure and the 20:34 jump. The export has no correction flag, algorithm, terrain-model provenance or independent truth, so the precise correction mechanism and absolute accuracy cannot be authenticated.

Prefer the exported Strava profile with explicit quality masks for a first terrain sensitivity experiment, retaining filtered Garmin as an alternative. This preference concerns gross drift and discontinuities, not proven local climbing-rate or gradient accuracy. Neither local feature is validated; do not build a switching hybrid yet. A limited physiological feasibility experiment is supportable with terrain ablation and uncertainty flags, not reliable second-by-second effort claims.

## Sources, export and alignment

- Garmin: `data/mtb_edge.fit`, official FIT SDK integrity-checked decoding; enhanced altitude in metres, speed in m/s, distance in metres. Original speed/distance/time remain the reference.
- Export: `data/VTT_Apero.gpx`; GPX 1.1, creator `StravaGPX`, 1 track, 1 segment; elevation `<ele>` in GPX metres, latitude/longitude in degrees, UTC `<time>`; temperature/HR extensions present but unused.
- Both: 10,409 points, 2026-10-16T17:32:22+00:00–2026-10-16T20:25:50+00:00 (19:32:22–22:25:50 Brussels, UTC+02:00); 173.47 min, 28.797 km recorded distance.
- Exact UTC timestamp matching: 10,409/10,409 (100%); all increments 1s. No missing timestamps, elevations or coordinates, duplicate times, omitted records, resampling or clipped endpoints observed. There is no timestamp shift, interpolation or extrapolation.
- Corresponding GPS separation median 0.035m, maximum 3.551m. Small export coordinate differences do not prevent exact time alignment; they may still affect terrain lookup.
- GPS bounds: latitude 50.514860–50.567475, longitude 5.283983–5.340437. Point counts/time coverage exclude temporal simplification, but cannot establish the spatial resolution of the elevation source.

### What did ‘200m to 400m’ mean?

The displayed field and its original/corrected values cannot be established from these files. GPX `<ele>` values are absolute altitude; it supplies no Strava cumulative-gain summary. Here the export peaks at 235.2m, while standardized cumulative gain is 472.9m. Thus a displayed value near 400m is consistent with gain, but is not evidence of a 400m absolute altitude. Confirming the user's exact numbers requires the labelled activity summary or screenshot; do not equate them with this calculation. Garmin FIT's ascent summary is 317m, not ~200m.

## Altitude comparison

Standardized gain/descent = sum of positive/negative adjacent finite altitude increments at consecutive 1s timestamps, including stops, with no threshold or smoothing. Both use the identical 10,408 intervals. This deliberately exposes noise/drift and is not a physical ground truth or Strava's proprietary gain calculation. Gaps and segment boundaries would not be summed across.

| Source | Start m | Finish m | Min m | Max m | Gain m | Descent m | >2m/1s steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Garmin | 198.6 | 304.0 | 64.4 | 304.8 | 347.0 | 241.6 | 8 |
| Strava | 199.5 | 198.7 | 66.3 | 235.2 | 472.9 | 473.7 | 0 |

Garmin session summary: ascent 317m, descent 211m; distinct from increment sums. Strava summary unavailable. Larger gain alone does not demonstrate improvement.

Matched altitude difference (Strava minus Garmin): mean -52.39m, median -8.00m, range -152.90 to 18.80m. Only 148 points agree within 0.11m; adjacent altitude-change correlation 0.340.

## Known problems and new cautions

- **20:34:13–16 Brussels:** Garmin 135.2→148.6m (+13.4m), export 151.6→152.2m (+0.6m). Recorded distance 10316.17→10326.31m (+10.14m): implied rise/path-distance 132.15% versus 5.92%; three-second vertical change 4.47 versus 0.20m/s. The obvious discontinuity disappears, rather than moving to another >2m/1s step nearby. No pause or recording gap explains it. These ratios use recorded path distance, not certified horizontal survey distance.
- **Route closure:** endpoints separated by 35.9m (Garmin GPS), 36.0m (export GPS). Garmin finish minus start +105.4m; export −0.8m. The points are nearby, not identical; a nonzero true elevation difference is possible, but the Garmin mismatch is strongly inconsistent with route closure.
- **Saved stops:** reused all 20 probable intervals from `data/processed/cycling_predictor_quality_summary.csv`, totalling 3381s (56m21s). No ravito labels inferred. Low speed, ≤3m distance range and ≤8m GPS radius are moderate-confidence evidence, with dependent recording channels. Export altitude range improves at 12 stops, worsens at 2, and ties within 0.05m at 6.
- At 22:08:22–22:11:40, Garmin drifts +8.2m with unchanged distance; export range 0.0m. At 21:20:59–21:27:50, Garmin range 7.2m versus export 0.0m. Counterexamples: 21:39:16–21:40:56 range 0.6→1.5m and 21:40:58–21:44:04 range 1.6→2.7m (Garmin→export), despite probable stationary states. Mask terrain at stops for both sources.

### Resolution, steep sections and stability

- Garmin: largest adjacent step 4.8m; 78.0% unchanged adjacent elevations. Longest moving unchanged-step run: 47 records starting 21:17:36, 272.1m between run endpoints. 20 instantaneous path slopes exceed ±40% where Δdistance>1m; these are inspection flags, not confirmed terrain errors.
- Strava: largest adjacent step 1.7m; 56.8% unchanged adjacent elevations. Longest moving unchanged-step run: 26 records starting 20:05:21, 153.9m between run endpoints. 13 instantaneous path slopes exceed ±40% where Δdistance>1m; these are inspection flags, not confirmed terrain errors.
- Export elevation is quantized to 0.1m; 44 adjacent identical exported GPS positions have changing elevation (maximum 0.1m). Temporal processing/rounding may contribute; this is not proof of a pure position-to-height lookup.
- Around 19:54 the exported 50m gradient reaches roughly −25%, with locally larger altitude changes than Garmin. This may be real steep descent, GPS/terrain mismatch or export processing: without surveyed terrain it cannot be classified as an improvement. Plateaus/stair steps exist in both, but neither native DEM spacing nor forest cover is encoded. Do not label sections forested or technical from elevation alone.

## Terrain features on common support

Reuse `audit_cycling_predictors.terrain_features`: centered 5-sample median (diagnostic only), 30-sample altitude/time least-squares trend (29s endpoint span), and altitude/distance least-squares gradient bracketing ±25m, at least 50m support and at most 120s duration. Gradient is rise/path-distance, not necessarily rise/horizontal-distance. Both sources use identical Garmin movement state, speed and distance. Full finite support is required, with splits at every recording gap, GPX segment boundary and low-movement transition; no smoothing or regression crosses a stop. Steps >2m/1s trigger the previous 30s caution margin plus feature support. These are screening criteria, not proof that unflagged data are valid. Source-specific flags explain coverage differences; numeric comparisons below use the intersection.

| Source | Feature (units) | Available n / all % | Common n | Common P95 abs | Common P95 1s change | Common max abs |
|---|---|---:|---:|---:|---:|---:|
| Garmin | climb30 (m/s) | 5580 / 53.61 | 5580 | 0.2649 | 0.0108 | 0.5902 |
| Strava | climb30 (m/s) | 5857 / 56.27 | 5580 | 0.4205 | 0.0190 | 0.7248 |
| Garmin | gradient50 (%) | 5798 / 55.70 | 5798 | 8.0437 | 0.4414 | 32.5595 |
| Strava | gradient50 (%) | 6030 / 57.93 | 5798 | 12.1147 | 0.7950 | 25.3130 |

P95 1s changes require both adjacent common timestamps; no change is measured across a support gap. Export gradients exceed ±20% at 93 common timestamps versus 47 for Garmin. The threshold is an inspection screen, not a validated maximum. The exported profile has higher local variability despite removal of gross drift; smoothness is neither the selection criterion nor an accuracy metric.

Sign/context checks below use the common Garmin-sign groups (±0.05m/s screen only), not independently verified climbs/descents. Values are median climbing rates:

- positive trend, 1591 timestamps: Garmin 0.068, export 0.113m/s; 83.9% sign agreement.
- negative trend, 934 timestamps: Garmin -0.107, export -0.213m/s; 90.1% sign agreement.

For sustained examples, select the longest continuous common-support run in each Garmin sign group (same ±0.05m/s screen), requiring at least 60s. This rule does not select by source agreement or appearance:

- climbing, 20:46:52–20:50:04 (192s): median Garmin 0.076, export 0.080m/s.
- descending, 21:04:12–21:08:24 (252s): median Garmin -0.072, export -0.354m/s.

Both features are deliberately unavailable at stops and until complete support after restarts. The plot's bottom ribbons show unavailable support, not zero climbing. A five-sample median cannot correct Garmin drift or establish Strava terrain resolution.

## Modelling choices and final answers

| Strategy | Advantages | Failure modes and coverage |
|---|---|---|
| Garmin with quality filtering | Native synchronized recording; small local numerical changes | Gross drift and residual slope bias remain; flags remove jump regions; stops unavailable |
| Strava with quality filtering | Corrects closure and obvious jump; more supported terrain timestamps | Larger local slopes/variability, quantization and possible map/GPS errors; stops still masked; 1s export does not prove 1s terrain resolution |
| Explicit-quality hybrid | Could reject a source with clear discontinuities | No independent rule chooses correct local slopes; switching offsets can create false climbs. Not implemented or justified |

1. **Corrected values present?** A genuinely altered Strava-exported elevation series is present, consistent with correction; the export does not authenticate its internal algorithm.
2. **Profile differences?** Closure is −0.8m versus +105.4m; exported range 66.3–235.2m versus Garmin 64.4–304.8m; standardized gain is higher, not evidence of accuracy alone.
3. **20:34 anomaly?** The 13.4m rise becomes 0.6m over the same 10.14m of movement.
4. **Stops?** Usually improved, but two probable stops have greater exported variation; neither source should generate terrain workload while stationary.
5. **Climbing rate?** Strava is the more defensible provisional source for gross-profile consistency; local 30-sample accuracy remains unresolved. Retain both in sensitivity tests.
6. **Gradient?** No validated local winner: Strava fixes the obvious jump but produces more large 50m gradients on common support. Treat it as a provisional input with flags, and compare Garmin alternatives; neither is a pedaling-power estimate.
7. **Strava less reliable?** Two stationary counterexamples and the larger ~19:54 descent are warning regions; the latter is unresolved rather than a proven error.
8. **Future source?** Prefer quality-masked exported Strava for the primary terrain sensitivity arm; retain Garmin and a no-terrain arm. No switching hybrid yet.
9. **Ready for feasibility?** Yes for a limited, explicitly uncertain physiological feasibility experiment with terrain ablation; no for claiming reliable second-by-second workload or HR. No measured power/cadence, independent altitude reference, verified stop ground truth or fine terrain resolution is available. Coasting, braking and technical effort weaken speed/gradient-to-workload inference. No HR model was built.

## Reproduction and verification

```powershell
.\.venv\Scripts\python.exe scripts/compare_strava_garmin_elevation.py
```

Figure: `docs/images/strava_vs_garmin_elevation_2026-10-16.png`. Script: `scripts/compare_strava_garmin_elevation.py`. No dense intermediate files or new dependencies.

Executed: FIT integrity/decoding; GPX field completeness and degree/metre scale checks; unique ordered UTC times; exact 1s alignment and GPS consistency; existing stop-boundary agreement; stationary feature masking; an injected missing-value/segment-boundary support probe; independent polynomial refits of sampled real 50m gradient windows; minimum 50m distance and maximum 120s support checks; common-timestamp metrics; gain-minus-descent equals endpoint change; all 150 protected source/prior-output SHA-256 hashes unchanged; non-empty report/figure. Visual readability is checked separately after generation. No external APIs, source edits, HR estimates, model fitting, commits or uploads.
