# Final October 16, 2026 MTB HR reconstruction

**Selected method: ML-B.** See [the ML evaluation](hr_ml_residual_evaluation.md) for the outdoor holdouts, target overlap and fixed selection criteria.

- Mean of the **supported combined recorded/estimated sequence**: **120.73 BPM**.
- Supported peak: **172.00 BPM**; missing short peaks cannot be recovered or ruled out.
- Coverage: **10359/10409 timestamps (99.52%)**.
- Original Edge observations: 1525, preserved exactly. Estimated rows: 8834, mean 124.92 BPM. Original watch summaries: 158, preserved separately and exactly.

These are descriptive statistics of supported one-second rows, not a measured full-ride physiological mean or guaranteed true peak. The last 50 seconds remain missing because there is no following HR anchor. No personalized HR zones are invented.

Continuity check on estimated rows and the handoff: P99 absolute 1s change 0.922 BPM, maximum 1.925 BPM, 0 changes >5 BPM. The combined sequence's larger 22.0 BPM maximum step belongs to original preserved Edge data, not an injected reconstruction jump. Numerical continuity does not validate true short fluctuations.

## Observations, handoff and provenance

Sources: `data/mtb_edge.fit`, `data/VTT_Apero.gpx`, `data/processed/watch_hr_ride_2026-10-16.csv`. Original Edge timestamps define the exact 1Hz grid; all calculations retain floating-point precision, UTC internally and Brussels time in the plot.

Last recorded Edge HR: 2026-10-16T17:57:46+00:00 (19:57:46 Brussels), 101 BPM. After overlap evaluation and model selection, this **real** observation becomes the left boundary for the first reconstructed interval, with the next real watch summary (94 BPM, 14s later). This prevents an arbitrary sensor-switch jump without fabricating an anchor, shifting time, calibrating the watch, or modifying recorded values. The handoff remains an explicitly flagged cross-sensor interval. No withheld Edge observation was used during the overlap evaluation.

After disconnection, interpolation uses original watch summaries with a maximum 180s gap, never extrapolating. Their storage/averaging mechanism is uncertain: treating them as pseudo-point constraints is a practical baseline assumption, not evidence of exact instantaneous HR. Even at a watch timestamp, a final value after disconnection is labelled an **estimate**. When Edge/watch coincide, Edge retains precedence and the watch observation stays unchanged in its separate column. No stop sets HR to zero or resets history.

CSV fields: `timestamp_utc`, `edge_hr`, `watch_hr`, `interpolation_baseline`, `ml_estimate`, `final_hr`, observation/estimate booleans, `source_status`, and `quality_indicator`. `ml_estimate` contains ML-B as a comparison/selected correction. Baseline/ML columns are post-cutoff candidates; they are intentionally empty before the last Edge observation. Quality labels mark terrain support, stationary evidence and the cross-sensor handoff; supported terrain does not certify HR accuracy. Missing HR is never encoded as zero.

Status values: `recorded_edge`, `estimated_at_watch_anchor`, `estimated_between_anchors`, `missing_unsupported`. Separate `watch_observed` preserves the distinction between an actual stored summary and a modelled instantaneous HR value.

## Reproduction and checks

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_hr_ml_residual.py
```

Outputs: `data/processed/hr_reconstructed_2026-10-16.csv`, `docs/images/hr_reconstructed_2026-10-16.png`, this report. Verified exact Edge grid, original Edge/watch preservation, supported handoff, missing terminal tail, mean/count identities and unchanged source/prior-analysis hashes. Figures are inspected after generation. This work was developed with AI assistance; empirical results come from the executed script, not inferred or fabricated physiology. No activity FIT is modified or uploaded.
