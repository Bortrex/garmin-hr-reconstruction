# Watch wellness HR audit — 2026-10-16

**158 real watch observations recovered during the ride; stored spacing is 60 or 120 seconds.** Automatic cycling detection is present, but no sub-minute wellness HR is exposed.

## Sources and extraction

Inspected all 84 FIT files recursively (81 in `data/watch_wellness/`, plus three activity files); no archives or other source files. All passed size/header/CRC integrity and decoding checks. Device identity, not folder name, determines provenance: 29 files identify the FR265 and 55 identify the Edge (including the three activities). The nominal watch folder therefore includes 52 Edge files.

The watch HR series is `monitoring` (global message 55), `heart_rate` (field 27, BPM), with `timestamp_16` (field 26). The ride is entirely in **`data/watch_wellness/494532015728_WELLNESS.fit`**: 158 samples, 2026-10-16T19:33:00+02:00 through 2026-10-16T22:25:00+02:00. This file has 166 HR samples overall, 19:29–22:29. Creator is `fr265_large` (product 4257), firmware 29.05.

The official Garmin SDK decodes fields/components with HR merging disabled. Monitoring timestamps are accumulated in original monitoring-message order within each file: `anchor += (timestamp_16 - (anchor & 65535)) & 65535`; full monitoring timestamps reset the anchor. The epoch is 1989-12-31 UTC. This handles the 16-bit wrap and is distinct from compressed record headers. SDK 21.218.0 rejects compressed timestamp headers; no such decoding errors occurred here. No standard `hr` array/event-time messages were found. Unknown message numbers and packed arrays remain preserved in the decoded JSON; their undocumented semantics cannot be ruled out as containing additional proprietary information.

No valid watch duplicates or conflicting watch timestamps were found. `494531572096_WELLNESS.fit` is **Edge wellness**, with 18 valid HR values (19:34–19:59), including 15 differing values at timestamps shared with the watch. It is exported separately and never merged into the watch series. Zero-BPM records in Edge wellness are retained as invalid; the watch has no zero-BPM observations. Resting-HR summaries (`monitoring_hr_data`, 35 watch messages), HRV values/status, SpO2, stress, respiration and session averages are not additional instantaneous HR samples.

## Time window and sampling

All times below are Europe/Brussels (UTC+02:00). Edge session start: **2026-10-16T19:32:22+02:00**. Start + elapsed duration gives **2026-10-16T22:25:50.271000+02:00**; final record and timer stop are 22:25:50. The session/lap summary timestamp itself repeats the start, so it was not used as the end. The ride does not cross midnight. Watch coverage is 00:01 on October 16 through exactly 00:00 on October 17; daily statistics use [00:00, next 00:00), excluding that last midnight sample. The plot shows 18:30–24:00.

| Period | Samples | Median / mean (s) | Min–max (s) | 60s / 120s intervals | Gaps >120s (elapsed seconds) |
|---|---:|---:|---:|---:|---:|
| before | 967 | 60 / 72.73 | 60–360 | 84.37% / 12.32% | 32 (7080s) |
| before_evening_only | 56 | 60 / 67.64 | 60–120 | 87.27% / 12.73% | 0 (0s) |
| during | 158 | 60 / 65.73 | 60–120 | 90.45% / 9.55% | 0 (0s) |
| after | 84 | 60 / 67.23 | 60–300 | 91.57% / 7.23% | 1 (300s) |

Before/after are the portions of this calendar day outside the session; evening-only begins 18:30. Intervals are calculated within each period, excluding boundary-crossing pairs. Significant gaps are defined as >120s (elapsed between recorded endpoints, not estimated missing duration). All watch proportions near 1, 5, 15 and 30 seconds are 0%, with “near” defined as ±1s.

| Period | P5 / P25 / P50 / P75 / P95 / P99 (s) | Full interval distribution (seconds: count) |
|---|---|---|
| before | 60.0 / 60.0 / 60.0 / 60.0 / 120.0 / 240.0 | 60: 815, 120: 119, 180: 19, 240: 8, 300: 1, 360: 4 |
| during | 60.0 / 60.0 / 60.0 / 60.0 / 120.0 / 120.0 | 60: 142, 120: 15 |
| after | 60.0 / 60.0 / 60.0 / 60.0 / 120.0 / 152.4 | 60: 76, 120: 6, 300: 1 |

During the ride, 142 intervals are 60s and 15 are 120s; none exceeds 120s. All timestamps lie on minute boundaries. This describes **stored samples**, not optical sensor sampling frequency. The public profile labels the field BPM but does not establish whether it is an instantaneous reading or a filtered/averaged value, nor its averaging window. Timestamp precision does not prove physiological precision.

## Automatic activity detection and intensity

The watch has explicit `event=auto_activity_detect` records classified as cycling, consistent with Move IQ. Intervals below use `start_timestamp + data duration in minutes`; event timestamps are reporting times, 15 minutes after the classified end for these events. Using reporting time as the end would incorrectly extend the exercise windows.

| Detected cycling (local) | Reported | Samples | Mean spacing (s) | Median before / during / after (s)* |
|---|---|---:|---:|---|
| 17:56:00–18:12:00 | 18:27:00 | 16 | 60.00 | 60 / 60 / 60 |
| 19:15:00–20:13:00 | 20:28:00 | 52 | 67.06 | 60 / 60 / 60 |
| 20:15:00–21:46:00 | 22:01:00 | 84 | 65.06 | 60 / 60 / 60 |
| 21:47:00–22:25:00 | 22:40:00 | 35 | 65.29 | 60 / 60 / 60 |

*Before/after use adjacent 15-minute windows; they may include other classified activity. Complete event statistics are in `auto_activity_intervals.json`. The final ride classification is in `494552312010_WELLNESS.fit`, although the HR samples it describes are in the earlier ride file.

Monitoring includes activity classifications, packed type/intensity components, active time, cycles and moderate/vigorous activity-minute records. These are preserved in `watch_monitoring.csv`; they are not HR observations. For interval endpoints with BPM <100, 100–129 and ≥130, mean spacing is respectively 72.48s, 66.43s, 62.37s; all three groups have 60s median and minimum. Higher BPM is associated with fewer skipped minute slots, but there is **no faster stored cadence**, and this single-day association cannot establish causation by Move IQ.

## Edge comparison

The Edge has 10409 activity records and **1525 consecutive 1Hz HR observations**, 19:32:22–19:57:46. The first missing HR record is **19:57:47**; the remaining 8,884 activity records lack HR. This is an observed recording cutoff, not proof of the exact radio failure instant; no public `comm_timeout` event was found. The watch retains **137 samples after that cutoff**, through 22:25 during the ride.

Exact timestamp matching gives 21 watch–Edge pairs (no time shift or interpolation). Mean watch minus Edge is -2.29 BPM, median -1 BPM, mean absolute difference 13.90 BPM, range -44 to 29 BPM. Both use UTC timestamps, but unknown averaging/processing windows mean timestamp equality does not imply identical underlying measurement windows. The small overlap does not establish a stable correction or clock offset.

## Conclusions and limitations

1. Ride watch HR is in `494532015728_WELLNESS.fit` only.
2. Recovered 158 valid watch HR observations during the precise Edge session.
3. Actual spacing: 60s (90.45%) or 120s (9.55%); median 60s, mean 65.73s.
4. Auto-detected cycling is documented, but neither it nor elevated HR produces sub-minute stored samples in the decoded public fields.
5. Compared with a two-minute display, these 158 points offer roughly 1.8 times as many observations over the ride (about 87 two-minute slots). That is useful additional temporal detail, not a hidden high-resolution stream. No Garmin Connect graph/API export was supplied to verify its exact averaging or information loss.
6. These observations can anchor a future coarse HR reconstruction, especially after the Edge cutoff. They cannot recover true second-by-second dynamics or missing peaks, and any later denser reconstruction must be labeled estimated. Unknown averaging and the short, imperfect Edge overlap limit calibration.

## Outputs and reproduction

- Figure: `docs/images/watch_wellness_hr_2026-10-16.png` (3300×1760).
- Full per-file/message inventory, fields, resolved UTC ranges and HR counts: `data/processed/fit_inventory.csv`.
- Every extracted scalar HR, with timestamp, device source, file and message index: `data/processed/hr_all_sources.csv`.
- Ride watch HR: `data/processed/watch_hr_ride_2026-10-16.csv`.
- Interval statistics (including hourly and per-file), gaps, exact Edge pairs, monitoring and automatic events: `data/processed/`.
- Lossless SDK-decoded watch-folder messages, including unknown fields: `data/processed/watch_decoded_messages.json`.
- Original-file SHA-256 manifest: `data/processed/source_sha256.json`; originals rechecked unchanged after analysis.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-audit.txt
.\.venv\Scripts\python.exe scripts/audit_watch_wellness_hr.py
```

No interpolation, synthetic HR, model training, FIT modification or uploads were performed. The script fails explicitly on unsupported inputs/HR encodings rather than silently omitting them.

## Format references

- [Official Garmin Python FIT SDK](https://github.com/garmin/fit-python-sdk), version 21.218.0 used.
- [Garmin FIT engineer explanation of timestamp_16](https://forums.garmin.com/developer/fit-sdk/f/discussion/424427/convert-timestamp-16-to-date).
- [Garmin Event message profile](https://github.com/garmin/fit-csharp-sdk/blob/main/Dynastream/Fit/Profile/Mesgs/EventMesg.cs): automatic-detection start and duration fields.
