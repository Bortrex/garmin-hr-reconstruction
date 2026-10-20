# Faithful FIT export of the completed October 16 HR reconstruction

**FIT export succeeded using fit-tool 0.9.16 PRESERVE.** No retraining or CSV edits. This supersedes the earlier official-Garmin-Encoder compatibility block; the old audit/report remain unchanged as historical evidence. No binary patch or TCX fallback was needed.

## Binary inspection and preservation gate

The original has two Record layouts, both with a one-byte uint8 HR field (native field 3). All 8,884 later missing-HR records already contain invalid 0xFF slots (case A); case B (field omitted) is zero. There is no mixture of A/B, despite two native layouts. The 1,525 initially measured HR records are separate and never edited.

The unchanged decode/encode was **byte-for-byte identical** (565782 bytes). Original and round-trip messages, order, fields, timestamps and values—including unknown message 140—were independently decoded and compared before any edits. fit-tool's documented preserve mode copies untouched records and reprojects dirty ones. `strict=True` would force canonical encoding, so validation was performed separately instead. [API](https://pypi.org/project/fit-tool/0.9.16/), [preservation policy](https://github.com/shaonianche/python_fit_tool/blob/main/docs/CAPABILITY_BOUNDARY.md).

Pinned community-maintained library source was downloaded into ignored `results/final/fit_tool_vendor/` and verified against SHA-256 `716b75b2fdfc66ca7b82df65f750c1427fd984c558284753f43e2bf8d2188f61`. It has no runtime dependencies; neither Conda deep nor TensorFlow was modified.

## Changes and independent validation

| Check | Result |
|---|---|
| Original / exported size | 565782 / 565782 bytes |
| Data / definition messages | 24945 / 40; counts and order unchanged |
| Original measured HR | 1,525 values and bytes unchanged |
| Estimated HR inserted | 8,834 at exact UTC timestamps; rounded to integer BPM |
| Unsupported HR | 50 original invalid slots remain invalid, never zero |
| Header and field definitions | Byte-identical; no timestamp compression or structural changes |
| All non-HR fields | 151353 decoded field occurrences identical; all non-HR bytes identical |
| GPS, altitude, speed, distance, timestamps | Unchanged exactly, along with timer/lap/session structure |
| Proprietary and developer content | All bytes unchanged, including message 140 and unknown record fields |
| CRC / independent decoder | Garmin decoder and separate header/file CRC checks pass, including on-disk reopen |
| fit-tool validation | WIRE and PRESERVATION pass |
| Session / lap average and maximum HR | 122 / 176 BPM for both |
| Whitelisted byte differences | 9107; exclusively HR values/summaries and trailing CRC |

Maximum integer-quantization error relative to the unchanged CSV: 0.499988903 BPM. The activity file carries integer HR; the CSV retains its full precision.

## HR summary definitions and limitations

fit-tool's bundled profile lacks split HR fields 15/16/124, identified by the newer Garmin profile. Its UnknownField setter clears the preservation payload, so the script refreshes only those verified one-byte uint8 payloads through read_all_from_bytes. No library source or field definitions were modified. Their decoded results and all other bytes pass the same independent checks.

Existing HR mean/max/min slots in session, lap, splits and split summaries were recomputed from available exported integer HR samples. Mean is the equal-weight mean on this verified continuous 1Hz recording, rounded nearest BPM (half upward). Session/lap ranges use start_time plus total_elapsed_time, not their original timestamp field (which is activity start). Split ranges use their explicit inclusive start/end timestamps; split-summary groups use the union of their referenced split_type and verified num_splits.

All 31 existing time-in-HR-zone arrays were updated from consecutive one-second intervals with both HR endpoints supported and inside the relevant scope. The left sample determines the zone; reaching a stored boundary enters the next zone. Missing HR is excluded. Split messages omit zone boundaries; they use the identical stored session/lap boundaries. Zone boundaries, configured maximum/resting/threshold HR, sport settings and non-HR zone fields remain unchanged. This is an explicit one-second calculation, not a recreation of Garmin's subsecond firmware integration. Applied to the original data, its largest zone-duration difference from original arrays is 1.000s across the 31 messages. New arrays are independently verified to 0.001s after decode.

Calories, training effect/load, HRV and all proprietary data remain original—not recalculated from estimated HR. Private fields may contain historical physiological summaries whose meaning cannot be established; they are preserved rather than fabricated or silently stripped. Exact original HR is retained, including any sensor artifacts. Estimated HR is not a measured signal.

Message inventory:

| Global message | Count before/after |
|---|---|
| 0 | 1 |
| 2 | 1 |
| 3 | 1 |
| 7 | 1 |
| 12 | 1 |
| 13 | 1 |
| 18 | 1 |
| 19 | 1 |
| 20 | 10409 |
| 21 | 2 |
| 22 | 6 |
| 23 | 7 |
| 34 | 1 |
| 49 | 1 |
| 78 | 3267 |
| 79 | 1 |
| 104 | 135 |
| 140 | 1 |
| 141 | 1 |
| 147 | 5 |
| 162 | 37 |
| 216 | 31 |
| 288 | 7 |
| 312 | 29 |
| 313 | 2 |
| 325 | 562 |
| 326 | 18 |
| 327 | 3 |
| 394 | 2 |
| 526 | 1 |
| 534 | 10409 |

## Artifacts, provenance and manual import

- Original backup remains `data/mtb_edge.fit`, SHA-256 `26757bcb50693a7a95b7ed60a32b3b8eb697f678e43a7c8d2cb4069f19fd5ef3`.
- Unchanged round-trip: `results/final/mtb_edge_preserve_roundtrip.fit`.
- Export: `results/final/hr_reconstructed_2026-10-16.fit`.
- Field-level edit and validation audit: `results/final/hr_fit_preserve_export_validation.json`.
- Provenance sidecar: `results/final/hr_reconstructed_2026-10-16_provenance.md`.

The file is suitable for a **manual import test**, not guaranteed Garmin/Strava acceptance. Original activity timestamps/identifiers were preserved; an existing activity may cause duplicate rejection or a second activity. Do not shift dates, delete originals or assume import updates an existing activity. If importing, label it as reconstructed HR in the activity description. No account connection, upload, deletion, commit or push was performed. Outputs/dependency are under the existing ignored results/final directory.

```powershell
conda run --no-capture-output -n deep python scripts/export_reconstructed_hr_fit.py
```

The script refuses to overwrite an existing final FIT. It contains no model or training imports. All prior files and the reconstructed CSV retain their original hashes.
