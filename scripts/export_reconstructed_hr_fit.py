"""Export the completed HR CSV without ML, preserving the original FIT payload.

Use the pinned dependency source in ignored results/final/fit_tool_vendor.
No environment installation, model import, CSV edit or upload is performed.
"""

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/final"
SOURCE = ROOT / "data/mtb_edge.fit"
CSV = ROOT / "data/processed/hr_tcn_soft_average_2026-10-16.csv"
OUTPUT = FINAL / "hr_reconstructed_2026-10-16.fit"
ROUNDTRIP = FINAL / "mtb_edge_preserve_roundtrip.fit"
AUDIT = FINAL / "hr_fit_preserve_export_validation.json"
REPORT = ROOT / "docs/hr_fit_preserve_export_validation.md"
SIDECAR = FINAL / "hr_reconstructed_2026-10-16_provenance.md"
VERSION = "0.9.16"
ARCHIVE_SHA = "716b75b2fdfc66ca7b82df65f750c1427fd984c558284753f43e2bf8d2188f61"
VENDOR = FINAL / "fit_tool_vendor"
LIBRARY = VENDOR / ("fit_tool-" + VERSION)


def sha(path):
    """Return a file hash for immutable input and artifact provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependencies():
    """Load the pinned source locally, leaving all Python environments unchanged."""
    archive = VENDOR / ("fit_tool-%s.tar.gz" % VERSION)
    if not archive.exists():
        VENDOR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen("https://pypi.org/pypi/fit-tool/%s/json" % VERSION, timeout=30) as stream:
            metadata = json.load(stream)
        source = next(f for f in metadata["urls"]
                      if f["packagetype"] == "sdist")
        assert source["digests"]["sha256"] == ARCHIVE_SHA
        with urllib.request.urlopen(source["url"], timeout=30) as stream:
            raw = stream.read()
        assert hashlib.sha256(raw).hexdigest() == ARCHIVE_SHA
        archive.write_bytes(raw)
    assert sha(archive) == ARCHIVE_SHA
    if not LIBRARY.exists():
        with tarfile.open(archive) as bundle:
            base = VENDOR.resolve()
            for member in bundle.getmembers():
                destination = (base / member.name).resolve()
                assert base in destination.parents or destination == base
                assert not member.issym() and not member.islnk()
            bundle.extractall(VENDOR)
    sys.path.insert(0, str(LIBRARY))
    if importlib.util.find_spec("garmin_fit_sdk") is None:
        sdk = ROOT / ".venv/Lib/site-packages/garmin_fit_sdk/__init__.py"
        spec = importlib.util.spec_from_file_location("garmin_fit_sdk", sdk,
                                                      submodule_search_locations=[str(sdk.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["garmin_fit_sdk"] = module
        spec.loader.exec_module(module)


def decode(raw):
    """Independently decode ordered messages and verify both CRCs using Garmin."""
    from garmin_fit_sdk import CrcCalculator, Decoder, Stream
    header_size = raw[0]
    assert header_size in (12, 14) and raw[8:12] == b".FIT"
    assert len(raw) == header_size + int.from_bytes(raw[4:8], "little") + 2
    if header_size == 14:
        assert CrcCalculator.calculate_crc(
            raw, 0, 12) == int.from_bytes(raw[12:14], "little")
    assert CrcCalculator.calculate_crc(raw, 0, len(
        raw) - 2) == int.from_bytes(raw[-2:], "little")
    assert Decoder(Stream.from_byte_array(bytearray(raw))).check_integrity()
    ordered = []
    messages, errors = Decoder(Stream.from_byte_array(bytearray(raw))).read(
        merge_heart_rates=False, expand_sub_fields=False, expand_components=False,
        mesg_listener=lambda number, message: ordered.append((number, message.copy())))
    assert not errors, errors
    return ordered, messages


def same(left, right):
    """Compare decoded dictionaries including unknown fields and NaN payloads."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float) and np.isnan(left) and np.isnan(right):
        return True
    return left == right


def fields_on_wire(raw):
    """Use fit-tool's parsed wire layout, not a custom FIT parser, for byte ranges."""
    assert not raw.header.is_time_compressed, "Compressed edits need separate assessment"
    position = raw.source_offset + raw.header.size
    result = {}
    for field in raw.definition.field_definitions:
        result[field.field_id] = (position, field.size, field.base_type)
        position += field.size
    return result


def scope(message, index):
    """Select recorded sample timestamps inside verified summary start/end bounds."""
    start = pd.Timestamp(message["start_time"])
    if "end_time" in message:
        end = pd.Timestamp(message["end_time"])
    else:
        end = start + pd.Timedelta(seconds=message["total_elapsed_time"])
    return (index >= start) & (index <= end)


def zones(hr, mask, boundaries):
    """Count one-second supported intervals by their left HR and stored thresholds.

    Both endpoints must have supported HR and lie in the selected interval.
    A boundary belongs to the next higher zone, as checked against this source.
    No duration is assigned to unsupported HR and there is no zero fill.
    """
    valid = (mask[:-1] & mask[1:] & hr.iloc[:-1].notna().to_numpy()
             & hr.iloc[1:].notna().to_numpy())
    assert (np.diff(hr.index.asi8) == 1_000_000_000).all()
    bins = np.searchsorted(
        boundaries, hr.iloc[:-1].to_numpy()[valid], side="right")
    return np.bincount(bins, minlength=len(boundaries) + 1).astype(float).tolist()


def markdown(headers, rows):
    """Render a dependency-free concise Markdown audit table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(v).replace("|", "/")
                                for v in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    """Round-trip first, change only whitelisted HR fields, validate, then save."""
    assert not OUTPUT.exists(), "Export exists; do not overwrite it or the original"
    dependencies()
    from fit_tool import ConformanceLevel, EncodeMode, FitFile, SDK_VERSION
    from garmin_fit_sdk import Profile
    # Historical work, input activity and reconstructed CSV all remain untouched.
    protected = [p for folder in ("data", "docs", "scripts", "results")
                 for p in (ROOT / folder).rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts and VENDOR not in p.parents
                 and p not in {ROUNDTRIP, OUTPUT, AUDIT, REPORT, SIDECAR, Path(__file__).resolve()}]
    protected += [ROOT / n for n in ("README.md",
                                     ".gitignore", "AGENTS.md", "requirements.txt")]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    raw = SOURCE.read_bytes()
    original_order, original = decode(raw)
    fit = FitFile.from_bytes(raw)
    assert len(fit.wire_document.segments) == 1
    unchanged = fit.to_bytes(mode=EncodeMode.PRESERVE)
    assert unchanged == raw, "Unchanged preserve round-trip failed"
    roundtrip_order, _ = decode(unchanged)
    assert same(original_order, roundtrip_order)
    ROUNDTRIP.write_bytes(unchanged)
    print("Unchanged PRESERVE round-trip is byte-identical; official decode/CRCs pass", flush=True)
    wires = fit.wire_document.segments[0].records
    assert len(wires) == len(fit.records)
    entries, j = [], 0
    for i, (record, wire) in enumerate(zip(fit.records, wires)):
        assert record.is_definition == wire.header.is_definition
        if record.is_definition:
            continue
        number, message = original_order[j]
        assert number == record.message.global_id == wire.definition.global_id
        entries.append(dict(wire_index=i, data_index=j, number=number, sdk=message,
                            record=record, wire=wire, fields=fields_on_wire(wire)))
        j += 1
    assert j == len(original_order)
    native = [e for e in entries if e["number"] == 20]
    inventory = Counter()
    layouts = Counter()
    for e in native:
        slot = e["fields"].get(3)
        if slot is None:
            kind = "B_HR_field_omitted"
        else:
            offset, size, base_type = slot
            assert size == 1 and base_type & 31 == 2
            kind = "A_HR_field_invalid" if raw[offset] == 255 else "recorded_HR"
        inventory[kind] += 1
        layout = tuple((f.field_id, f.size, f.base_type)
                       for f in e["wire"].definition.field_definitions)
        layouts[layout] += 1
    assert inventory["recorded_HR"] == 1525 and inventory["A_HR_field_invalid"] == 8884
    assert inventory["B_HR_field_omitted"] == 0
    print("Binary HR inventory:", dict(inventory),
          "record layouts:", len(layouts), flush=True)
    table = pd.read_csv(CSV)
    table.index = pd.to_datetime(table.timestamp_utc, utc=True)
    index = pd.DatetimeIndex([e["sdk"]["timestamp"] for e in native])
    assert table.index.equals(
        index) and index.is_unique and index.is_monotonic_increasing
    assert np.all(np.diff(index.asi8) == 1_000_000_000)
    original_hr = pd.Series([e["sdk"].get("heart_rate", np.nan)
                            for e in native], index=index)
    assert np.allclose(original_hr, table.edge_hr,
                       equal_nan=True, atol=0, rtol=0)
    final_hr = table.final_combined_hr
    quantized = np.floor(final_hr + .5)
    assert quantized.dropna().between(1, 254).all()
    assert np.allclose(quantized[original_hr.notna()],
                       original_hr.dropna(), atol=0, rtol=0)
    allowed_bytes = set(range(len(raw) - 2, len(raw)))
    allowed_fields, changes = {}, []

    def change(entry, field_id, value, name):
        """Edit only an existing native field and track its exact permitted bytes."""
        if field_id not in entry["fields"]:
            return False
        position, size, _ = entry["fields"][field_id]
        field = entry["record"].message.get_field(field_id)
        assert field is not None and field.size == size
        if isinstance(value, list):
            field.set_values(value)
        else:
            field.set_value(0, value)
        if getattr(field, "is_unknown", False):
            # Garmin 21.218 identifies these slots, absent from fit-tool 21.212.
            # Its setter clears raw_bytes; refresh the explicit uint8 payload
            # through the library's decode API for PRESERVE validation.
            assert entry["number"] == 312 and field_id in (15, 16, 124)
            assert size == 1 and entry["fields"][field_id][2] & 31 == 2
            field.read_all_from_bytes(bytes([value]))
        assert field.size == size
        allowed_bytes.update(range(position, position + size))
        allowed_fields.setdefault(entry["data_index"], set()).add(name)
        changes.append(dict(message=entry["number"], field=name, field_id=field_id,
                            index=entry["sdk"].get("message_index"),
                            before=entry["sdk"].get(name), after=value))
        return True

    inserted, missing = 0, 0
    for i, entry in enumerate(native):
        if original_hr.iloc[i] == original_hr.iloc[i]:
            continue  # Original measured HR bytes are never edited.
        if pd.isna(quantized.iloc[i]):
            missing += 1
            continue  # Leave the genuine invalid 0xFF byte as it was.
        assert change(entry, 3, int(quantized.iloc[i]), "heart_rate")
        inserted += 1
    assert inserted == 8834 and missing == 50
    contexts, group_masks = {}, {}
    summary_entries = [e for e in entries if e["number"] in (18, 19, 312)]
    for entry in summary_entries:
        m, n = entry["sdk"], entry["number"]
        mask = scope(m, index)
        contexts[n, m.get("message_index", 0)] = mask
        if n == 312:
            group_masks.setdefault(m["split_type"], []).append(mask)
    summary_checks = []
    for entry in entries:
        n, message = entry["number"], entry["sdk"]
        if n not in (18, 19, 312, 313):
            continue
        if n == 313:
            masks = group_masks[message["split_type"]]
            assert len(masks) == message["num_splits"]
            mask = np.logical_or.reduce(masks)
        else:
            mask = contexts[n, message.get("message_index", 0)]
        values = quantized.loc[mask].dropna()
        assert len(values) > 0
        functions = {"avg_heart_rate": lambda v: int(np.floor(v.mean() + .5)),
                     "max_heart_rate": lambda v: int(v.max()),
                     "min_heart_rate": lambda v: int(v.min())}
        for field_id, profile in Profile["messages"][n]["fields"].items():
            name = profile["name"]
            if name in functions and field_id in entry["fields"]:
                value = functions[name](values)
                change(entry, field_id, value, name)
                summary_checks.append((entry["data_index"], name, value))
    zone_checks, original_zone_max_difference = [], 0.
    stored_boundaries = [e["sdk"]["hr_zone_high_boundary"] for e in entries
                         if e["number"] == 216
                         and "hr_zone_high_boundary" in e["sdk"]]
    assert stored_boundaries and all(
        b == stored_boundaries[0] for b in stored_boundaries)
    for entry in entries:
        if entry["number"] != 216:
            continue
        message = entry["sdk"]
        reference_number = {"session": 18, "lap": 19,
                            "split": 312}[message["reference_mesg"]]
        mask = contexts[reference_number, message["reference_index"]]
        # Split zone messages omit settings; the session/lap settings agree.
        boundaries = message.get("hr_zone_high_boundary", stored_boundaries[0])
        values = zones(quantized, mask, boundaries)
        assert len(values) == len(message["time_in_hr_zone"])
        old_computed = zones(original_hr, mask, boundaries)
        original_zone_max_difference = max(original_zone_max_difference,
                                           float(np.max(np.abs(np.asarray(old_computed) - message["time_in_hr_zone"]))))
        change(entry, 2, values, "time_in_hr_zone")
        zone_checks.append((entry["data_index"], values))
    # Do not request strict=True: that forces canonical rebuilding instead of preserve.
    preservation = fit.validate(
        levels={ConformanceLevel.WIRE, ConformanceLevel.PRESERVATION})
    assert not preservation.has_errors, preservation.errors
    modified = fit.to_bytes(mode=EncodeMode.PRESERVE)
    assert len(modified) == len(raw)
    changed_bytes = {i for i, (a, b) in enumerate(
        zip(raw, modified)) if a != b}
    assert changed_bytes <= allowed_bytes, "Unexpected non-HR byte change"
    assert modified[:raw[0]] == raw[:raw[0]
                                    ], "Original header/length/profile changed"
    exported_order, exported = decode(modified)
    assert len(exported_order) == len(original_order)
    non_hr_fields = 0
    for i, ((number, old), (new_number, new)) in enumerate(zip(original_order, exported_order)):
        assert number == new_number
        allowed = allowed_fields.get(i, set())
        for key in set(old) | set(new):
            if key not in allowed:
                assert key in old and key in new and same(
                    old[key], new[key]), (i, number, key)
                non_hr_fields += 1
    for i, field, value in summary_checks:
        assert exported_order[i][1][field] == value
    for i, values in zone_checks:
        assert np.allclose(
            exported_order[i][1]["time_in_hr_zone"], values, atol=.001, rtol=0)
    hr_records = pd.DataFrame(exported["record_mesgs"])
    assert pd.DatetimeIndex(pd.to_datetime(
        hr_records.timestamp, utc=True)).equals(index)
    assert np.allclose(hr_records.heart_rate, quantized,
                       equal_nan=True, atol=0, rtol=0)
    assert np.array_equal(hr_records.loc[original_hr.notna(
    ).to_numpy(), "heart_rate"], original_hr.dropna())
    assert hr_records.heart_rate.tail(50).isna().all()
    counts = Counter(n for n, _ in original_order)
    assert counts == Counter(n for n, _ in exported_order)
    assert counts[140] == 1
    assert len(exported["session_mesgs"]) == 1 and len(
        exported["lap_mesgs"]) == 1
    session, lap = exported["session_mesgs"][0], exported["lap_mesgs"][0]
    assert session["num_laps"] == 1 and session["first_lap_index"] == lap["message_index"] == 0
    assert session["avg_heart_rate"] == lap["avg_heart_rate"] == int(
        np.floor(quantized.mean() + .5))
    assert session["max_heart_rate"] == lap["max_heart_rate"] == int(
        quantized.max())
    OUTPUT.write_bytes(modified)
    # Reopen the actual on-disk file independently, not only the encoded buffer.
    reopened_order, _ = decode(OUTPUT.read_bytes())
    assert same(reopened_order, exported_order)
    assert all(sha(ROOT / p) == digest for p, digest in hashes.items())
    audit = dict(approach="fit-tool PRESERVE; no custom binary patch or TCX required",
                 status="validated_for_manual_import_test_not_platform_acceptance", fit_tool_version=VERSION,
                 fit_tool_profile=SDK_VERSION, dependency_archive_sha256=ARCHIVE_SHA,
                 source_sha256=sha(SOURCE), reconstruction_csv_sha256=sha(CSV), output_sha256=sha(OUTPUT),
                 unchanged_roundtrip_byte_identical=unchanged == raw,
                 inventory=dict(inventory), native_record_layouts=len(layouts),
                 data_messages=len(original_order), definition_messages=len(wires) - len(original_order),
                 message_counts={str(k): v for k, v in counts.items()},
                 bytes_before=len(raw), bytes_after=len(modified), changed_bytes=len(changed_bytes),
                 all_changes_whitelisted_HR_or_CRC=True, non_hr_fields_compared=non_hr_fields,
                 original_hr_preserved=1525, inserted_hr=inserted, unsupported_hr=missing,
                 session_average_hr=session["avg_heart_rate"], session_maximum_hr=session["max_heart_rate"],
                 quantization="nearest integer BPM, exact half ties upward", max_quantization_error=float((quantized - final_hr).abs().max()),
                 summary_edits=[c for c in changes if c["message"] != 20],
                 original_zone_max_difference_seconds=original_zone_max_difference,
                 preserved_file_sha256=hashes, fit_tool_validation="WIRE and PRESERVATION pass",
                 independent_decoder="Garmin FIT SDK 21.218.0: original/roundtrip/output/reopened output; header and file CRC checks pass",
                 no_ml_import_or_training=True, output=str(OUTPUT.relative_to(ROOT)))
    AUDIT.write_text(json.dumps(
        audit, indent=2, default=str), encoding="utf-8")
    report = ["# Faithful FIT export of the completed October 16 HR reconstruction", "",
              "**FIT export succeeded using fit-tool 0.9.16 PRESERVE.** No retraining or CSV edits. "
              "This supersedes the earlier official-Garmin-Encoder compatibility block; the old audit/report "
              "remain unchanged as historical evidence. No binary patch or TCX fallback was needed.", "",
              "## Binary inspection and preservation gate", "",
              "The original has two Record layouts, both with a one-byte uint8 HR field (native field 3). "
              "All 8,884 later missing-HR records already contain invalid 0xFF slots (case A); "
              "case B (field omitted) is zero. There is no mixture of A/B, despite two native layouts. "
              "The 1,525 initially measured HR records are separate and never edited.", "",
              "The unchanged decode/encode was **byte-for-byte identical** (%d bytes). "
              "Original and round-trip messages, order, fields, timestamps and values—including unknown "
              "message 140—were independently decoded and compared before any edits. "
              "fit-tool's documented preserve mode copies untouched records and reprojects dirty ones. "
              "`strict=True` would force canonical encoding, so validation was performed separately instead. "
              "[API](https://pypi.org/project/fit-tool/0.9.16/), "
              "[preservation policy](https://github.com/shaonianche/python_fit_tool/blob/main/docs/CAPABILITY_BOUNDARY.md)."
              % len(raw), "",
              "Pinned community-maintained library source was downloaded into ignored "
              "`results/final/fit_tool_vendor/` and verified against SHA-256 `%s`. "
              "It has no runtime dependencies; neither Conda deep nor TensorFlow was modified." % ARCHIVE_SHA, "",
              "## Changes and independent validation", "",
              markdown(["Check", "Result"], [
                  ["Original / exported size", "%d / %d bytes" %
                      (len(raw), len(modified))],
                  ["Data / definition messages", "%d / %d; counts and order unchanged" %
                      (len(original_order), len(wires) - len(original_order))],
                  ["Original measured HR", "1,525 values and bytes unchanged"],
                  ["Estimated HR inserted",
                      "8,834 at exact UTC timestamps; rounded to integer BPM"],
                  ["Unsupported HR", "50 original invalid slots remain invalid, never zero"],
                  ["Header and field definitions",
                      "Byte-identical; no timestamp compression or structural changes"],
                  ["All non-HR fields", "%d decoded field occurrences identical; all non-HR bytes identical" % non_hr_fields],
                  ["GPS, altitude, speed, distance, timestamps",
                      "Unchanged exactly, along with timer/lap/session structure"],
                  ["Proprietary and developer content",
                      "All bytes unchanged, including message 140 and unknown record fields"],
                  ["CRC / independent decoder",
                      "Garmin decoder and separate header/file CRC checks pass, including on-disk reopen"],
                  ["fit-tool validation", "WIRE and PRESERVATION pass"],
                  ["Session / lap average and maximum HR", "%d / %d BPM for both" %
                      (session["avg_heart_rate"], session["max_heart_rate"])],
                  ["Whitelisted byte differences", "%d; exclusively HR values/summaries and trailing CRC" % len(changed_bytes)]]), "",
              "Maximum integer-quantization error relative to the unchanged CSV: %.9f BPM. "
              "The activity file carries integer HR; the CSV retains its full precision." % audit[
                  "max_quantization_error"], "",
              "## HR summary definitions and limitations", "",
              "fit-tool's bundled profile lacks split HR fields 15/16/124, identified by the newer "
              "Garmin profile. Its UnknownField setter clears the preservation payload, so the script "
              "refreshes only those verified one-byte uint8 payloads through read_all_from_bytes. "
              "No library source or field definitions were modified. Their decoded results and all "
              "other bytes pass the same independent checks.", "",
              "Existing HR mean/max/min slots in session, lap, splits and split summaries were recomputed "
              "from available exported integer HR samples. Mean is the equal-weight mean on this verified "
              "continuous 1Hz recording, rounded nearest BPM (half upward). Session/lap ranges use start_time "
              "plus total_elapsed_time, not their original timestamp field (which is activity start). "
              "Split ranges use their explicit inclusive start/end timestamps; split-summary groups use "
              "the union of their referenced split_type and verified num_splits.", "",
              "All 31 existing time-in-HR-zone arrays were updated from consecutive one-second intervals "
              "with both HR endpoints supported and inside the relevant scope. The left sample determines "
              "the zone; reaching a stored boundary enters the next zone. Missing HR is excluded. "
              "Split messages omit zone boundaries; they use the identical stored session/lap boundaries. "
              "Zone boundaries, configured maximum/resting/threshold HR, sport settings and non-HR zone "
              "fields remain unchanged. This is an explicit one-second calculation, not a recreation of "
              "Garmin's subsecond firmware integration. Applied to the original data, its largest "
              "zone-duration difference from original arrays is %.3fs across the 31 messages. "
              "New arrays are independently verified to 0.001s after decode." % original_zone_max_difference, "",
              "Calories, training effect/load, HRV and all proprietary data remain original—not recalculated "
              "from estimated HR. Private fields may contain historical physiological summaries whose "
              "meaning cannot be established; they are preserved rather than fabricated or silently stripped. "
              "Exact original HR is retained, including any sensor artifacts. Estimated HR is not a measured signal.", "",
              "Message inventory:", "", markdown(
                  ["Global message", "Count before/after"], sorted(counts.items())), "",
              "## Artifacts, provenance and manual import", "",
              "- Original backup remains `data/mtb_edge.fit`, SHA-256 `%s`." % audit["source_sha256"],
              "- Unchanged round-trip: `results/final/mtb_edge_preserve_roundtrip.fit`.",
              "- Export: `results/final/hr_reconstructed_2026-10-16.fit`.",
              "- Field-level edit and validation audit: `results/final/hr_fit_preserve_export_validation.json`.",
              "- Provenance sidecar: `results/final/hr_reconstructed_2026-10-16_provenance.md`.", "",
              "The file is suitable for a **manual import test**, not guaranteed Garmin/Strava acceptance. "
              "Original activity timestamps/identifiers were preserved; an existing activity may cause "
              "duplicate rejection or a second activity. Do not shift dates, delete originals or assume import "
              "updates an existing activity. If importing, label it as reconstructed HR in the activity description. "
              "No account connection, upload, deletion, commit or push was performed. Outputs/dependency are "
              "under the existing ignored results/final directory.", "",
              "```powershell", "conda run --no-capture-output -n deep python scripts/export_reconstructed_hr_fit.py", "```", "",
              "The script refuses to overwrite an existing final FIT. It contains no model or training imports. "
              "All prior files and the reconstructed CSV retain their original hashes.", ""]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    SIDECAR.write_text("# Reconstructed HR provenance\n\n"
                       "This is a separate copy of the original Garmin Edge October 16, 2026 activity. "
                       "The first 1,525 recorded HR samples are original; 8,834 later HR samples are TCN/soft-average "
                       "estimates from the completed CSV, rounded to integer BPM; 50 unsupported samples remain missing. "
                       "Estimated HR must not be described as sensor-recorded or recovered true HR. "
                       "HR summaries are recalculated as documented in docs/hr_fit_preserve_export_validation.md; "
                       "calories/training effect/HRV/private metrics remain original.\n\n"
                       "Source SHA-256: %s\n\nCSV SHA-256: %s\n\nExport SHA-256: %s\n"
                       % (audit["source_sha256"], audit["reconstruction_csv_sha256"], audit["output_sha256"]), encoding="utf-8")
    assert all(sha(ROOT / p) == digest for p, digest in hashes.items())
    assert all(p.stat().st_size > 0 for p in (
        OUTPUT, ROUNDTRIP, AUDIT, REPORT, SIDECAR))
    print("FIT EXPORT VALIDATED:", OUTPUT, flush=True)
    print("Preserved original HR / inserted / missing:", 1525, inserted, missing)
    print("Session/lap HR mean/max:",
          session["avg_heart_rate"], session["max_heart_rate"])
    print("Changed bytes:", len(changed_bytes),
          "decoded unchanged non-HR fields:", non_hr_fields)
    print("Original zone calculation max difference:",
          original_zone_max_difference)


if __name__ == "__main__":
    main()
