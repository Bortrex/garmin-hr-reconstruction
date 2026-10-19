"""Diagnose wellness HR summaries against a same-sensor activity stream.

Run: .venv/Scripts/python scripts/diagnose_watch_anchors.py
No source timestamps are shifted and no HR reconstruction is produced.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import hashlib
import json
from collections import Counter
from pathlib import Path

from garmin_fit_sdk import Decoder, Stream
import matplotlib
import numpy as np
import pandas as pd
import pytz

from audit_watch_wellness_hr import monitoring_times
from reconstruct_hr_baseline import clean_observations

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-10-12"
ZONE = pytz.timezone("Europe/Brussels")
STEM = f"watch_anchor_diagnostic_{DATE}"
WIDTHS = (15, 30, 60, 120)
OFFSETS = range(-60, 61, 5)


def decode(path):
    """Read a FIT with the existing SDK and require integrity and decode success."""
    if not Decoder(Stream.from_file(str(path))).check_integrity():
        raise ValueError(f"FIT integrity failure: {path}")
    messages, errors = Decoder(Stream.from_file(str(path))).read(
        merge_heart_rates=False
    )
    if errors:
        raise ValueError(f"{path}: {errors}")
    if messages.get("hr_mesgs"):
        raise ValueError("Array HR requires explicit event-time handling")
    return messages


def distribution(series):
    """Count actual timestamp spacings without assuming a recording cadence."""
    seconds = np.diff(series.index.asi8) / 1e9
    return {str(k): int(v) for k, v in
            pd.Series(seconds).value_counts().sort_index().items()}


def local(timestamp):
    """Format an aware timestamp in the user's local timezone."""
    return pd.Timestamp(timestamp).tz_convert(ZONE).isoformat()


def load_watch(folder):
    """Reuse monitoring timestamp decoding; retain provenance and other records."""
    rows, inventory, metadata, events = [], [], {}, []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".fit":
            raise ValueError(f"Inspect unsupported input: {path}")
        messages = decode(path)
        identity = messages.get("file_id_mesgs", [{}])[0]
        product = identity.get("garmin_product")
        metadata[path.name] = {
            key: messages.get(key, []) for key in
            ("file_id_mesgs", "device_info_mesgs", "monitoring_hr_data_mesgs",
             "timestamp_correlation_mesgs", "ohr_settings_mesgs")
        }
        for kind, records in messages.items():
            inventory.append({
                "file": path.name, "product": product, "type": kind,
                "count": len(records),
                "fields": ", ".join(sorted({str(k) for r in records for k in r})),
            })
        if product != "fr265_large":
            continue
        for i, row in enumerate(monitoring_times(
                messages.get("monitoring_mesgs", []))):
            if "heart_rate" in row:
                rows.append({
                    "timestamp": row["timestamp"], "bpm": row["heart_rate"],
                    "source_file": str(path.relative_to(ROOT)),
                    "message_type": "monitoring", "message_index": i,
                })
        events.extend({"source_file": path.name, **e}
                      for e in messages.get("event_mesgs", []))
    raw = pd.DataFrame(rows)
    watch, invalid, duplicates = clean_observations(raw)
    return watch, raw, inventory, metadata, events, (invalid, duplicates)


def load_activity(path):
    """Extract recorded HR without imputing absent timestamps or readings."""
    messages = decode(path)
    records = pd.DataFrame(messages["record_mesgs"])
    series, invalid, duplicates = clean_observations(
        records.rename(columns={"heart_rate": "bpm"})
    )
    return series, records, messages, (invalid, duplicates)


def window_offsets(alignment, width, shift=0):
    """Specify inclusive integer-second support relative to an anchor.

    Trailing W: t-W+1 through t; leading W: t+1 through t+W.
    Centered W: t-floor(W/2) through t+ceil(W/2)-1.
    Positive shift samples the reference later, never changes source times.
    """
    if alignment == "instant":
        return np.array([shift])
    if alignment == "trailing":
        return np.arange(-width + 1, 1) + shift
    if alignment == "leading":
        return np.arange(1, width + 1) + shift
    return np.arange(-(width // 2), width - width // 2) + shift


def reference_values(reference, times, specification):
    """Average only fully observed windows: missing seconds make a value NaN."""
    offsets = window_offsets(*specification).astype(np.int64)
    target_ns = times.asi8[:, None] + offsets[None, :] * 1_000_000_000
    values = reference.reindex(pd.to_datetime(
        target_ns.ravel(), utc=True
    )).to_numpy().reshape(len(times), len(offsets))
    complete = np.isfinite(values).all(axis=1)
    output = np.full(len(times), np.nan)
    output[complete] = values[complete].mean(axis=1)
    return pd.Series(output, index=times)


def label(specification):
    """Describe a candidate's shape, width, and reference time displacement."""
    alignment, width, shift = specification
    name = ("Point record" if shift else "Same-time record") if alignment == "instant" else (
        f"{alignment.capitalize()} {width}s"
    )
    return name if not shift else f"{name} ({shift:+d}s)"


def metrics(watch, candidate):
    """Report signed errors as wellness minus reference representation."""
    paired = pd.concat([watch, candidate], axis=1).dropna()
    if paired.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan,
                "correlation": np.nan, "max_abs_error": np.nan}
    a, b = paired.iloc[:, 0], paired.iloc[:, 1]
    error = a - b
    corr = a.corr(b) if len(a) > 2 and a.std() > 0 and b.std() > 0 else np.nan
    return {"n": len(a), "mae": error.abs().mean(),
            "rmse": np.sqrt(np.mean(error ** 2)), "bias": error.mean(),
            "correlation": corr, "max_abs_error": error.abs().max()}


def select_candidates(watch, reference, split):
    """Select with training only; reserve an identical fully covered anchor set."""
    times = watch.index
    # Every permitted window is within this envelope. Use the same envelope
    # for all candidates so coverage differences cannot change rankings.
    envelope = reference_values(reference, times, ("centered", 361, 0))
    common = envelope.notna()
    train = common & (times + pd.Timedelta(seconds=180) < split)
    held = common & (times - pd.Timedelta(seconds=180) >= split)
    assert train.sum() >= 10 and held.sum() >= 10
    specifications = {"instant": ("instant", 1, 0)}
    for alignment in ("trailing", "centered", "leading"):
        for width in WIDTHS:
            specifications[f"{alignment}_{width}"] = (alignment, width, 0)
    predictions = pd.DataFrame({
        name: reference_values(reference, times, spec)
        for name, spec in specifications.items()
    })
    train_mae = {
        name: metrics(watch[train], predictions.loc[train, name])["mae"]
        for name in specifications
    }
    average = min((n for n in specifications if n != "instant"),
                  key=train_mae.get)
    main_names = list(specifications)
    offset_rows, selected_offsets = [], []
    # Limited displacement search: same-time and the TRAIN-selected averaging
    # shape only, in five-second steps. Never choose using held-out outcomes.
    for base in ("instant", average):
        choices = []
        for shift in OFFSETS:
            spec = (*specifications[base][:2], shift)
            key = base if shift == 0 else f"{base}_shift_{shift:+d}"
            if key not in predictions:
                specifications[key] = spec
                predictions[key] = reference_values(reference, times, spec)
            score = metrics(watch[train], predictions.loc[train, key])
            offset_rows.append({"candidate": key, "base": base,
                                "shift_seconds": shift, **score})
            choices.append((score["mae"], abs(shift), key))
        winner = min(choices)[2]
        selected_offsets.append(winner)
        if winner not in main_names:
            main_names.append(winner)
    selected = min(main_names, key=lambda n: metrics(
        watch[train], predictions.loc[train, n])["mae"])
    rows = []
    for name in main_names:
        for part, mask in (("exploratory", train), ("held_out", held)):
            score = metrics(watch[mask], predictions.loc[mask, name])
            assert score["n"] == int(mask.sum())
            rows.append({"candidate": name, "label": label(specifications[name]),
                         "part": part, **score})
    return (predictions, specifications, main_names, selected, average,
            train, held, pd.DataFrame(rows), pd.DataFrame(offset_rows))


def regimes_and_blocks(watch, reference, predictions, names, train, held):
    """Describe errors by fixed HR regimes and contiguous blocks, not IID tests."""
    before = reference_values(reference, watch.index, ("instant", 1, -30))
    after = reference_values(reference, watch.index, ("instant", 1, 30))
    delta = after - before
    threshold = predictions.loc[train, "instant"].quantile(.75)
    masks = {"increasing": delta > 5, "decreasing": delta < -5,
             "stable": delta.abs() <= 5,
             "high_hr": predictions.instant >= threshold}
    rows, blocks = [], []
    for part, eligible in (("exploratory", train), ("held_out", held)):
        for name in names:
            for regime, mask in masks.items():
                keep = eligible & mask
                rows.append({"candidate": name, "part": part,
                             "regime": regime,
                             **metrics(watch[keep], predictions.loc[keep, name])})
            for block, values in watch[eligible].groupby(
                    watch[eligible].index.floor("5min")):
                blocks.append({"candidate": name, "part": part,
                               "block_start_utc": block,
                               **metrics(values, predictions.loc[values.index, name])})
    return pd.DataFrame(rows), pd.DataFrame(blocks), threshold


def make_figure(watch, reference, predictions, specification, selected,
                scores, split, train, path):
    """Plot observations, calculated summaries, and separately labeled scores."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.25]})
    for ax in axes[:2]:
        # NaN reindexing breaks lines across absent seconds; never fills HR.
        observed = reference.reindex(pd.date_range(
            reference.index[0], reference.index[-1], freq="s"))
        ax.plot(observed.index, observed, color=".55", lw=.8,
                label="MyWhoosh recorded HR (same FR265 sensor)")
        ax.scatter(watch.index, watch, s=28, color="#007c91", zorder=4,
                   label="FR265 wellness anchors (stored)")
        ax.scatter(watch.index, predictions[selected], s=45, marker="x",
                   color="#d46b14", zorder=5,
                   label=f"Calculated: {label(specification)}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZONE))
        ax.set_ylabel("HR (BPM)")
        ax.grid(alpha=.2)
    axes[0].axvline(split, color="#753a93", ls="--", label="60%/40% split")
    axes[0].set_xlim(reference.index[0], reference.index[-1])
    axes[0].set_title(
        "Complete virtual ride — candidate selected on first 60% only")
    axes[0].legend(loc="lower right", fontsize=8, ncol=2)
    # Choose detail by reference range in TRAINING only, never by model error.
    starts = pd.date_range(reference.index[0],
                           split - pd.Timedelta(minutes=15), freq="5min")
    detail = max(starts, key=lambda t: reference.loc[
        t:t + pd.Timedelta(minutes=15)].max() - reference.loc[
            t:t + pd.Timedelta(minutes=15)].min())
    axes[1].scatter(watch.index, predictions.instant, marker="+", s=45,
                    color="#823b9d", label="Same-time MyWhoosh records", zorder=3)
    axes[1].set_xlim(detail, detail + pd.Timedelta(minutes=15))
    axes[1].set_title(
        "15-minute training detail — chosen by reference HR range")
    axes[1].set_xlabel("Local time — Europe/Brussels (UTC+02:00)")
    axes[1].legend(loc="lower right", fontsize=8, ncol=2)
    names = list(scores.candidate.drop_duplicates())
    x = np.arange(len(names))
    for offset, part, color in ((-.2, "exploratory", "#90b9c5"),
                                (.2, "held_out", "#e6a15b")):
        rows = scores[scores.part == part].set_index("candidate").loc[names]
        axes[2].bar(x + offset, rows.mae, width=.4, color=color,
                    label=f"{part.replace('_', ' ')} (n={rows.n.iloc[0]})")
    labels = [scores[scores.candidate == n].label.iloc[0].replace(" ", "\n", 1)
              + ("\nTRAIN choice" if n == selected else "") for n in names]
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    axes[2].set_ylabel("Anchor MAE (BPM)")
    axes[2].set_title(
        "Identical covered anchors within each split; no held-out tuning")
    axes[2].legend(fontsize=9)
    axes[2].grid(axis="y", alpha=.2)
    fig.suptitle("FR265 wellness anchor diagnostic — 2026-10-12\n"
                 "One optical sensor, two recording paths; no reconstruction",
                 fontsize=16)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return detail


def write_report(summary, watch, ride, reference, raw, activity, inventory,
                 events, predictions, names, selected, specifications,
                 scores, offset_scores, regimes, blocks, train, held, split):
    """Document findings, common-support exclusions, and causal uncertainty."""
    auto = [e for e in events if e.get("event") == "auto_activity_detect"]
    counts = Counter(e.get("activity_type") for e in auto)
    inv = pd.DataFrame(inventory)
    watch_inv = inv[inv["product"] == "fr265_large"]
    types = watch_inv.groupby("type")["count"].sum().to_dict()
    app = bytes(activity["developer_data_id_mesgs"][0]["application_id"])
    selected_score = scores[(scores.candidate == selected)
                            & (scores.part == "held_out")].iloc[0]
    common = train | held
    training_start, training_end = ride[train].index[[0, -1]]
    held_start, held_end = ride[held].index[[0, -1]]
    all_instant = metrics(ride, predictions.instant)
    selected_blocks = blocks[(blocks.candidate == selected)
                             & (blocks.part == "held_out")]
    lines = [f"# FR265 wellness anchor diagnostic — {DATE}", "",
             "**No unique wellness averaging/timestamp mechanism is established.** "
             "A short centered mean improves the training fit, but the ranking "
             "changes later. All results are exploratory evidence from one ride.",
             "", "## Recording paths and input inspection", "",
             "The user confirms one FR265 optical sensor: (A) normal watch "
             "wellness storage and (B) Bluetooth to a Windows PC approximately "
             "50 cm away, recorded by MyWhoosh. No Edge was involved in this "
             "ride. MyWhoosh is a same-sensor reference path, not raw optical "
             "data or independent physiological ground truth. Proximity does "
             "not exclude latency, smoothing, buffering or retransmission loss.",
             "",
             f"Inspected {inv.file.nunique()} files in `data/2026-10-12/`: "
             f"{watch_inv.file.nunique()} identify `fr265_large`; one is Edge "
             "wellness with no valid HR and is excluded. All FIT integrity and "
             "decode checks passed. `monitoring` (55), field `heart_rate` (27), "
             "provides the watch observations. The existing audit's "
             "`monitoring_times` function resolves `timestamp_16` (26) against "
             "full monitoring timestamps, with rollover. No new FIT parser was "
             "implemented and no source timestamps were adjusted.", "",
             f"Recovered **{len(watch)}** valid watch observations across "
             f"{local(watch.index[0])}–{local(watch.index[-1])}, including the "
             f"next midnight endpoint. **{len(ride)}** occur during the virtual "
             f"ride: {local(ride.index[0])}–{local(ride.index[-1])}, all in "
             "`492952012911_WELLNESS.fit`. No invalid watch rows, duplicate "
             "anchors or conflicting timestamps were found. Exact UTC times, "
             "BPM and source filenames are in the observations CSV.", "",
             f"Ride spacing (seconds: count): **{summary['ride_spacing']}**. "
             f"Median {np.median(np.diff(ride.index.asi8) / 1e9):g}s; mean "
             f"{np.mean(np.diff(ride.index.asi8) / 1e9):.2f}s. "
             f"Whole-export spacing: {summary['all_watch_spacing']}.", "",
             f"Other watch HR-related records: `monitoring_hr_data` "
             f"({types.get('monitoring_hr_data_mesgs', 0)} resting-HR summaries), "
             f"`hrv_value` ({types.get('hrv_value_mesgs', 0)}), "
             f"`hrv_status_summary` ({types.get('hrv_status_summary_mesgs', 0)}). "
             "OHR settings, stress, respiration, SpO2 and monitoring fields "
             "are inventoried, not counted as additional BPM observations. "
             "No standard array `hr` messages occur. Unknown numeric message "
             "types are inventoried; their proprietary meanings remain unknown.",
             "",
             f"Automatic activity detections: {dict(counts)}; **no cycling "
             "classification** is present. Generic/sedentary detections during "
             "a stationary ride do not establish a cycling Move IQ event. "
             "Event reporting times are distinct from stored start/duration.",
             "", "## MyWhoosh activity and clock checks", "",
             "`data/virtual_road_ride_1.fit` contains "
             f"**{len(reference)} valid HR records**, from "
             f"{local(reference.index[0])} to {local(reference.index[-1])}. "
             "Session start is 16:27:52 and lap/session stop is 17:53:07 local: "
             "85m15s elapsed and timer time. The summary timestamp repeats the "
             "start and is not used as an end. All 5,114 inter-record intervals "
             "are 1s; no missing/zero/invalid HR, duplicate records, recording "
             "gaps or explicit pause events were found. This cannot rule out "
             "held values masking absent Bluetooth updates.", "",
             f"**{100 * summary['reference_unchanged_fraction']:.2f}%** of "
             "consecutive record pairs have identical BPM. The longest "
             f"constant run is {summary['reference_longest_equal_run']} records "
             "(48 seconds between endpoints). Thus 1 Hz storage does not prove "
             "1 Hz independent optical observations. Largest one-second "
             f"change: {reference.diff().abs().max():g} BPM at "
             f"{local(reference.diff().abs().idxmax())}; no >=10 BPM jumps. "
             "Changes are retained, not removed as outliers.", "",
             "The FIT header says Garmin/product 4062 (`edge_840`), which "
             "conflicts with the confirmed setup. Conversely, the developer "
             f"application ID decodes to **`{app.decode()}`**, and the session "
             "sub-sport is `virtual_activity`, corroborating MyWhoosh. No "
             "`device_info` sensor metadata establishes the Bluetooth device. "
             "The header is therefore not evidence of an actual Edge in the "
             "recording chain. User confirmation is retained.", "",
             "UTC record times align with the wellness date and are converted "
             "using Europe/Brussels (UTC+02:00), not naive local timestamps or "
             "the export's `local_timestamp`. Both have integer-second times; "
             "wellness anchors fall on minute boundaries. There is no "
             "independent clock synchronization log to attribute any offset.",
             "", "## Experiment and evaluation protocol", "",
             "Thirteen predefined zero-offset candidates: same-time recorded "
             "HR plus 15/30/60/120-second trailing, centered and leading means. "
             "For integer-second anchor t and width W: trailing uses "
             "[t-W+1, t]; leading [t+1, t+W]; centered "
             "[t-floor(W/2), t+ceil(W/2)-1]. Endpoints are inclusive, giving "
             "exactly W observations. Even centered windows have a -0.5s "
             "discrete midpoint; odd ones are exactly centered. Every second "
             "must contain an actual valid record (100% coverage); no "
             "interpolation, carry-forward or partial-window means.", "",
             "Select the best unshifted averaging shape by TRAIN MAE. Then "
             "search offsets -60…+60s in 5s steps for that shape and for "
             "instantaneous records only (61 unique candidates total, "
             "including zero offsets). Positive offset reads later MyWhoosh "
             "data; negative reads earlier. The final candidate minimizes "
             "TRAIN MAE across the fixed candidates and these two selected "
             "offset variants. No held-out score sets any parameter. No "
             "exponential smoothing search was added: this small dataset "
             "already cannot resolve competing window/offset explanations.",
             "",
             f"Chronological split at **{local(split)}**, 60% of the recorded "
             "time span. All candidates use the same anchor sets. A fully "
             "observed ±180s envelope covers every candidate support; it must "
             "also lie wholly inside its split. This intentionally purges "
             "boundary anchors so no reference window crosses training into "
             "evaluation. No anchor is excluded based on error or HR value.",
             "",
             f"Common TRAIN set: **{int(train.sum())}**, "
             f"{local(training_start)}–{local(training_end)}. Common held-out "
             f"set: **{int(held.sum())}**, {local(held_start)}–{local(held_end)}. "
             f"Excluded by common support/split guard: **{int((~common).sum())}** "
             "of 74 anchors. The original overlap has 74 same-time matches "
             f"(MAE {all_instant['mae']:.3f}, RMSE {all_instant['rmse']:.3f}); "
             "that larger set is not used to rank candidates.", "",
             "## Candidate results", "",
             "Errors below are **wellness minus reference representation**, "
             "in BPM. Correlations are descriptive. Displaying held-out scores "
             "for predefined alternatives does not authorize selecting a new "
             "winner from that portion.", "",
             "| Candidate | TRAIN MAE | Held-out MAE | RMSE | Bias | r | Max absolute error |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        rows = scores[scores.candidate == name].set_index("part")
        a, b = rows.loc["exploratory"], rows.loc["held_out"]
        title = label(specifications[name])
        if name == selected:
            title = f"**{title} — TRAIN choice**"
        lines.append(f"| {title} | {a.mae:.3f} | {b.mae:.3f} | "
                     f"{b.rmse:.3f} | {b.bias:.3f} | {b.correlation:.3f} | "
                     f"{b.max_abs_error:.3f} |")
    lines += ["",
              "The TRAIN choice is centered 60s at -5s: it averages reference "
              "records from **t-35 through t+24**. Its held-out MAE/RMSE is "
              f"**{selected_score.mae:.3f}/{selected_score.rmse:.3f} BPM**, "
              "versus 2.591/4.528 for same-time readings. The zero-offset "
              "centered 60s mean gives 2.519/3.773. Displacement alone selects "
              "-25s on TRAIN, but this differs from the averaged variant's -5s.",
              "",
              "Trailing 120s happens to score best on the held-out portion "
              "(0.871 BPM MAE), despite much worse TRAIN MAE (3.341). It is "
              "**not promoted as the selected mechanism**. This rank reversal "
              "and the changing HR regime argue against claiming a single "
              "fixed rule was identified. A shifted centered window can also "
              "be described as another alignment of the same samples: window "
              "placement and recording delay are intrinsically confounded.",
              "", "## HR regimes, discrepancies and temporal dependence", "",
              "Regimes use original reference HR at t+30 minus t-30: increasing "
              ">5 BPM, decreasing <-5 BPM, stable otherwise. High HR is "
              f">={summary['high_threshold']:g} BPM, the TRAIN same-time 75th "
              "percentile, fixed before evaluation; it overlaps other regimes.",
              "",
              "| Held-out regime | n | Same-time MAE | Selected MAE | Selected RMSE | Selected bias |",
              "|---|---:|---:|---:|---:|---:|"]
    for regime in ("increasing", "decreasing", "stable", "high_hr"):
        rows = regimes[(regimes.part == "held_out")
                       & (regimes.regime == regime)]
        a = rows[rows.candidate == "instant"].iloc[0]
        b = rows[rows.candidate == selected].iloc[0]
        lines.append(f"| {regime} | {b.n} | {a.mae:.3f} | {b.mae:.3f} "
                     f"| {b.rmse:.3f} | {b.bias:.3f} |")
    lines += ["", "Only one held-out increasing and one decreasing anchor "
              "are available; neither subgroup supports reliable generalization. "
              "Full regime metrics and correlations are saved in the regimes CSV.",
              "",
              f"Selected-candidate held-out MAE varies from "
              f"{selected_blocks.mae.min():.3f} to {selected_blocks.mae.max():.3f} "
              f"BPM across {len(selected_blocks)} contiguous five-minute blocks "
              "(1–5 anchors per block). Neighboring values, averaging windows "
              "and errors are temporally correlated. We do not treat 22 anchors "
              "as independent replicates, report IID confidence intervals, or "
              "claim statistical significance for the modest improvement.",
              "", "Largest selected-candidate held-out discrepancies:", "",
              "| Local timestamp | Watch | Reference representation | Signed error |",
              "|---|---:|---:|---:|"]
    errors = (ride[held] - predictions.loc[held, selected])
    for t in errors.abs().nlargest(3).index:
        lines.append(f"| {local(t)} | {ride[t]:g} | "
                     f"{predictions.loc[t, selected]:.3f} | {errors[t]:+.3f} |")
    lines += ["", "## Interpretation and October 16 comparison", "",
              "The results are compatible with averaging or filtering of a "
              "common optical signal, but they do not establish that wellness "
              "values are instantaneous, exactly 60-second means, or use a "
              "particular causal window. Modest TRAIN-selected displacement "
              "improves some agreement; its small benefit and dependence on "
              "window choice do not prove a clock error or a Bluetooth delay. "
              "Garmin storage conventions, optical filtering, Bluetooth "
              "updates and MyWhoosh buffering/recording remain confounded. "
              "The two paths originate from the same sensor.", "",
              "For October 16, the selected window SHAPE (centered 60s) was "
              "tested with **zero offset** against original Edge HR. The "
              "MyWhoosh-fitted -5s delay was not transferred. On the identical "
              f"{summary['mtb_same_time']['n']} fully supported MTB anchor "
              "timestamps, same-time MAE/RMSE is "
              f"{summary['mtb_same_time']['mae']:.3f}/"
              f"{summary['mtb_same_time']['rmse']:.3f} BPM; centered-mean "
              "MAE/RMSE is "
              f"{summary['mtb_zero_offset_shape']['mae']:.3f}/"
              f"{summary['mtb_zero_offset_shape']['rmse']:.3f} BPM. This small "
              "improvement does not explain most MTB disagreement.", "",
              "These are anchor-level comparisons, **not replacements** for "
              "the original linear baseline's 1,487 per-second pairs and "
              "12.78 BPM MAE. The two scores target different quantities. "
              "Smoothing and sparse storage can miss rapid changes, but the "
              "October 12 recording path is not a calibration of the Edge "
              "path. Keep original instantaneous-reference baseline metrics; "
              "add window-level diagnostics as sensitivity analyses rather "
              "than redefine success to lower errors.", "",
              "Next step: collect additional same-sensor paired recordings "
              "with repeated controlled rises, falls and recoveries, preserve "
              "clock/synchronization evidence, and test these fixed hypotheses "
              "on whole unseen rides. Include a separate watch-to-Edge session "
              "before transferring timing assumptions to MTB. No new "
              "reconstruction or machine-learning model is justified here.",
              "", "## Outputs and reproduction", "",
              f"- Figure: `docs/images/{STEM}.png` (3200×2600).",
              f"- Report: `docs/{STEM}.md`.",
              f"- Processed tables: `data/processed/{STEM}_*.csv`: all observed "
              "anchors with provenance, candidate values, full metrics, "
              "TRAIN-only offset search, regimes, five-minute blocks, file "
              "inventory and separate MTB check. Metadata and summary JSON "
              "files accompany them.", "",
              "```powershell",
              ".\\.venv\\Scripts\\python.exe scripts/diagnose_watch_anchors.py",
              "```", "",
              "Uses existing NumPy, pandas, Matplotlib and Garmin SDK; no new "
              "dependencies. UTC is retained in CSVs; plot labels use Brussels. "
              "Checks independently recompute every main candidate's windows "
              "from original rows, enforce full coverage and split separation, "
              "and verify exact same-time matches. Source FITs and existing "
              "October 16 artifacts are hash-checked unchanged. No commits, "
              "uploads or modifications to original activities were performed.", ""]
    (ROOT / "docs" /
     f"{STEM}.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run the diagnostic while preserving source FITs and October 16 results."""
    protected = [p for p in ROOT.rglob("*") if p.is_file()
                 and ".venv" not in p.parts and ".git" not in p.parts
                 and (p.suffix.lower() == ".fit" or "2026-10-16" in p.name)]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    watch, raw, inventory, metadata, events, cleaning = load_watch(
        ROOT / "data/2026-10-12")
    reference, records, activity, ref_cleaning = load_activity(
        ROOT / "data/virtual_road_ride_1.fit")
    ride = watch[(watch.index >= reference.index[0])
                 & (watch.index <= reference.index[-1])]
    split = reference.index[0] + \
        (reference.index[-1] - reference.index[0]) * .6
    (predictions, specifications, names, selected, average, train, held,
     scores, offset_scores) = select_candidates(ride, reference, split)
    regimes, blocks, high_threshold = regimes_and_blocks(
        ride, reference, predictions, ["instant", average, selected], train, held)
    processed = ROOT / "data/processed"
    raw.to_csv(processed / f"{STEM}_observations.csv", index=False)
    comparison = predictions.copy()
    comparison.insert(0, "watch_hr", ride)
    comparison["part"] = np.where(train, "exploratory",
                                  np.where(held, "held_out", "excluded_support"))
    comparison["selected_error_watch_minus_reference"] = ride - \
        predictions[selected]
    comparison.index.name = "timestamp_utc"
    comparison.to_csv(processed / f"{STEM}_comparisons.csv")
    scores.to_csv(processed / f"{STEM}_metrics.csv", index=False)
    offset_scores.to_csv(processed / f"{STEM}_offset_search.csv", index=False)
    regimes.to_csv(processed / f"{STEM}_regimes.csv", index=False)
    blocks.to_csv(processed / f"{STEM}_blocks.csv", index=False)
    pd.DataFrame(inventory).to_csv(
        processed / f"{STEM}_inventory.csv", index=False)
    details = {"wellness_metadata": metadata, "wellness_events": events,
               "activity_metadata": {k: v for k, v in activity.items()
                                     if k != "record_mesgs"}}
    (processed / f"{STEM}_metadata.json").write_text(
        json.dumps(details, default=str, indent=2), encoding="utf-8")
    # Transfer the selected SHAPE with zero displacement to the distinct Edge
    # recording path; do not transfer a MyWhoosh-fitted delay or change baseline.
    mtb = pd.read_csv(processed / "watch_hr_ride_2026-10-16.csv")
    mtb_watch, _, _ = clean_observations(mtb)
    edge, _, _, _ = load_activity(ROOT / "data/mtb_edge.fit")
    shape = (*specifications[selected][:2], 0)
    mtb_values = pd.DataFrame({
        "watch": mtb_watch,
        "same_time": reference_values(edge, mtb_watch.index, ("instant", 1, 0)),
        "zero_offset_shape": reference_values(edge, mtb_watch.index, shape),
    }).dropna()
    mtb_values.to_csv(
        processed / f"{STEM}_mtb_check.csv", index_label="timestamp_utc")
    detail = make_figure(ride, reference, predictions, specifications[selected],
                         selected, scores, split, train,
                         ROOT / "docs/images" / f"{STEM}.png")
    summary = {
        "all_watch_n": len(watch), "ride_watch_n": len(ride),
        "ride_spacing": distribution(ride), "all_watch_spacing": distribution(watch),
        "reference_n": len(reference), "reference_spacing": distribution(reference),
        "reference_unchanged_fraction": float(reference.diff().eq(0).iloc[1:].mean()),
        "reference_longest_equal_run": int(reference.groupby(
            reference.ne(reference.shift()).cumsum()).size().max()),
        "start": local(reference.index[0]), "end": local(reference.index[-1]),
        "split": local(split), "train_n": int(train.sum()), "held_n": int(held.sum()),
        "selected": selected, "selected_spec": specifications[selected],
        "best_unshifted_average": average, "cleaning": cleaning,
        "reference_cleaning": ref_cleaning, "high_threshold": high_threshold,
        "detail_start": local(detail),
        "mtb_same_time": metrics(mtb_values.watch, mtb_values.same_time),
        "mtb_zero_offset_shape": metrics(mtb_values.watch, mtb_values.zero_offset_shape),
    }
    (processed / f"{STEM}_summary.json").write_text(
        json.dumps(summary, default=str, indent=2), encoding="utf-8")
    write_report(summary, watch, ride, reference, raw, activity, inventory,
                 events, predictions, names, selected, specifications,
                 scores, offset_scores, regimes, blocks, train, held, split)
    # Every selected window can be independently recomputed from original rows.
    for name in names:
        offsets = window_offsets(*specifications[name])
        for t in ride.index[train | held]:
            support = t + pd.to_timedelta(offsets, unit="s")
            values = reference.reindex(support)
            assert values.notna().all()
            assert np.isclose(predictions.loc[t, name], values.mean())
            assert (support.max() < split if train.loc[t]
                    else support.min() >= split)
    assert np.array_equal(predictions.instant.to_numpy(),
                          reference.reindex(ride.index).to_numpy())
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h
               for p, h in hashes.items())
    outputs = list(processed.glob(f"{STEM}_*")) + [
        ROOT / "docs" / f"{STEM}.md",
        ROOT / "docs/images" / f"{STEM}.png",
    ]
    assert all(p.stat().st_size > 0 for p in outputs)
    print(json.dumps(summary, indent=2, default=str))
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
