"""Run one final trial-10 fit with soft centered wellness-average observations.

Run in deep: python scripts/reconstruct_hr_soft_average.py
--reuse-trained regenerates outputs from this experiment only, without fitting.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

import search_hr_tcn_128 as search
import evaluate_hr_tcn_local as local
from garmin_fit_sdk import Decoder, Encoder, Profile, Stream

previous, base, window, tf = local.previous, local.base, local.window, local.tf
ROOT, ZONE = local.ROOT, local.ZONE
RIDES = tuple("road_ride_" + str(i)
              for i in range(1, 6)) + ("mtb_ride_1", "mtb_ride_2")
CONFIG = dict(trial_id="trial_10", dilations=[1, 2, 4, 8], channels=16,
              learning_rate=.0003, batch_size=64, receptive_field=61, parameters=6753)
WEIGHTS = dict(fidelity=1., local_mean=60., adjustment_difference=25.)
ASSUMPTION = "Centered 60-s arithmetic mean, [t-30,t+29]; unverified firmware assumption"
CSV = ROOT / "data/processed/hr_tcn_soft_average_2026-10-16.csv"
REPORT = ROOT / "docs/hr_final_soft_average_reconstruction.md"
FIGURE = ROOT / "docs/images/hr_tcn_soft_average_2026-10-16.png"
FINAL = ROOT / "results/final"
MODEL = FINAL / "tcn_trial10_soft_average_model.h5"
STATE = FINAL / "tcn_trial10_soft_average_normalization.json"
FIT_PATH = FINAL / "hr_reconstructed_2026-10-16.fit"
AUDIT = FINAL / "fit_export_audit.json"
SOURCE = ROOT / "data/mtb_edge.fit"


def mean_observations(reference, times, frame):
    """Reuse complete centered 60-s means; require the same recorded block."""
    observations = base.make_anchors(reference, times, "averaged")
    for timestamp, value in observations.items():
        interval = pd.date_range(
            timestamp - pd.Timedelta(seconds=30), periods=60, freq="s")
        assert interval.isin(frame.index).all()
        actual = reference.reindex(interval)
        assert actual.notna().all() and np.isclose(actual.mean(), value)
    return observations


def training_arrays(rides, packages, norm):
    """Train on true HR residuals, including average-observation center seconds."""
    inputs, labels, counts = [], [], {}
    for name in RIDES:
        ride, packs = rides[name], packages[name]
        frame, reference = ride["frame"], ride["reference"]
        normalized = {phase: previous.normalized(packs[phase]["features"], norm)
                      for phase in base.PHASES}
        used, centers, n, pairs = [], 0, 0, 0
        for block in previous.block_indices(frame):
            for start in range(0, len(block) - 128 + 1, 16):
                phase = base.PHASES[(start // 16) % 3]
                package = packs[phase]
                positions = block[start:start + 128]
                assert (
                    np.diff(frame.index[positions].asi8) == 1_000_000_000).all()
                times = frame.index[block[start + 56:start + 72]]
                baseline = package["baseline"].linear.reindex(times)
                residual = reference.reindex(times) - baseline
                # An averaged observation does not reveal instantaneous HR.
                valid = residual.notna().to_numpy()
                if not valid.any():
                    continue
                inputs.append(normalized[phase][positions])
                labels.append(np.column_stack([residual.fillna(0), valid,
                                               baseline.fillna(0)]).astype(np.float32))
                used.extend(times[valid])
                centers += int((valid &
                               times.isin(package["anchors"].index)).sum())
                pairs += int((valid[:-10] & valid[10:]).sum())
                n += 1
        assert n > 0 and pd.DatetimeIndex(used).is_unique
        counts[name] = dict(windows=n, loss_seconds=len(used), temporal_pairs=pairs,
                            supervised_average_centers=centers)
    x, y = np.stack(inputs), np.stack(labels)
    assert x.shape[1:] == (128, 28) and y.shape[1:] == (16, 3)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return x, y, counts


def unanchored_series(baseline, residual):
    """Smooth residuals inside supported blocks, without a taper or dot overwrite."""
    result = pd.Series(np.nan, index=baseline.index, dtype=float)
    for block in base.continuous_blocks(baseline.dropna()):
        values = residual.reindex(block.index)
        assert values.notna().all()
        result.loc[block.index] = block + \
            values.rolling(5, center=True, min_periods=1).mean()
    assert result.notna().equals(baseline.notna())
    return result


def predict(frame, package, fitted):
    """Reuse padded window extraction/overlap averaging and omit hard anchoring."""
    values = previous.normalized(package["features"], fitted["normalization"])
    residual = pd.Series(np.nan, index=frame.index, dtype=float)
    for block in previous.block_indices(frame):
        centers = np.arange(0, len(block), 8)
        batches = []
        for offset in range(0, len(centers), 64):
            batch = np.stack([previous.pad_window(values[block], int(center))
                              for center in centers[offset:offset + 64]])
            batches.append(fitted["network"](
                batch, training=False).numpy()[..., 0])
        prediction = np.concatenate(batches)
        assert np.isfinite(prediction).all()
        residual.iloc[block] = previous.overlap(
            len(block), centers, prediction)
    return unanchored_series(package["baseline"].linear, residual)


def window_means(series, watch):
    """Calculate only fully supported, exact centered 60-s means and reasons."""
    rows = []
    for timestamp, value in watch.items():
        interval = pd.date_range(
            timestamp - pd.Timedelta(seconds=30), periods=60, freq="s")
        positions = series.index.get_indexer(interval)
        if (positions < 0).any():
            reason, mean = "window_outside_recorded_timeline_or_recording_gap", np.nan
        elif series.iloc[positions].isna().any():
            reason, mean = "unsupported_HR_inside_60s_window", np.nan
        else:
            reason, mean = "complete", float(series.iloc[positions].mean())
        rows.append(dict(timestamp_utc=timestamp, watch_hr=value, mean=mean,
                         residual=mean - value, reason=reason))
    return pd.DataFrame(rows).set_index("timestamp_utc")


def soft_adjust(candidate, watch, fixed):
    """Solve one fixed sparse quadratic problem; never fill gaps or alter Edge HR.

    delta is zero at fixed samples. A averages complete 60s windows. D takes
    differences between adjacent variable samples; at a measured/estimated seam
    it penalizes the final HR jump instead. All differences are on the 1Hz grid.
    """
    fixed = fixed.reindex(candidate.index, fill_value=False)
    support = candidate.notna()
    variable = support & ~fixed
    locations = np.flatnonzero(variable)
    mapping = np.full(len(candidate), -1, dtype=int)
    mapping[locations] = np.arange(len(locations))
    diagnostics = window_means(candidate, watch)
    rows, cols, data, targets = [], [], [], []
    for timestamp, row in diagnostics.iterrows():
        if row.reason != "complete":
            continue
        interval = pd.date_range(
            timestamp - pd.Timedelta(seconds=30), periods=60, freq="s")
        positions = candidate.index.get_indexer(interval)
        columns = mapping[positions]
        columns = columns[columns >= 0]
        diagnostics.loc[timestamp, "adjustable_seconds"] = len(columns)
        if not len(columns):
            continue  # Frozen-only windows add a constant objective term.
        equation = len(targets)
        rows.extend([equation] * len(columns))
        cols.extend(columns.tolist())
        data.extend([1. / 60] * len(columns))
        targets.append(-row.residual)
    a = sparse.csr_matrix((data, (rows, cols)),
                          shape=(len(targets), len(locations)))
    rows, cols, data, jumps = [], [], [], []
    for right in range(1, len(candidate)):
        left = right - 1
        if (candidate.index[right] - candidate.index[left]).total_seconds() != 1:
            continue
        if not (support.iloc[left] and support.iloc[right]):
            continue
        if mapping[left] < 0 and mapping[right] < 0:
            continue
        row = len(jumps)
        for position, sign in ((left, -1.), (right, 1.)):
            if mapping[position] >= 0:
                rows.append(row)
                cols.append(mapping[position])
                data.append(sign)
        seam = mapping[left] < 0 or mapping[right] < 0
        jumps.append(-(candidate.iloc[right] -
                     candidate.iloc[left]) if seam else 0.)
    d = sparse.csr_matrix((data, (rows, cols)),
                          shape=(len(jumps), len(locations)))
    target, boundary = np.asarray(targets), np.asarray(jumps)
    w = WEIGHTS
    hessian = (w["fidelity"] * sparse.eye(len(locations), format="csr")
               + w["local_mean"] * (a.T @ a)
               + w["adjustment_difference"] * (d.T @ d))
    rhs = w["local_mean"] * (a.T @ target) + \
        w["adjustment_difference"] * (d.T @ boundary)
    solution = spsolve(hessian.tocsc(), rhs) if len(
        locations) else np.array([])
    assert np.isfinite(solution).all()
    assert np.linalg.norm(hessian @ solution - rhs) <= 1e-7 * \
        max(1., np.linalg.norm(rhs))
    adjustment = pd.Series(0., index=candidate.index).where(support)
    adjustment.iloc[locations] = solution
    final = candidate + adjustment
    assert final.notna().equals(support)
    assert np.array_equal(final.loc[fixed], candidate.loc[fixed])
    objective_before = w["local_mean"] * \
        np.sum(target**2) + w["adjustment_difference"] * np.sum(boundary**2)
    objective_after = (w["fidelity"] * np.sum(solution**2)
                       + w["local_mean"] * np.sum((a @ solution - target)**2)
                       + w["adjustment_difference"] * np.sum((d @ solution - boundary)**2))
    assert objective_after <= objective_before + 1e-6
    stats = dict(adjustable_seconds=len(locations), adjustable_windows=len(targets),
                 objective_before=float(objective_before), objective_after=float(objective_after),
                 normal_equation_relative_error=float(np.linalg.norm(hessian @ solution - rhs) / max(1., np.linalg.norm(rhs))))
    return final, adjustment, diagnostics, stats


def deterministic_checks():
    """Check mean alignment, no anchoring, gaps and one soft solve before fitting."""
    times = pd.date_range("2026-01-01", periods=181, freq="s", tz="UTC")
    reference = pd.Series(100 + np.arange(181), index=times, dtype=float)
    center = times[[90]]
    averaged = mean_observations(reference, center, pd.DataFrame(index=times))
    assert averaged.iloc[0] == 189.5 and reference.loc[center[0]] == 190
    gapped = reference.drop(times[89])
    assert mean_observations(
        gapped, center, pd.DataFrame(index=gapped.index)).empty
    baseline = pd.Series(100., index=times)
    raw = unanchored_series(baseline, baseline * 0 + 3)
    assert raw.loc[center[0]] == 103  # No 4f(1-f) taper or overwrite to 100.
    summary = pd.Series(120., index=center)
    final, delta, _, stats = soft_adjust(baseline, summary, baseline < 0)
    assert 100 < final.loc[center[0]] < 120
    assert abs(window_means(final, summary).residual.iloc[0]) < 20
    assert not np.isclose(final.loc[center[0]], 120) and delta.max() < 20
    fixed = baseline.index <= times[59]
    candidate = baseline.copy()
    candidate.iloc[100:110] = np.nan
    gap_observations = pd.Series([120., 120.], index=times[[90, 150]])
    immutable, change, reasons, _ = soft_adjust(
        candidate, gap_observations, pd.Series(fixed, index=times))
    assert np.array_equal(immutable.loc[fixed], candidate.loc[fixed])
    assert immutable.iloc[100:110].isna().all(
    ) and change.iloc[100:110].isna().all()
    assert np.allclose(change.iloc[:100], 0) and change.iloc[110:].max() > 0
    assert reasons.loc[times[90],
                       "reason"] == "unsupported_HR_inside_60s_window"
    seam = baseline.copy()
    seam.iloc[:60] = 95.
    joined, _, _, _ = soft_adjust(seam, summary, pd.Series(fixed, index=times))
    assert np.array_equal(joined.iloc[:60], seam.iloc[:60])
    assert abs(joined.iloc[60] - joined.iloc[59]) < 5
    assert stats["normal_equation_relative_error"] < 1e-7
    print("Pre-fit deterministic average/soft-consistency checks passed", flush=True)


def fit_audit():
    """Probe the original with the installed SDK; refuse lossy unsupported export."""
    assert SOURCE.is_file() and not FIT_PATH.exists()
    integrity = Decoder(Stream.from_file(str(SOURCE))).check_integrity()
    assert integrity
    ordered = []
    messages, errors = Decoder(Stream.from_file(str(SOURCE))).read(
        merge_heart_rates=False, expand_sub_fields=False, expand_components=False,
        mesg_listener=lambda number, message: ordered.append((number, message.copy())))
    assert not errors, errors
    counts = Counter(number for number, _ in ordered)
    unknown = Counter()
    for number, message in ordered:
        known = {f["name"] for f in Profile["messages"].get(
            number, {}).get("fields", {}).values()}
        for field in message:
            if field not in known and field not in ("developer_fields", "mesg_num"):
                unknown[number, str(field)] += 1
    failure = None
    try:
        encoder = Encoder()
        for number, message in ordered:
            encoder.on_mesg(number, message)
        # In-memory compatibility probe only; never write stripped data.
        encoder.close()
    except ValueError as error:
        failure = str(error)
    # This source has unknown/proprietary content. No lossy export is permitted.
    assert failure or unknown, "Source changed: reassess safe encoding before exporting"
    records = messages["record_mesgs"]
    timestamps = pd.to_datetime([r["timestamp"] for r in records], utc=True)
    assert timestamps.is_unique and timestamps.is_monotonic_increasing
    required = ["file_id_mesgs", "activity_mesgs", "session_mesgs",
                "lap_mesgs", "record_mesgs", "event_mesgs"]
    assert all(messages.get(key) for key in required)
    import garmin_fit_sdk
    audit = dict(status="blocked_no_FIT_created", ready_for_manual_upload=False,
                 source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                 sdk_version=garmin_fit_sdk.__version__, original_integrity=integrity,
                 source_records=len(records), source_recorded_hr=sum("heart_rate" in r for r in records),
                 record_timestamp_sha256=hashlib.sha256(
                     timestamps.asi8.tobytes()).hexdigest(),
                 original_hr_sha256=hashlib.sha256(np.nan_to_num(np.asarray(
                     [r.get("heart_rate", np.nan) for r in records], dtype=np.float64), nan=-1.).tobytes()).hexdigest(),
                 original_required_messages={
                     key: len(messages[key]) for key in required},
                 message_counts={str(k): v for k, v in counts.items()},
                 unknown_message_counts={
                     str(k): v for k, v in counts.items() if k not in Profile["messages"]},
                 unknown_fields=[dict(message=k[0], field=k[1], count=v)
                                 for k, v in sorted(unknown.items())],
                 encoder_error=failure, exported_integrity="not_run_no_export",
                 export_roundtrip="not_run_no_export",
                 summary_updates="none: export blocked; original proprietary and HR summaries left untouched")
    search.write_json(AUDIT, audit)
    return audit


def summary_stats(values):
    """Describe finite signed differences in BPM without disguising missing data."""
    values = values.dropna()
    return dict(n=len(values), bias=float(values.mean()), mae=float(values.abs().mean()),
                rmse=float(np.sqrt((values**2).mean())), p95_abs=float(values.abs().quantile(.95)),
                max_abs=float(values.abs().max()))


def target_inputs(ride, watch):
    """Build the target's inputs from cycling records and watch means only."""
    frame = ride["frame"]
    package = base.examples(frame, watch, frame.index,
                            base.activity_features(frame))
    package["features"] = previous.raw_features(frame, package)
    return package


def make_table(target, watch, raw):
    """Combine measured HR after the withheld-overlap diagnostic, then solve once."""
    frame = target["frame"]
    edge = target["reference"].reindex(frame.index)
    before = raw.where(edge.isna(), edge)
    final, delta, eligible, adjustment_stats = soft_adjust(
        before, watch, edge.notna())
    initial_means, before_means, final_means = (
        window_means(v, watch) for v in (raw, before, final))
    result = pd.DataFrame(dict(edge_hr=edge, watch_wellness_hr=watch.reindex(frame.index),
                               initial_tcn_estimate=raw, soft_adjustment_bpm=delta,
                               final_soft_adjusted_estimate=raw + delta,
                               final_combined_hr=final))
    result["source"] = np.select([edge.notna(), final.notna()], [
                                 "recorded", "estimated"], default="missing")
    result["model_supported"] = raw.notna()
    result["terrain_supported"] = frame.climb30.notna()
    result["probable_stop"] = frame.probable_stop
    result["watch_window_complete"] = False
    result["watch_window_reason"] = "no_watch_observation"
    for name, values in (("initial_tcn", initial_means), ("before_soft", before_means), ("final", final_means)):
        result["watch_60s_mean_" + name] = values["mean"].reindex(frame.index)
        result["watch_60s_residual_" +
               name] = values.residual.reindex(frame.index)
    result.loc[watch.index, "watch_window_reason"] = final_means.reason
    result.loc[watch.index, "watch_window_complete"] = final_means.reason.eq(
        "complete")
    result["watch_window_adjustable_seconds"] = eligible.adjustable_seconds.reindex(
        frame.index)
    result.index.name = "timestamp_utc"
    assert result.index.equals(frame.index)
    assert np.array_equal(
        result.loc[edge.notna(), "final_combined_hr"], edge.dropna())
    assert np.array_equal(result.loc[watch.index, "watch_wellness_hr"], watch)
    assert final.loc[frame.index > watch.index[-1]].isna().all()
    finite = final.dropna()
    assert finite.gt(0).all() and finite.lt(255).all()
    assert not np.isclose(result.loc[result.source.eq("estimated") & result.watch_wellness_hr.notna(),
                                     "final_combined_hr"],
                          result.loc[result.source.eq("estimated") & result.watch_wellness_hr.notna(),
                                     "watch_wellness_hr"]).all()
    return result, adjustment_stats


def figures(table):
    """Display the complete estimate, fixed post-disconnection zoom and averages."""
    plt = window.plt
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), constrained_layout=True)
    edge = table.edge_hr.dropna()
    cutoff = edge.index[-1]
    estimated = table.source.eq("estimated")
    watch = table.watch_wellness_hr.dropna()
    axes[0].plot(table.index, table.final_combined_hr.where(
        estimated), color="#2679b5", lw=1.1, label="Estimated HR (soft averages)")
    axes[0].plot(table.index, table.edge_hr, color=".25",
                 lw=1.1, label="Original Edge HR")
    axes[0].scatter(watch.index, watch, color="#e58c28", s=15,
                    zorder=5, label="Watch summaries (assumed 60s means)")
    axes[0].axvline(cutoff, color="#bc3035", ls="--",
                    label="Last recorded Edge HR")
    for i, positions in enumerate(previous.runs(table.probable_stop, table.index)):
        times = table.index[positions]
        axes[0].axvspan(times[0], times[-1], color=".5", alpha=.09,
                        label="Probable sustained stop" if i == 0 else None)
    axes[0].set(title="A — Final experimental reconstruction; estimates are not recovered measurements",
                ylabel="HR (BPM)", xlim=(table.index[0], table.index[-1]))
    axes[0].legend(ncol=2, fontsize=8)
    # Fixed before looking at predictions: central 20min of post-disconnection support.
    supported = table.loc[estimated & table.final_combined_hr.notna()]
    center = supported.index[0] + \
        (supported.index[-1] - supported.index[0]) / 2
    start = (center - pd.Timedelta(minutes=10)).floor("s")
    end = start + pd.Timedelta(minutes=20)
    zoom = table.loc[start:end]
    axes[1].plot(zoom.index, zoom.initial_tcn_estimate, color=".6",
                 lw=1, ls="--", label="Initial TCN estimate")
    axes[1].plot(zoom.index, zoom.final_combined_hr,
                 color="#2679b5", lw=1.3, label="Soft-adjusted estimate")
    visible = watch.loc[start:end]
    axes[1].scatter(visible.index, visible, color="#e58c28", s=25,
                    zorder=5, label="Watch summary, not instantaneous HR")
    for timestamp, value in visible.items():
        axes[1].hlines(value, timestamp - pd.Timedelta(seconds=30), timestamp + pd.Timedelta(seconds=29),
                       color="#e58c28", alpha=.35, lw=2)
    axes[1].set(title="B — Fixed central 20min after disconnection; orange spans are assumed averaging windows",
                ylabel="HR (BPM)", xlim=(start, end))
    axes[1].legend(ncol=2, fontsize=8)
    observations = table.loc[table.watch_wellness_hr.notna()]
    used = observations.watch_window_complete
    variable = observations.watch_window_adjustable_seconds.gt(0)
    for mask, color, label in ((used & variable, "#2679b5", "Adjustable complete windows"),
                               (used & ~variable, ".35", "Original Edge-only windows (frozen)")):
        axes[2].scatter(observations.loc[mask, "watch_wellness_hr"], observations.loc[mask, "watch_60s_mean_final"],
                        s=22, alpha=.7, color=color, label=label)
    lo = min(observations.watch_wellness_hr.min(),
             observations.watch_60s_mean_final.min()) - 3
    hi = max(observations.watch_wellness_hr.max(),
             observations.watch_60s_mean_final.max()) + 3
    axes[2].plot([lo, hi], [lo, hi], color="#e58c28", ls=":",
                 label="Equal local means (reference line, not forced)")
    axes[2].set(title="C — Complete 60s windows; incomplete windows excluded", xlabel="Watch summary (BPM)",
                ylabel="Mean final HR over [t−30,t+29] (BPM)", xlim=(lo, hi), ylim=(lo, hi))
    axes[2].legend(fontsize=8)
    for ax in axes[:2]:
        ax.xaxis.set_major_formatter(
            window.mdates.DateFormatter("%H:%M", tz=ZONE))
        ax.set_xlabel("Local time — Europe/Brussels")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)


def report(table, payload, audit, overlap, adjustment_stats, comparison, preserved):
    """Report actual results and the explicitly blocked FIT acceptance checks."""
    def markdown(frame):
        return "\n".join(local.markdown_table(list(frame.columns), frame.itertuples(index=False, name=None)))
    observations = table.loc[table.watch_wellness_hr.notna()]
    complete = observations.watch_window_complete
    variable = observations.watch_window_adjustable_seconds.gt(0) & complete
    means = pd.DataFrame([
        dict(series="Model-only (Edge withheld)", **
             summary_stats(observations.watch_60s_residual_initial_tcn)),
        dict(series="Combined before soft adjustment", **
             summary_stats(observations.watch_60s_residual_before_soft)),
        dict(series="Final, all complete windows", **
             summary_stats(observations.watch_60s_residual_final)),
        dict(series="Final, adjustable windows", **
             summary_stats(observations.loc[variable, "watch_60s_residual_final"])),
        dict(series="Final, frozen Edge-only windows", **summary_stats(observations.loc[complete & ~variable, "watch_60s_residual_final"]))])
    estimated = table.source.eq("estimated")
    dot = observations.loc[observations.source.eq("estimated")]
    instantaneous = summary_stats(
        dot.final_combined_hr - dot.watch_wellness_hr)
    instantaneous["exact_dot_matches"] = int(np.isclose(
        dot.final_combined_hr, dot.watch_wellness_hr, atol=1e-9, rtol=0).sum())
    magnitude = summary_stats(table.loc[estimated, "soft_adjustment_bpm"])
    valid = table.final_combined_hr.dropna()
    counts = dict(recorded=int(table.source.eq("recorded").sum()), estimated=int(estimated.sum()),
                  missing=int(table.source.eq("missing").sum()), supported=len(valid),
                  coverage_percent=100 * len(valid) / len(table), mean=float(valid.mean()), maximum=float(valid.max()))
    step = table.final_combined_hr.diff().where(
        table.index.to_series().diff().dt.total_seconds().eq(1)).abs()
    estimated_step = step.where(estimated)
    lines = ["# Final experimental HR reconstruction: soft wellness averages", "",
             "**Reconstruction CSV, model and figure completed; FIT export blocked safely.** "
             "The installed official Encoder cannot preserve this original file's unsupported messages/fields. "
             "No reconstructed FIT or upload-ready claim is produced, and nothing was uploaded.", "",
             "## Model and observation assumption", "",
             "One fresh trial-10 TCN fit on five road rides plus MTB rides 1 and 2; no MyWhoosh or October 16 training. "
             "Input 128×28; central positions 56–71 predict 16 residuals; 16 channels, kernel 3, "
             "two same-padded noncausal Conv1D layers per residual block, dilations 1/2/4/8, "
             "ReLU/skip connections, linear projection, 6,753 parameters, RF 61s. "
             "Adam 0.0003, batch 64, 30 epochs, seed 265, float32; training/inference strides 16/8. "
             "GPU matmul and all-output RF gradients checked on the RTX 3080 Laptop GPU in unchanged Conda deep.", "",
             "**Unverified observation assumption:** a watch value at t approximates mean HR over "
             "[t−30,t+29], not instantaneous HR at t. Firmware averaging/filtering remains unknown. "
             "Synthetic means require 60 valid reference seconds in one recorded block, using the exact "
             "142×60s/15×120s interval sequence and phases 0/20/40. Only averaged examples are trained. "
             "The 28 existing channels retain their meanings, with sparse HR and baseline now representing summaries. "
             "Normalization uses these seven reference rides only. True instantaneous residual labels include "
             "observation-center seconds, because a mean does not disclose their instantaneous HR. "
             "Masks exclude missing reference HR/unsupported baseline; temporal pairs stay in one 16s output/block.", "",
             "Target context uses the original Edge speed, movement and UTC timeline, plus the existing "
             "quality-masked Strava climbing features from `data/VTT_Apero.gpx`. These are predictor "
             "features only; no original route, altitude, time, distance or other sensor value is rewritten.", "",
             "Loss remains masked Huber(delta 5) + 0.5×10s absolute-HR-change MAE. "
             "Inference averages overlapping residual predictions and retains the existing centered five-sample "
             "residual mean inside each supported block. Both the 4f(1−f) taper and observation-time overwrite "
             "are absent. The interpolated average baseline is context, never proof of HR at a dot. "
             "No support is extended beyond the observations, across a missing record, or across >180s gaps.", "",
             "## Training", "",
             "Python %s, TensorFlow %s, Keras %s; actual fit %.2fs. "
             "Final total/Huber/temporal losses: %.5f / %.5f / %.5f."
             % (payload["environment"]["python"].split()[0], payload["environment"]["tensorflow"],
                payload["environment"]["keras"], payload["training_seconds"], payload["history"]["loss"][-1],
                payload["history"]["huber_component"][-1], payload["history"]["temporal_component"][-1]), "",
             "Keras epoch total loss and component metrics aggregate the final partial batch differently; "
             "the displayed epoch components need not add exactly to the displayed total. "
             "Each batch uses the exact fixed combined loss above.", "",
             markdown(pd.DataFrame(payload["counts"]).T.reset_index().rename(
                 columns={"index": "ride"})), "",
             "## Soft consistency, fixed before outcome inspection", "",
             "Let h0 be the unadjusted TCN with actual Edge samples substituted only after the withheld "
             "overlap diagnostic. Solve for adjustment δ at supported estimated seconds; δ=0 at actual Edge "
             "samples. With A the complete-window 60-s mean and w the watch summaries:", "",
             "`J = sum(δ²) + 60 sum((A(h0+δ)−w)²) + 25 sum((Dδ−b)²)`", "",
             "All residuals/differences are BPM on the one-second grid; the objective is BPM² with fixed "
             "dimensionless weights 1/60/25. The factor 60 gives a minute-level mean penalty comparable "
             "to 60 per-second fidelity terms; these are conservative engineering weights, not calibrated "
             "measurement precisions. D connects adjacent supported estimated samples only, with b=0, "
             "discouraging rough adjustments. At a measured/estimated seam it instead includes the available "
             "one-step boundary with b=−(h0_right−h0_left), encouraging continuity with the fixed real HR. "
             "No operator crosses unsupported time. The sparse SPD normal equations are solved once using "
             "SciPy. Finite weights do not impose exact dot or window equality. "
             "Frozen-only windows contribute constant disagreement and cannot change real Edge HR.", "",
             "Solver/objective audit: `%s`." % adjustment_stats, "",
             "## Original Edge overlap (model only, before substitution or consistency)", "",
             markdown(pd.DataFrame([overlap]).round(5)), "",
             "Actual Edge HR was absent from prediction features, baseline, normalization and training. "
             "Future non-HR measurements and actual watch summaries are permitted offline. "
             "Watch-center timestamps remain scored since averages are not instantaneous HR observations. "
             "These masks/full-ride observation support differ from the earlier 1,420-second hard-anchor "
             "diagnostic, so their errors are not a controlled head-to-head improvement claim. "
             "This previously examined initial overlap is not independent validation; the new average-trained "
             "model has no new held-out ride evaluation or tuned consistency weights.", "",
             "## Final reconstruction and agreement", "",
             markdown(pd.DataFrame([counts]).round(5)), "",
             "Mean/max describe estimates on supported seconds, not recovered physiological measurements. "
             "Every original timestamp, all original Edge values and all 158 separate watch summaries remain. "
             "Unsupported final time stays missing, not zero. Maximum/P99 adjacent supported final change: "
             "%.4f / %.4f BPM." % (step.max(), step.quantile(.99)), "",
             "For estimated seconds, including the measured/estimated handoff, maximum/P99 change is "
             "%.4f / %.4f BPM. The largest full-ride jump occurs in the `%s` portion and is preserved "
             "because original HR must remain exact."
             % (estimated_step.max(), estimated_step.quantile(.99), table.loc[step.idxmax(), "source"]), "",
             "Watch-window residual = reconstructed mean minus watch summary (BPM):", "",
             markdown(means.round(5)), "",
             "Incomplete reasons: `%s`. Frozen-only complete windows: %d. "
             "A summary cannot override conflicting measured Edge HR."
             % (observations.loc[~complete, "watch_window_reason"].value_counts().to_dict(), int((complete & ~variable).sum())), "",
             "Adjustment at estimated seconds (signed mean, MAE, RMS, P95/max magnitude in BPM): "
             "`%s`. Instantaneous final HR minus orange dots at estimated observation centers: `%s`."
             % (magnitude, instantaneous), "",
             "The CSV stores every observation's raw/combined/final 60-second mean and residual, "
             "complete-window flag, exclusion reason and adjustable sample count, along with full-precision "
             "one-second HR, measured/estimated/missing provenance and terrain/support indicators.", "",
             "## Difference from earlier hard-anchored results", "",
             markdown(pd.DataFrame(comparison).round(5)), "",
             "Earlier final candidates preserve the same measured Edge prefix. The new fit, average-based "
             "training, removed taper/overwrite and soft adjustment change together; differences cannot be "
             "attributed to one component, and neither a smoother curve nor closer means establish true "
             "missing HR. The seven-ride gradient-boosting benchmark remains accessible and is not replaced.", "",
             "![Final HR, fixed zoom and mean consistency](images/hr_tcn_soft_average_2026-10-16.png)", "",
             "Zoom rule: central 20min of supported post-disconnection time, fixed independently of errors.", "",
             "## FIT audit: no uploadable export", "",
             "Original source: `data/mtb_edge.fit`, %d chronological unique records, %d recorded HR seconds. "
             "Official SDK %s; original header/CRC valid; file_id/activity/session/lap/record/timer messages present. "
             "The Encoder was tested in memory on original ordered unexpanded messages. It fails with: `%s`."
             % (audit["source_records"], audit["source_recorded_hr"], audit["sdk_version"], audit["encoder_error"]), "",
             "Unsupported message types and counts:", "",
             markdown(pd.DataFrame([dict(message=k, count=v, exported="not exported")
                                    for k, v in audit["unknown_message_counts"].items()])), "",
             "Additional unknown fields include record 107 (10,409 messages) and record 137/138/144 "
             "(1,525 each), plus session/lap/device/sport fields. Full inventory is in "
             "`results/final/fit_export_audit.json`. Garmin's Encoder ignores unrecognized fields; "
             "silently stripping these messages/fields is unsafe. No new binary encoder or unverified "
             "proprietary-field conversion is attempted. [Official Python SDK](https://github.com/garmin/fit-python-sdk), "
             "[Activity requirements](https://developer.garmin.com/fit/file-types/activity), "
             "[encoding cookbook](https://developer.garmin.com/fit/articles/cookbook/encoding_activity_files.html).", "",
             "| Check | Original | Export |", "|---|---|---|",
             "| Header/CRC and required Activity messages | Pass | Not run: no export |",
             "| Record chronology / duplicates | Unique, chronological | Not run: no export |",
             "| Route, timestamps, distance, speed, altitude, sensor/event/lap data | Original bytes unchanged | Not run: no export |",
             "| HR records, quantization and HR summaries | Original unchanged; CSV verified separately | Not run: no export |",
             "| Unknown messages/fields preserved | Present | Incompatible with installed Encoder |", "",
             "The requested `results/final/hr_reconstructed_2026-10-16.fit` is intentionally absent. "
             "No session/lap summaries, training effect, HRV, calories or proprietary metrics were recomputed "
             "or fabricated. A separate future lossless encoder capable of this exact Garmin payload would "
             "be needed, followed by every specified round-trip comparison. No file is ready for upload.", "",
             "## Manual import, only if a separately validated FIT later exists", "",
             "Do not upload this reconstruction CSV as an Activity FIT. For a future validated file: "
             "Strava web → plus icon → Upload Activity → File → Choose files "
             "([official instructions](https://support.strava.com/en-us/articles/15402173-how-do-i-bulk-upload-activities-to-strava)); "
             "Garmin Connect web → upload/cloud icon → Import Data → Browse → Import "
             "([official instructions](https://support.garmin.com/en-US/?faq=Ht3ZP52Kju075uKvqTqu99)). "
             "Inspect provenance/privacy before importing. Existing activity timestamps are unchanged: "
             "platforms may reject a duplicate or create another activity; import does not reliably update "
             "the existing one. Do not delete the original or shift its timestamps to bypass detection. "
             "A valid decode still cannot guarantee platform acceptance. No account was connected and "
             "no upload, publication, commit or push occurred.", "",
             "## Verification, reproduction and limitations", "",
             "Pre-fit deterministic checks passed for the exact centered mean, missing-window exclusion, "
             "no taper/overwrite, a known soft solution, missing-block preservation and frozen measurements. "
             "Actual GPU execution/RF, input/output shapes, all-seven-only normalization, hidden-HR substitution, "
             "finite HR, model reload, solver normal equations, exact measured HR/watch retention, original "
             "grid/support, CSV read-back and nonempty artifacts are checked in the script. "
             "%d prior files remain byte-identical, except the explicitly updated README/.gitignore. "
             "The generated figure is visually inspected after execution. FIT export checks above remain unperformed "
             "because encoding compatibility failed before file creation." % preserved, "",
             "One seed/fit, GPU nondeterminism, unknown wellness firmware filtering/latency, optical/reference "
             "sensor errors, uncertain terrain/effort relationships and absent target power/cadence remain. "
             "Physically plausible predictions and approximate local-mean agreement do not recover an unknown "
             "measured signal. This is the final experimental reconstruction, not a claim of model superiority. "
             "Implementation/documentation were AI-assisted; all numerical results come from local script execution.", "",
             "```powershell", "conda run --no-capture-output -n deep python scripts/reconstruct_hr_soft_average.py", "```", "",
             "The normal command refuses existing model/metadata to avoid another fit. "
             "`--reuse-trained` regenerates outputs using only this final model, without training. "
             "Model/normalization/audit are local under `results/final/`; the sensitive CSV and that folder are Git-ignored.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return counts, means, instantaneous, magnitude


def main():
    """Train once on average examples, reconstruct once, and stop unsafe export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-trained", action="store_true",
                        help="Regenerate outputs from this final model; never fit")
    args = parser.parse_args()
    if not args.reuse_trained:
        assert not MODEL.exists() and not STATE.exists(
        ), "Final fit already started; use --reuse-trained, never refit"
    FINAL.mkdir(parents=True, exist_ok=True)
    protected = [p for folder in ("data", "docs", "scripts", "results")
                 for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                 and p not in {CSV, REPORT, FIGURE, Path(__file__).resolve()} and FINAL not in p.parents]
    protected += [ROOT /
                  name for name in ("AGENTS.md", "requirements.txt", ".gitignore")]
    before = base.fingerprints(protected)
    best = search.read_json(search.RESULTS / "best.json")
    selected = search.read_json(search.RESULTS / "trial_10/status.json")
    assert best["trial_id"] == "trial_10" and {
        k: selected[k] for k in CONFIG} == CONFIG
    deterministic_checks()
    audit = fit_audit()
    info = previous.environment()
    previous.WIDTH, previous.OUTPUT, previous.LEFT = 128, 16, 56
    previous.TRAIN_STRIDE, previous.INFER_STRIDE = 16, 8
    watch, _, _ = window.clean_observations(pd.read_csv(
        ROOT / "data/processed/watch_hr_ride_2026-10-16.csv"))
    pattern = np.diff(watch.index.asi8) // 1_000_000_000
    assert len(watch) == 158 and (pattern == 60).sum(
    ) == 142 and (pattern == 120).sum() == 15
    if args.reuse_trained:
        payload = search.read_json(STATE)
        assert payload["fits"] == 1 and payload["assumption"] == ASSUMPTION
        assert payload["config"] == {**search.FIXED,
                                     **CONFIG} and payload["weights"] == WEIGHTS
        norm = payload["normalization"]
        network = tf.keras.models.load_model(MODEL, compile=False)
    else:
        rides, packages = {}, {}
        for name in RIDES:
            ride = window.prepare(name)
            assert "heart_rate" not in ride["frame"]
            packs = {}
            for phase in base.PHASES:
                times = base.sampling_times(
                    ride["reference"].index[0], ride["reference"].index[-1], pattern, phase)
                observations = mean_observations(
                    ride["reference"], times, ride["frame"])
                package = base.examples(ride["frame"], observations, ride["reference"].index,
                                        base.activity_features(ride["frame"]))
                package["features"] = previous.raw_features(
                    ride["frame"], package)
                packs[phase] = package
            rides[name], packages[name] = ride, packs
            print("Prepared averaged observations:", name, flush=True)
        # Reuse the normalizer's dictionary interface; every supplied value is an average.
        normalizer_input = {name: {(phase, "point"): pack for phase, pack in packs.items()}
                            for name, packs in packages.items()}
        norm = previous.normalization(RIDES, normalizer_input)
        assert norm["training"] == list(RIDES) and len(norm["features"]) == 28
        x, y, counts = training_arrays(rides, packages, norm)
        previous.seed()
        network = search.build_model(local, CONFIG)
        search.verify_actual_rf(local, network, CONFIG, float("inf"))
        payload = dict(status="training_started", fits=0, config={**search.FIXED, **CONFIG},
                       weights=WEIGHTS, assumption=ASSUMPTION, normalization=norm,
                       environment=info, counts=counts, initialization="fresh_seed_265",
                       imported_checkpoint=False,
                       source_sha256={str(p.relative_to(ROOT)): v for p, v in before.items()})
        search.write_json(STATE, payload)
        options = tf.data.Options()
        options.experimental_deterministic = True
        options.experimental_threading.private_threadpool_size = 2
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=265,
                                                                     reshuffle_each_iteration=True).batch(64).with_options(options).prefetch(2)
        started = time.monotonic()
        history = network.fit(dataset, epochs=30, verbose=0,
                              callbacks=[local.Progress()])
        duration = time.monotonic() - started
        history = {k: [float(v) for v in values]
                   for k, values in history.history.items()}
        assert all(len(v) == 30 and np.isfinite(v).all()
                   for v in history.values())
        network.save(MODEL, include_optimizer=False)
        payload.update(status="trained", fits=1,
                       history=history, training_seconds=duration)
        search.write_json(STATE, payload)
    target = window.prepare("mtb_edge")
    target["frame"] = target["strava"]
    assert "heart_rate" not in target["frame"] and "mtb_edge" not in norm["training"]
    package = target_inputs(target, watch)
    replaced = {**target, "reference": target["reference"] * 0 + 200}
    rebuilt = target_inputs(replaced, watch)
    pd.testing.assert_frame_equal(rebuilt["features"], package["features"])
    pd.testing.assert_frame_equal(rebuilt["baseline"], package["baseline"])
    fitted = dict(network=network, normalization=norm)
    raw = predict(target["frame"], package, fitted)
    assert raw.dropna().gt(0).all() and raw.dropna().lt(255).all()
    reloaded = tf.keras.models.load_model(MODEL, compile=False)
    assert all(np.array_equal(a, b)
               for a, b in zip(network.get_weights(), reloaded.get_weights()))
    sample = previous.normalized(package["features"], norm)[None, :128]
    assert np.allclose(network(sample, training=False).numpy(), reloaded(
        sample, training=False).numpy(), atol=1e-7, rtol=1e-7)
    repeated = predict(replaced["frame"], rebuilt, dict(
        network=reloaded, normalization=norm))
    pd.testing.assert_series_equal(raw, repeated, atol=1e-7, rtol=1e-7)
    reference = target["reference"].reindex(raw.index)
    error = (raw - reference).dropna()
    high = reference >= target["reference"].quantile(.9)
    overlap = summary_stats(error)
    overlap.update(high_hr_n=int(error.index.isin(reference[high].index).sum()),
                   high_hr_mae=float(error.reindex(
                       reference[high].index).abs().mean()),
                   reference_n=len(target["reference"]), coverage_percent=100 * len(error) / len(target["reference"]))
    raw_means = window_means(
        raw, watch).loc[target["reference"].index[0]:target["reference"].index[-1]]
    overlap.update(watch_window_n=int(raw_means.residual.notna().sum()),
                   watch_mean_bias=float(raw_means.residual.mean()),
                   watch_mean_mae=float(raw_means.residual.abs().mean()))
    table, adjustment_stats = make_table(target, watch, raw)
    assert len(table) == audit["source_records"]
    assert hashlib.sha256(table.index.asi8.tobytes(
    )).hexdigest() == audit["record_timestamp_sha256"]
    assert hashlib.sha256(np.nan_to_num(table.edge_hr.to_numpy(
        dtype=np.float64), nan=-1.).tobytes()).hexdigest() == audit["original_hr_sha256"]
    table.to_csv(CSV)
    saved = pd.read_csv(CSV)
    assert pd.DatetimeIndex(pd.to_datetime(
        saved.timestamp_utc, utc=True)).equals(table.index)
    for col in table.select_dtypes(include="number").columns:
        assert np.allclose(saved[col], table[col],
                           equal_nan=True, atol=1e-10, rtol=1e-10)
    assert np.array_equal(saved.loc[table.edge_hr.notna(
    ).to_numpy(), "final_combined_hr"], table.edge_hr.dropna())
    comparison = []
    for name in ("hr_seven_ride_candidate_2026-10-16.csv", "hr_tcn_tensorflow_candidate_2026-10-16.csv"):
        old = pd.read_csv(ROOT / "data/processed" / name)
        assert pd.DatetimeIndex(pd.to_datetime(
            old.timestamp_utc, utc=True)).equals(table.index)
        delta = table.final_combined_hr - old.final_hr.to_numpy()
        comparison.append(dict(previous=name, previous_mean=float(old.final_hr.mean()), previous_max=float(old.final_hr.max()),
                               **summary_stats(delta)))
    figures(table)
    assert base.fingerprints(protected) == before
    counts, means, instantaneous, magnitude = report(table, payload, audit, overlap, adjustment_stats,
                                                     comparison, len(protected))
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    heading = "## Final experimental reconstruction — soft wellness averages"
    marker_start = "<!-- soft-average-final:start -->"
    marker_end = "<!-- soft-average-final:end -->"
    final_text = (marker_start + "\n" + heading + "\n\n"
                  "The final [soft-average reconstruction report](docs/hr_final_soft_average_reconstruction.md) "
                  "uses one fresh trial-10 TCN trained on seven real rides. Watch summaries are assumed "
                  "centered 60-second averages and used as soft constraints; predictions need not pass through "
                  "their timestamps. Supported combined mean/max HR: %.2f/%.2f BPM; coverage %.2f%%. "
                  "Original Edge HR is preserved. The unknown signal has not been recovered as a measurement.\n\n"
                  "![Final experimental soft-average reconstruction](docs/images/hr_tcn_soft_average_2026-10-16.png)\n\n"
                  "FIT export is blocked by unsupported original Garmin messages/fields; no uploadable FIT "
                  "was produced. The existing gradient-boosting benchmark and all historical reports remain "
                  "accessible. Implementation was AI-assisted; results came from local script execution.\n"
                  % (counts["mean"], counts["maximum"], counts["coverage_percent"]))
    final_text += marker_end
    if marker_start in text and marker_end in text:
        left = text.index(marker_start)
        right = text.index(marker_end, left) + len(marker_end)
        text = text[:left] + final_text + text[right:]
    else:
        assert heading not in text, "Existing unmarked final section: preserve it for manual review"
        text = text.rstrip() + "\n\n" + final_text + "\n"
    readme.write_text(text, encoding="utf-8")
    payload.update(status="completed_export_blocked", overlap=overlap, final_summary=counts,
                   adjustment=adjustment_stats, instantaneous_dot_discrepancy=instantaneous,
                   adjustment_magnitude=magnitude)
    search.write_json(STATE, payload)
    assert all(p.stat().st_size > 0 for p in (
        CSV, REPORT, FIGURE, MODEL, STATE, AUDIT))
    assert not FIT_PATH.exists() and base.fingerprints(protected) == before
    print("ONE final fit;",
          payload["training_seconds"], "seconds. Overlap:", overlap)
    print("Final summary:", counts)
    print(means.round(5).to_string(index=False))
    print("Adjustment magnitude:", magnitude,
          "Dot discrepancy:", instantaneous)
    print("FIT export BLOCKED:", audit["encoder_error"])


if __name__ == "__main__":
    main()
