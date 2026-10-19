"""Audit movement and terrain predictors without estimating heart rate.

Run: .venv/Scripts/python scripts/audit_cycling_predictors.py
"""

import hashlib
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytz

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from diagnose_watch_anchors import decode


ROOT = Path(__file__).resolve().parents[1]
ZONE = pytz.timezone("Europe/Brussels")
RIDES = ("mtb_edge", "road_ride_1", "road_ride_2", "virtual_road_ride_1")
STEM = "cycling_predictor_quality"
LOW_SPEED = 0.3
STOP_SECONDS = 30


def gps_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in metres from FIT semicircles."""
    factor = np.pi / 2 ** 31
    p1, p2 = np.asarray(lat1) * factor, np.asarray(lat2) * factor
    dp = p2 - p1
    dl = (np.asarray(lon2) - np.asarray(lon1)) * factor
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371000 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def runs(mask, times):
    """Return true runs, always splitting at missing recorded seconds."""
    mask = np.asarray(mask, dtype=bool)
    breaks = np.r_[True, np.diff(times.asi8) != 1_000_000_000]
    group = np.cumsum(breaks | np.r_[True, mask[1:] != mask[:-1]])
    return [np.flatnonzero((group == k) & mask) for k in np.unique(group[mask])]


def timer_pauses(messages, end):
    """Read explicit timer stop/resume intervals without inventing missing ones."""
    result, start = [], None
    for event in messages.get("event_mesgs", []):
        if event.get("event") != "timer" or "timestamp" not in event:
            continue
        t = pd.Timestamp(event["timestamp"])
        kind = event.get("event_type")
        if kind in ("stop", "stop_all", "stop_disable", "stop_disable_all"):
            if start is None:
                start = t
        elif kind == "start" and start is not None:
            if t > start:
                result.append((start, t))
            start = None
    if start is not None and end > start:
        result.append((start, end))
    return result


def load_records(name):
    """Reuse the official FIT decoder and retain native units and UTC times."""
    messages = decode(ROOT / "data" / f"{name}.fit")
    frame = pd.DataFrame(messages["record_mesgs"])
    frame.index = pd.to_datetime(frame.timestamp, utc=True)
    frame = frame.sort_index()
    if not frame.index.is_unique:
        raise ValueError(f"Duplicate record timestamps: {name}")
    for target, source in (("speed_raw", "enhanced_speed"),
                           ("altitude_raw", "enhanced_altitude")):
        fallback = "speed" if target == "speed_raw" else "altitude"
        frame[target] = frame[source if source in frame else fallback]
    dt = frame.index.to_series().diff().dt.total_seconds()
    frame["dt"] = dt
    frame["distance_speed"] = (frame.distance.diff() / dt).where(
        dt.eq(1) & frame.distance.diff().ge(0))
    frame["gps_speed"] = pd.Series(gps_distance(
        frame.position_lat.shift(), frame.position_long.shift(),
        frame.position_lat, frame.position_long), index=frame.index).where(dt.eq(1))
    return frame, messages


def identify_stops(frame, virtual=False):
    """Classify low-speed runs using distance, position stability and duration."""
    low = frame.speed_raw.le(LOW_SPEED)
    state = pd.Series("riding", index=frame.index)
    state.loc[frame.speed_raw.gt(
        LOW_SPEED) & frame.speed_raw.le(1.5)] = "slow_movement"
    state.loc[frame.speed_raw.isna()] = "uncertain"
    rows = []
    stationary = pd.Series(False, index=frame.index)
    for indices in runs(low, frame.index):
        block = frame.iloc[indices]
        start, end = block.index[[0, -1]]
        duration = (end - start).total_seconds()
        radius = gps_distance(block.position_lat.median(),
                              block.position_long.median(),
                              block.position_lat, block.position_long)
        radius = float(np.nanmax(radius)) if np.isfinite(
            radius).any() else np.nan
        travel = block.distance.max() - block.distance.min()
        complete = block[["distance", "position_lat",
                          "position_long"]].notna().all().all()
        stable = complete and travel <= 3 and radius <= 8
        if virtual:
            kind = "platform_low_speed"
        elif stable:
            kind = "probable_stop" if duration >= STOP_SECONDS else "short_pause"
            stationary.iloc[indices] = True
        elif complete and (travel > 3 or radius > 8):
            kind = "slow_or_ambiguous_movement"
        else:
            kind = "uncertain"
        state.iloc[indices] = kind
        if duration >= 15:
            changes = block.altitude_raw.diff().abs()
            rows.append({
                "start": start, "end": end, "duration_s": duration,
                "classification": kind, "distance_range_m": travel,
                "confidence": "moderate" if stable and not virtual else "low_or_not_physical",
                "gps_radius_m": radius, "max_speed_m_s": block.speed_raw.max(),
                "altitude_change_m": block.altitude_raw.iloc[-1] - block.altitude_raw.iloc[0],
                "altitude_range_m": block.altitude_raw.max() - block.altitude_raw.min(),
                "max_altitude_step_m": changes.max(),
                "zero_speed_records": int(block.speed_raw.eq(0).sum()),
            })
    frame["movement_state"] = state
    frame["stationary_evidence"] = stationary
    frame["probable_stop"] = state.eq("probable_stop")
    return rows


def movement_segments(frame):
    """Split at all stationary/uncertain transitions and every recording gap."""
    # Low speed is not itself a stop classification, but terrain estimates at
    # near-zero movement are conservatively unavailable even if ambiguous.
    moving = frame.speed_raw.gt(LOW_SPEED) & ~frame.stationary_evidence
    return moving, runs(moving, frame.index)


def linear_trend(values, x, width):
    """Fit centered local trends on complete fixed-size windows only."""
    result = np.full(len(values), np.nan)
    half = width // 2
    for i in range(half, len(values) - (width - half - 1)):
        a, b = i - half, i - half + width
        y, xx = values[a:b], x[a:b]
        if np.isfinite(y).all():
            xx = xx - xx.mean()
            denominator = np.dot(xx, xx)
            if denominator > 0:
                result[i] = np.dot(xx, y - y.mean()) / denominator
    return result


def terrain_features(frame):
    """Compare robust diagnostics, masking stops, gaps and elevation flags."""
    moving, segments = movement_segments(frame)
    dh = frame.altitude_raw.diff()
    frame["altitude_flag"] = frame.dt.eq(1) & dh.abs().gt(2)
    # The median is a diagnostic, never a correction of the source altitude.
    for column in ("altitude_median5", "speed_mean5", "speed_mean10",
                   "climb15", "climb30", "climb15_raw", "climb30_raw",
                   "gradient20", "gradient50", "gradient20_raw", "gradient50_raw"):
        frame[column] = np.nan
    # Separate moving, stopped and uncertain runs so smoothing cannot blend
    # either side of a stop, nor cross even a one-second recording gap.
    smooth_group = ((frame.dt.ne(1)) | moving.ne(moving.shift())).cumsum()
    for _, part in frame.groupby(smooth_group):
        frame.loc[part.index, "altitude_median5"] = part.altitude_raw.rolling(
            5, center=True, min_periods=5).median()
        for width in (5, 10):
            frame.loc[part.index, f"speed_mean{width}"] = part.speed_raw.rolling(
                width, center=True, min_periods=width).mean()
    for indices in segments:
        part = frame.iloc[indices]
        times = (part.index.asi8 - part.index.asi8[0]) / 1e9
        raw = part.altitude_raw.to_numpy()
        filtered = part.altitude_median5.to_numpy()
        flags = part.altitude_flag.to_numpy()
        distance = part.distance.to_numpy()
        for width in (15, 30):
            for values, suffix in ((raw, "_raw"), (filtered, "")):
                trend = linear_trend(values, times, width)
                # Suppress any trend window containing a flagged altitude
                # step, including a 30-second caution margin on either side.
                bad = pd.Series(flags).rolling(61 + width, center=True,
                                               min_periods=1).max().astype(bool)
                trend[bad] = np.nan
                frame.loc[part.index, f"climb{width}{suffix}"] = trend
        if not np.isfinite(distance).all() or np.any(np.diff(distance) < 0):
            continue
        for width in (20, 50):
            outputs = {"": np.full(len(part), np.nan),
                       "_raw": np.full(len(part), np.nan)}
            for i, center in enumerate(distance):
                left = np.searchsorted(
                    distance, center - width / 2, side="right") - 1
                right = np.searchsorted(
                    distance, center + width / 2, side="left")
                if left < 0 or right >= len(part):
                    continue
                if times[right] - times[left] > 120:
                    continue
                # Conservative caution around flagged step timestamps.
                a = max(0, left - 30)
                b = min(len(part), right + 31)
                if flags[a:b].any():
                    continue
                x = distance[left:right + 1]
                assert x[-1] - x[0] >= width - 1e-6
                x = x - x.mean()
                for suffix, values in (("", filtered), ("_raw", raw)):
                    y = values[left:right + 1]
                    if np.isfinite(y).all() and np.dot(x, x) > 0:
                        outputs[suffix][i] = 100 * \
                            np.dot(x, y - y.mean()) / np.dot(x, x)
            for suffix, values in outputs.items():
                frame.loc[part.index, f"gradient{width}{suffix}"] = values
    for column in ("climb15", "climb30", "gradient20", "gradient50"):
        assert frame.loc[~moving, column].isna().all()
    return segments


def consistency(frame, mask, column="distance_speed"):
    """Compare speeds only at matching observed one-second intervals."""
    paired = frame.loc[mask, ["speed_raw", column]].dropna()
    error = paired[column] - paired.speed_raw
    return {"n": len(paired), "mae": error.abs().mean(),
            "p95": error.abs().quantile(.95), "bias": error.mean()}


def describe(name, frame, stops, messages):
    """Summarize availability, quality, and observed versus inferred time."""
    duration = (frame.index[-1] - frame.index[0]).total_seconds()
    probable = [r for r in stops if r["classification"] == "probable_stop"]
    zero = frame.speed_raw.eq(0)
    pairs = frame.dt.eq(1)
    summary = {
        "name": name, "records": len(frame), "duration_s": duration,
        "start": frame.index[0].tz_convert(ZONE).isoformat(),
        "end": frame.index[-1].tz_convert(ZONE).isoformat(),
        "record_coverage": 100 * len(frame) / (duration + 1),
        "missing_seconds": int(duration + 1 - len(frame)),
        "gaps": int(frame.dt.gt(1).sum()),
        "zero_records": int(zero.sum()), "zero_percent": 100 * zero.mean(),
        "stop_n": len(probable), "stop_seconds": sum(r["duration_s"] for r in probable),
        "zero_explained": int((zero & frame.probable_stop).sum()),
        "zero_short_pause": int((zero & frame.movement_state.eq("short_pause")).sum()),
        "altitude_flags": int(frame.altitude_flag.sum()),
        "closure_distance_m": float(gps_distance(
            frame.position_lat.iloc[0], frame.position_long.iloc[0],
            frame.position_lat.iloc[-1], frame.position_long.iloc[-1])),
        "altitude_change_m": frame.altitude_raw.iloc[-1] - frame.altitude_raw.iloc[0],
        "pauses": timer_pauses(messages, frame.index[-1]),
        "speed": consistency(frame, pairs), "gps": consistency(frame, pairs, "gps_speed"),
        "availability": {c: int(frame[c].notna().sum()) if c in frame else 0
                         for c in ("position_lat", "altitude_raw", "speed_raw",
                                   "distance", "cadence", "power")},
    }
    gap_times = frame.index[frame.dt.gt(1)]
    summary["gaps_with_timer_pause"] = sum(
        any(a < t and b > t - pd.Timedelta(seconds=frame.loc[t, "dt"])
            for a, b in summary["pauses"]) for t in gap_times)
    frame["transition"] = False
    changes = frame.stationary_evidence.ne(frame.stationary_evidence.shift())
    for t in frame.index[changes]:
        frame.loc[t - pd.Timedelta(seconds=5):t +
                  pd.Timedelta(seconds=5), "transition"] = True
    summary["state_speeds"] = {
        state: consistency(frame, pairs & mask)
        for state, mask in {
            "riding (>1.5m/s)": frame.speed_raw.gt(1.5),
            "slow/technical candidate": frame.movement_state.isin(
                ["slow_movement", "slow_or_ambiguous_movement"]),
            "probable stop": frame.probable_stop,
            "start/stop +/-5s": frame.transition,
        }.items()
    }
    summary["features"] = {}
    for column in ("speed_mean5", "speed_mean10", "climb15", "climb30",
                   "gradient20", "gradient50"):
        values = frame[column]
        delta = values.diff().where(pairs).abs()
        raw_column = column + "_raw"
        difference = (values - frame[raw_column]).abs() if raw_column in frame else (
            values - frame.speed_raw).abs()
        summary["features"][column] = {
            "coverage": 100 * values.notna().mean(),
            "moving_coverage": 100 * values[frame.speed_raw.gt(LOW_SPEED)].notna().mean(),
            "p95_abs": values.abs().quantile(.95),
            "p95_step": delta.quantile(.95),
            "raw_filtered_mae": difference.mean(),
        }
    return summary


def plot_mtb(frame, stops, pauses, path):
    """Show raw terrain, inferred movement, and masked candidate features."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True,
                             constrained_layout=True)
    t = frame.index
    axes[0].plot(t, frame.altitude_raw, color=".6",
                 lw=1.4, label="Raw altitude")
    axes[0].plot(t, frame.altitude_median5, color="#276c9b", lw=.9, ls="--",
                 label="Diagnostic 5s median (no drift correction)")
    flags = frame.altitude_flag
    axes[0].scatter(t[flags], frame.loc[flags, "altitude_raw"], color="#b33437",
                    marker="x", s=35, label="Altitude step >2m in 1s", zorder=5)
    axes[0].set_ylabel("Altitude (m)")
    axes[0].legend(loc="upper left", fontsize=8)
    zoom = axes[0].inset_axes([.54, .08, .31, .36])
    detail = frame.loc["2026-10-16 18:33:55+00:00":"2026-10-16 18:34:30+00:00"]
    zoom.plot(detail.index, detail.altitude_raw, color=".5", lw=1.4)
    zoom.plot(detail.index, detail.altitude_median5,
              color="#276c9b", ls="--", lw=1)
    flagged = detail.altitude_flag
    zoom.scatter(detail.index[flagged], detail.loc[flagged, "altitude_raw"],
                 color="#b33437", marker="x", s=15)
    zoom.set_title("20:34 detail (raw and median)", fontsize=8)
    zoom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=ZONE))
    zoom.tick_params(labelsize=7)
    zoom.grid(alpha=.2)
    axes[1].plot(t, frame.speed_raw, color=".5", lw=.9, label="Recorded speed")
    axes[1].plot(t, frame.distance_speed, color="#d5771d", lw=.7, alpha=.7,
                 label="From recorded distance / elapsed time")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].legend(loc="upper left", fontsize=8)
    twin = axes[2].twinx()
    for column, color, style in (("climb15", "#9c4ba2", "-"),
                                 ("climb30", "#442469", "--")):
        axes[2].plot(t, frame[column], color=color, ls=style, lw=.8,
                     label=f"{column[-2:]}s climbing rate")
    for column, color, style in (("gradient20", "#61a965", "-"),
                                 ("gradient50", "#05614d", "--")):
        twin.plot(t, frame[column], color=color, ls=style, lw=.7, alpha=.8,
                  label=f"{column[-2:]}m gradient")
    axes[2].set_ylabel("Climbing rate (m/s)")
    twin.set_ylabel("Gradient (%)")
    lines, labels = axes[2].get_legend_handles_labels()
    other, other_labels = twin.get_legend_handles_labels()
    axes[2].legend(lines + other, labels + other_labels, loc="upper left",
                   ncol=2, fontsize=8)
    bad = frame[["climb30", "gradient50"]].isna().any(axis=1)
    axes[2].fill_between(t, 0, .045, where=bad.to_numpy(),
                         transform=axes[2].get_xaxis_transform(), color=".35",
                         alpha=.7, label="Terrain feature unavailable")
    anomaly = pd.Timestamp("2026-10-16 20:34:15", tz=ZONE)
    for i, ax in enumerate(axes):
        for row in stops:
            if row["classification"] == "probable_stop":
                ax.axvspan(row["start"], row["end"],
                           color="#c6b16c", alpha=.18)
        for a, b in pauses:
            ax.axvspan(a, b, color=".4", alpha=.2, hatch="//")
        ax.axvline(anomaly, color="#b33437", ls=":", lw=1)
        ax.grid(alpha=.15)
        ax.set_title(("MTB elevation; dotted red line: 20:34 anomaly",
                      "Movement; gold shading: inferred sustained stops; no internal timer pauses",
                      "Terrain candidates; dark bottom strip: 30s climb or 50m gradient unavailable")[i],
                     fontsize=10)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZONE))
    axes[2].set_xlabel("Local time — Europe/Brussels (UTC+02:00)")
    axes[2].set_xlim(t[0], t[-1])
    fig.suptitle("Cycling predictor quality — October 16 MTB\n"
                 "Diagnostic features only; stationary intervals retain their elapsed time",
                 fontsize=14)
    fig.savefig(path, dpi=210)
    plt.close(fig)


def write_report(summaries, all_stops, frames, path):
    """Write the evidence, conservative stop table, and predictor limitations."""
    mtb = summaries[0]
    frame = frames[0]
    stops = all_stops[0]
    probable = [r for r in stops if r["classification"] == "probable_stop"]
    minutes = mtb["stop_seconds"] / 60
    lines = ["# Cycling predictor quality and MTB stops", "",
             f"**MTB: {mtb['stop_n']} probable sustained stops, "
             f"{mtb['stop_seconds']:.0f}s ({minutes:.2f} min), "
             f"{100 * mtb['stop_seconds'] / mtb['duration_s']:.2f}% of "
             "recorded elapsed duration.** These are inferred stationary "
             "intervals, not identified ravitos. Five ravitos and additional "
             "waits are known from the rider's account; their identities and "
             "boundaries cannot be recovered from these channels alone.", "",
             "## Data and recording quality", "",
             "Existing official Garmin SDK decoder reused; UTC internally, "
             "Europe/Brussels (UTC+02:00) for displays. Source FITs and existing "
             "HR artifacts remain unchanged. No HR model or HR estimates "
             "are produced. FIT scaling is applied by the SDK: speed m/s, "
             "distance/elevation metres, cadence rpm, power watts. GPS "
             "semicircles convert with π/2³¹ to radians for great-circle metres.", "",
             "| Activity | Records | Elapsed min | Missing s / gaps / coverage % | Speed zero % | Probable observed stop seconds | Altitude flags | Distance-speed MAE / P95 (m/s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        stationary = "not physical" if s["name"].startswith(
            "virtual") else f"{s['stop_seconds']:.0f}"
        lines.append(f"| {s['name']} | {s['records']} | {s['duration_s']/60:.2f} "
                     f"| {s['missing_seconds']} / {s['gaps']} / {s['record_coverage']:.2f} | {s['zero_percent']:.2f} "
                     f"| {stationary} | {s['altitude_flags']} | "
                     f"{s['speed']['mae']:.3f} / {s['speed']['p95']:.3f} |")
    lines += [""]
    for s in summaries:
        lines.append(f"- {s['name']}: {s['start']}–{s['end']}.")
    lines += ["", "Missing seconds are absent integer-second timestamps "
              "between the first/last record; they are not labelled observed "
              "stops. All recorded fields are nominally 1Hz. Outdoor timer "
              "pauses explain much of the discontinuous recording but do not "
              "observe physical motion during the gaps. MTB has one timer "
              "start and final stop only: genuine physical stops occurred "
              "while recording continued.", "",
              "| Activity | GPS / elevation / speed / distance valid rows | Cadence / power rows | Resumed timer-pause intervals |",
              "|---|---|---|---:|"]
    for s in summaries:
        a = s["availability"]
        lines.append(f"| {s['name']} | {a['position_lat']} / {a['altitude_raw']} / "
                     f"{a['speed_raw']} / {a['distance']} | {a['cadence']} / "
                     f"{a['power']} | {len(s['pauses'])} |")
    lines += ["", "Recording gaps intersecting explicit resumed timer pauses: "
              + "; ".join(f"{s['name']} {s['gaps_with_timer_pause']}/{s['gaps']}"
                          for s in summaries[1:3])
              + ". Other gaps are missing records of unknown cause, not "
              "automatically observed stops."]
    lines += ["", "Outdoor Edge records include an ANT+ HR device labelled "
              "OHR and barometer metadata, but do not establish that the "
              "exported altitude is error-free. Cadence/power are absent from "
              "all outdoor files. MyWhoosh coordinates, altitude, distance "
              "and speed describe platform terrain/motion, not physical GPS "
              "or vertical travel. Its cadence and power may represent "
              "physical effort, but the FIT metadata does not identify their "
              "hardware source. Virtual terrain statistics below are numerical "
              "diagnostics only, not transferable outdoor workload validation.", "",
              "## Stop criteria and chronological MTB candidates", "",
              "A candidate is a consecutive observed run with speed ≤0.3m/s. "
              "A stationary classification additionally requires complete "
              "position/distance, cumulative-distance range ≤3m, and maximum "
              "GPS radius ≤8m around the median position. A run ≥30 elapsed "
              "seconds is a **probable sustained stop**; a shorter supported "
              "run is a short pause/transition. No runs are merged through "
              "evidence of movement. The table retains candidates ≥15s; "
              "shorter runs remain represented internally.", "",
              "Very slow intervals with appreciable travel or GPS spread "
              "are slow/ambiguous movement, not stops. Riding at 0.3–1.5m/s "
              "is labelled slow movement without claiming it is a technical "
              "climb. Thresholds are conservative operational definitions, "
              "not externally validated classifiers. Distance, speed and GPS "
              "may share firmware processing; frozen coordinates are "
              "consistent with stopping but are not three independent proofs.", "",
              "Duration is end timestamp minus start timestamp, never row "
              "count. Stop boundaries have at least sample-level uncertainty; "
              "slow manoeuvres may split one real-world visit into several "
              "candidate stops. The recording cannot establish their purposes.", "",
              "Confidence is moderate when all export channels agree, because "
              "they may be coupled. There is no independent stop ground truth.", "",
              "| Local start–end | Elapsed s | Classification | Confidence | Distance range m | Max GPS radius m | Altitude change / range m |",
              "|---|---:|---|---|---:|---:|---:|"]
    for row in stops:
        a = row["start"].tz_convert(ZONE).strftime("%H:%M:%S")
        b = row["end"].tz_convert(ZONE).strftime("%H:%M:%S")
        lines.append(f"| {a}–{b} | {row['duration_s']:.0f} | "
                     f"{row['classification']} | {row['confidence']} | {row['distance_range_m']:.2f} | "
                     f"{row['gps_radius_m']:.2f} | {row['altitude_change_m']:+.2f} "
                     f"/ {row['altitude_range_m']:.2f} |")
    lines += ["", f"Zero-speed records: **{mtb['zero_records']} "
              f"({mtb['zero_percent']:.2f}% of records)**. Of these, "
              f"**{mtb['zero_explained']} "
              f"({100 * mtb['zero_explained'] / mtb['zero_records']:.2f}%)** "
              "lie in probable sustained stops. This record-count fraction "
              "is distinct from elapsed stationary duration. Supported "
              f"short pauses account for another {mtb['zero_short_pause']} "
              "zero-speed records. Durations should not be interpreted as "
              "exact visit boundaries.", "",
              "For threshold sensitivity, requiring at least 60s instead "
              f"of 30s retains {sum(r['duration_s'] >= 60 for r in probable)} "
              "intervals totalling "
              f"{sum(r['duration_s'] for r in probable if r['duration_s'] >= 60):.0f}s. "
              "This is a sensitivity check, not an alternative selected to "
              "match the number of known ravitos.", "",
              "## Elevation findings", "",
              f"MTB raw altitude starts at {frame.altitude_raw.iloc[0]:.1f}m "
              f"and ends at {frame.altitude_raw.iloc[-1]:.1f}m "
              f"(difference {mtb['altitude_change_m']:+.1f}m), while endpoints "
              f"are {mtb['closure_distance_m']:.1f}m apart horizontally. "
              "The discrepancy supports caution about altitude calibration "
              "or drift; it does not identify their cause or justify forcing "
              "loop closure with an arbitrary correction.", "",
              "The previously stated 13.4m jump needs a boundary clarification: "
              "altitude is 135.2m at **20:34:13** and 148.6m at **20:34:16**, "
              "a 13.4m increase over 10.14m recorded travel. From the "
              "20:34:14 sample itself to 20:34:16 the increase is 9.0m. "
              "Speed is ~3.4m/s, GPS/distance advance, there is no timestamp "
              "gap or pause, and it is not classified as a stop. The implied "
              "slope is highly suspicious; actual terrain versus altitude "
              "offset/noise cannot be established without independent "
              "elevation. The diagnostic flags it instead of correcting it.", ""]
    if probable:
        worst = max(probable, key=lambda r: abs(r["altitude_change_m"]))
        lines += [f"During the probable stop beginning "
                  f"{worst['start'].tz_convert(ZONE).strftime('%H:%M:%S')}, "
                  f"altitude changes {worst['altitude_change_m']:+.1f}m "
                  f"over {worst['duration_s']:.0f}s while distance range is "
                  f"{worst['distance_range_m']:.2f}m and GPS radius "
                  f"{worst['gps_radius_m']:.2f}m. This is evidence consistent "
                  "with measurement/calibration drift during likely "
                  "stationarity, but not proof of a particular sensor cause. "
                  "Stable/frozen horizontal GPS alone does not guarantee "
                  "numerically constant altitude.", ""]
    abrupt_stops = [r for r in probable if r["max_altitude_step_m"] > 2]
    if abrupt_stops:
        lines += ["Abrupt altitude behaviour also occurs within probable "
                  "stops: " + "; ".join(
                      f"{r['start'].tz_convert(ZONE).strftime('%H:%M:%S')}–"
                      f"{r['end'].tz_convert(ZONE).strftime('%H:%M:%S')}, "
                      f"largest adjacent step {r['max_altitude_step_m']:.1f}m"
                      for r in abrupt_stops) + ". The net change alone "
                  "would hide this short-timescale variation.", ""]
    lines += ["The altitude flag is an adjacent observed change >2m in 1s; "
              "it is a screening threshold, not proof that every flagged "
              "change is false or every unflagged point is accurate. "
              "A centered five-sample median is shown as diagnostic filtered "
              "altitude. It suppresses isolated noise, not persistent offsets "
              "or slow drift. Source altitude remains unchanged.", "",
              "Only MTB has >2m adjacent altitude steps among the outdoor "
              "records. MyWhoosh's 96 threshold exceedances describe virtual "
              "terrain/export behaviour and must not be read as physical "
              "altimeter errors. Even unflagged MTB altitude can remain biased.", "",
              "## Feature definitions, stability and usable coverage", "",
              "Speed candidates are centered 5/10-sample means. Climbing "
              "rates are least-squares altitude-versus-time trends over "
              "15/30 consecutive samples (14/29s endpoint spans). Gradient "
              "uses local altitude-versus-cumulative-distance regression, "
              "bracketing approximately ±10/±25m; actual support spans at "
              "least 20/50m because of discrete records. Distance windows "
              "taking >120s are unavailable. No near-zero denominator is used.", "",
              "This is rise divided by recorded path distance, an approximate "
              "gradient; the export does not certify a horizontal-distance "
              "denominator. Small road grades are less sensitive to that "
              "distinction than extreme slopes.", "",
              "All smoothing/trend windows split at recording gaps and "
              "near-zero movement transitions; full window coverage is "
              "required. Terrain features are missing during stationary "
              "evidence and speed ≤0.3m/s, not calculated from altitude drift. "
              "Flagged altitude steps have an additional 30s caution margin "
              "plus the feature's support window. Legitimate steep climbs "
              "are not clipped or flattened. Elapsed stop timestamps stay "
              "on the modelling timeline with movement/quality labels.", "",
              "Coverage is the percentage of recorded rows with a finite "
              "candidate, not a certification of terrain truth. P95 step "
              "is the 95th percentile absolute consecutive change. Units: "
              "m/s for speed/climbing rate; percentage points for gradient.", "",
              "| Activity | Candidate | Coverage all / moving % | P95 absolute value | P95 1s change | Raw-v-filtered MAE |",
              "|---|---|---:|---:|---:|---:|"]
    for s in summaries:
        for feature, values in s["features"].items():
            lines.append(f"| {s['name']} | {feature} | {values['coverage']:.2f} / {values['moving_coverage']:.2f} "
                         f"| {values['p95_abs']:.3f} | {values['p95_step']:.3f} "
                         f"| {values['raw_filtered_mae']:.5f} |")
    lines += ["", "Prefer 30s climbing rate and 50m gradient as the "
              "conservative candidates if their lower variability and usable "
              "coverage are acceptable; retain 15s/20m as sensitivity checks "
              "for sharper terrain changes. Longer windows naturally suppress "
              "real variation as well as noise, so numerical smoothness is "
              "not evidence of superior accuracy. No independent terrain "
              "reference establishes an optimal window. Use the 5s speed "
              "summary initially; 10s trades extra smoothness for slower "
              "start/stop response. All are candidates, not calibrated settings.", "",
              "Moving coverage uses only recorded speed >0.3m/s; overall "
              "coverage includes genuine stops, where terrain features are "
              "deliberately unavailable. The near-zero raw/median trend "
              "difference shows that this filter does not resolve the "
              "underlying terrain uncertainty. Large unflagged slopes still "
              "require caution; no maximum slope was imposed to improve "
              "appearance or flatten real terrain.", "",
              "## Speed/distance consistency by movement context", "",
              "Derived speed is Δ recorded cumulative distance / actual Δt; "
              "only consecutive 1s observations enter comparisons. Gaps are "
              "not averaged into observed speeds. Bias is derived minus "
              "recorded. The channels may be coupled internally, so close "
              "agreement does not independently prove physical accuracy. "
              "Rows with n=0 have undefined (nan) metrics, not zero error.", "",
              "| Activity | Context | n | MAE m/s | P95 absolute error m/s | Bias m/s |",
              "|---|---|---:|---:|---:|---:|"]
    for s in summaries:
        for state, values in s["state_speeds"].items():
            lines.append(f"| {s['name']} | {state} | {values['n']} | "
                         f"{values['mae']:.3f} | {values['p95']:.3f} | {values['bias']:.3f} |")
    lines += ["", "Outdoor GPS step-speed versus recorded speed (additional, "
              "noisier position-based diagnostic): " + "; ".join(
                  f"{s['name']} MAE {s['gps']['mae']:.3f}m/s, "
                  f"P95 {s['gps']['p95']:.3f}m/s" for s in summaries[:3]) + ". "
              "GPS scatter, quantization, smoothing and frozen positions "
              "limit interpretation. Virtual coordinates are not used as "
              "independent physical evidence.", "",
              "## Recommendation and limitations", "",
              "Use elapsed time, raw/5s speed, movement state and explicit "
              "quality flags as the small common outdoor predictor set. "
              "Add positive/negative 30s climbing rate only on supported "
              "segments; use 50m gradient cautiously as a sensitivity input "
              "rather than assume it measures workload. Do not require "
              "power/cadence. Treat MyWhoosh terrain as a different domain.", "",
              "Retain states for riding, slow movement, probable stop, "
              "short pause, uncertain movement and recording gaps. Do not "
              "equate a stationary state with zero cardiovascular effort "
              "or reset physiological history. HR recovery may continue "
              "through ravitos and waits; no HR is estimated here.", "",
              "The speed/movement predictors support a small next modelling "
              "experiment. Terrain predictors require an ablation with and "
              "without them; they do not yet support strong accuracy claims "
              "or confident short-peak recovery. Independent altitude/route "
              "evidence and confirmation of stationary intervals would help "
              "distinguish drift from actual terrain. Five ravitos cannot "
              "be identified by ranking the five longest stops.", "",
              "## Reproduction and checks", "",
              "```powershell",
              ".\\.venv\\Scripts\\python.exe scripts/audit_cycling_predictors.py",
              "```", "",
              f"Figure: `docs/images/{STEM}.png`. Optional compact CSV: "
              f"`data/processed/{STEM}_summary.csv` (MTB candidate stops and "
              "feature summary statistics only). No dense intermediates.", "",
              "Verified real-file decoding, UTC/local conversions, elapsed "
              "stop durations, gap splitting, full smoothing support, "
              "stationary feature masking, minimum gradient distance and "
              "unchanged original FIT/HR output hashes. No reconstruction, "
              "model fitting, new dependencies, commits or uploads.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Execute the diagnostic on all four activities and verify its outputs."""
    protected = list((ROOT / "data").rglob("*.fit"))
    protected += [p for folder in ("docs", "data/processed")
                  for p in (ROOT / folder).rglob("*") if p.is_file()
                  and ("hr_" in p.name or "watch_" in p.name)]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    frames, summaries, all_stops = [], [], []
    for name in RIDES:
        frame, messages = load_records(name)
        stops = identify_stops(frame, virtual=name.startswith("virtual"))
        terrain_features(frame)
        summary = describe(name, frame, stops, messages)
        duration_from_pairs = frame.dt.where(
            frame.probable_stop & frame.probable_stop.shift().fillna(False)
            & frame.dt.eq(1), 0).sum()
        assert duration_from_pairs == summary["stop_seconds"]
        for row in stops:
            assert row["duration_s"] == (
                row["end"] - row["start"]).total_seconds()
            assert frame.loc[row["start"]:row["end"],
                             "dt"].iloc[1:].eq(1).all()
        assert (frame.distance.diff().dropna() >= 0).all()
        frames.append(frame)
        summaries.append(summary)
        all_stops.append(stops)
    assert len(frames[0]) == 10409
    assert frames[0].speed_raw.eq(0).sum() == 3669
    assert (frames[0].index[-1] - frames[0].index[0]).total_seconds() == 10408
    event = frames[0].loc["2026-10-16 18:34:13+00:00":"2026-10-16 18:34:16+00:00"]
    assert np.isclose(
        event.altitude_raw.iloc[-1] - event.altitude_raw.iloc[0], 13.4)
    assert np.isclose(event.distance.iloc[-1] - event.distance.iloc[0], 10.14)
    assert event[["climb15", "climb30", "gradient20",
                  "gradient50"]].isna().all().all()
    assert np.isclose(gps_distance(0, 0, 0, 2 ** 30), np.pi * 6371000 / 2)
    report = ROOT / "docs" / f"{STEM}.md"
    figure = ROOT / "docs/images" / f"{STEM}.png"
    write_report(summaries, all_stops, frames, report)
    plot_mtb(frames[0], all_stops[0], summaries[0]["pauses"], figure)
    rows = [{"record_type": "mtb_stop", **row} for row in all_stops[0]]
    rows += [{"record_type": "feature_summary", "activity": s["name"],
              "feature": feature, **values} for s in summaries
             for feature, values in s["features"].items()]
    csv = ROOT / "data/processed" / f"{STEM}_summary.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    assert all(p.stat().st_size > 0 for p in (report, figure, csv))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()
               == h for p, h in hashes.items())
    print(pd.DataFrame([{k: s[k] for k in (
        "name", "records", "gaps", "stop_n", "stop_seconds", "zero_records",
        "zero_explained", "altitude_flags")} for s in summaries]).to_string(index=False))
    print("Verified elapsed durations, gap splits, units, feature masking, "
          "minimum slope support, output existence and unchanged FIT/HR files.")


if __name__ == "__main__":
    main()
