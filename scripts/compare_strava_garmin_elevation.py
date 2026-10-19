"""Compare exported Strava elevation with Garmin terrain diagnostics.

Run: .venv/Scripts/python scripts/compare_strava_garmin_elevation.py
No source files are edited; no elevation or HR model is fitted.
"""

import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from audit_cycling_predictors import (
    ROOT, ZONE, gps_distance, identify_stops, load_records, runs,
    terrain_features,
)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


STEM = "strava_vs_garmin_elevation_2026-10-16"
NS = {"g": "http://www.topografix.com/GPX/1/1"}
COLORS = {"Garmin": "#b14d32", "Strava": "#167ba0"}
FEATURES = ("climb30", "gradient50")


def hashes(paths):
    """Fingerprint source files and existing analyses before and after running."""
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_gpx(path):
    """Read observed GPX points without resampling or shifting timestamps."""
    root = ET.parse(path).getroot()
    rows = []
    for segment, node in enumerate(root.findall(".//g:trkseg", NS)):
        for point in node.findall("g:trkpt", NS):
            rows.append({
                "timestamp": point.findtext("g:time", namespaces=NS),
                "altitude_raw": point.findtext("g:ele", namespaces=NS),
                "lat": point.get("lat"), "lon": point.get("lon"),
                "segment": segment,
            })
    frame = pd.DataFrame(rows)
    frame.index = pd.to_datetime(frame.pop("timestamp"), utc=True)
    for column in ("altitude_raw", "lat", "lon"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.index.hasnans or not frame.index.is_unique:
        raise ValueError(
            "GPX timestamps are missing or duplicated; inspect alignment")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(
            "GPX timestamps are unordered; inspect before matching")
    return frame, root


def source_features(timeline, elevation, segments):
    """Reuse audit preprocessing, splitting additionally at absent GPX support."""
    frame = timeline.copy()
    frame["altitude_raw"] = elevation
    frame["source_segment"] = segments
    valid = elevation.notna() & segments.notna()
    groups = (frame.dt.ne(1) | segments.ne(segments.shift())
              | valid.ne(valid.shift())).cumsum()
    for column in ("altitude_median5", "climb30", "gradient50"):
        frame[column] = np.nan
    frame["altitude_flag"] = False
    for _, part in frame.loc[valid].groupby(groups[valid]):
        part = part.copy()
        part.loc[part.index[0], "dt"] = np.nan
        terrain_features(part)
        for column in ("altitude_median5", "climb30", "gradient50",
                       "altitude_flag"):
            frame.loc[part.index, column] = part[column]
    for column in FEATURES:
        assert frame.loc[frame.speed_raw.le(.3)
                         | frame.stationary_evidence, column].isna().all()
    return frame


def altitude_summary(frame):
    """Sum signed increments only on consecutive, finite recorded seconds."""
    altitude = frame.altitude_raw
    delta = altitude.diff().where(
        frame.dt.eq(1) & frame.source_segment.eq(frame.source_segment.shift()))
    return [altitude.iloc[0], altitude.iloc[-1], altitude.min(), altitude.max(),
            delta.clip(lower=0).sum(), -delta.clip(upper=0).sum(),
            int(delta.abs().gt(2).sum())]


def verify_support(timeline, sources):
    """Probe missing support and check an independent distance-gradient fit."""
    moving = timeline.speed_raw.gt(.3) & ~timeline.stationary_evidence
    indices = max(runs(moving, timeline.index), key=len)
    block = timeline.iloc[indices[:180]].copy()
    assert len(block) == 180
    midpoint = len(block) // 2
    elevation = block.altitude_raw.copy()
    elevation.iloc[midpoint] = np.nan
    segments = pd.Series(0, index=block.index)
    segments.iloc[midpoint:] = 1
    probe = source_features(block, elevation, segments)
    # A hole and segment transition require fresh full support on each side.
    assert probe.climb30.iloc[midpoint - 14:midpoint + 18].isna().all()
    for side in (probe.iloc[:midpoint], probe.iloc[midpoint + 1:]):
        usable = side.gradient50.notna()
        distance = side.distance[usable]
        assert distance.ge(side.distance.iloc[0] + 25).all()
        assert distance.le(side.distance.iloc[-1] - 25).all()
    for frame in sources.values():
        candidates = np.flatnonzero(frame.gradient50.notna().to_numpy())
        # Independently refit a few observed support windows using polyfit.
        for i in candidates[::max(1, len(candidates) // 10)]:
            left = np.searchsorted(frame.distance, frame.distance.iloc[i] - 25,
                                   side="right") - 1
            right = np.searchsorted(frame.distance, frame.distance.iloc[i] + 25,
                                    side="left")
            part = frame.iloc[left:right + 1]
            assert (part.index[-1] - part.index[0]).total_seconds() <= 120
            assert part.dt.iloc[1:].eq(1).all()
            assert not part.stationary_evidence.any()
            assert part.distance.iloc[-1] - part.distance.iloc[0] >= 50
            slope = 100 * np.polyfit(part.distance - part.distance.mean(),
                                     part.altitude_median5, 1)[0]
            assert np.isclose(slope, frame.gradient50.iloc[i])


def feature_summary(sources):
    """Compare numerical behaviour on identical supported timestamps."""
    rows = []
    for feature in FEATURES:
        common = np.logical_and.reduce(
            [frame[feature].notna() for frame in sources.values()])
        for name, frame in sources.items():
            values = frame[feature].where(common)
            step = values.diff().abs().where(frame.dt.eq(1))
            rows.append({
                "source": name, "feature": feature,
                "n": frame[feature].count(), "coverage": 100 * frame[feature].notna().mean(),
                "common": int(common.sum()),
                "p95": values.abs().quantile(.95),
                "step": step.quantile(.95), "max": values.abs().max(),
                "extreme": int(values.abs().gt(20).sum())
                if feature == "gradient50" else None,
            })
    return rows


def stop_summary(sources, stops):
    """Evaluate the saved probable stops on the unchanged Garmin timeline."""
    rows = []
    for stop in stops.itertuples():
        row = {"start": stop.start, "end": stop.end,
               "duration": stop.duration_s}
        for name, frame in sources.items():
            block = frame.loc[stop.start:stop.end, "altitude_raw"]
            if not block.notna().all():
                raise ValueError(
                    "A saved stop lacks complete source elevation")
            row[name] = (block.iloc[-1] - block.iloc[0],
                         block.max() - block.min())
        rows.append(row)
    return rows


def plot_comparison(sources, path):
    """Display raw profiles and source-specific supported terrain candidates."""
    fig, axes = plt.subplots(4, 1, figsize=(15, 14), constrained_layout=True)
    reference = sources["Garmin"]
    for name, frame in sources.items():
        axes[0].plot(frame.distance / 1000, frame.altitude_raw,
                     color=COLORS[name], lw=1.1, label=name + " recorded/exported")
        flags = frame.altitude_flag
        axes[0].scatter(frame.loc[flags, "distance"] / 1000,
                        frame.loc[flags, "altitude_raw"], color=COLORS[name],
                        marker="x", s=26, zorder=4)
    axes[0].plot(reference.distance / 1000, reference.altitude_median5,
                 color=".35", ls=":", lw=.7, label="Garmin diagnostic median (5 samples)")
    axes[0].axvspan(10.30, 10.35, color="#b14d32", alpha=.17,
                    label="20:34 anomaly; crosses = >2m/1s flags")
    steep = reference.loc["2026-10-16 17:53:20+00:00":
                          "2026-10-16 17:55:40+00:00", "distance"] / 1000
    axes[0].axvspan(steep.iloc[0], steep.iloc[-1], color="#167ba0", alpha=.10,
                    label="Inspect steep exported descent (~19:54)")
    axes[0].set(xlabel="Garmin cumulative recorded distance (km)",
                ylabel="Altitude (m)", title="1. Complete profiles: different elevation, same activity timeline")
    start = pd.Timestamp("2026-10-16 18:33:50", tz="UTC")
    end = pd.Timestamp("2026-10-16 18:34:40", tz="UTC")
    for name, frame in sources.items():
        part = frame.loc[start:end]
        axes[1].plot(part.index, part.altitude_raw, color=COLORS[name],
                     marker=".", ms=3, label=name)
    axes[1].axvspan(pd.Timestamp("2026-10-16 18:34:13", tz="UTC"),
                    pd.Timestamp("2026-10-16 18:34:16", tz="UTC"),
                    color=".4", alpha=.15)
    axes[1].set(ylabel="Altitude (m)", xlabel="Local time (Europe/Brussels, UTC+02:00)",
                title="2. 20:34:13–16: 10.14m path travel; Garmin +13.4m, export +0.6m")
    axes[1].xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M:%S", tz=ZONE))
    for ax, feature, unit in zip(axes[2:], FEATURES, ("m/s", "%")):
        for name, frame in sources.items():
            ax.plot(frame.index, frame[feature], color=COLORS[name], lw=.85,
                    label=name)
            # Shaded ribbons use axes coordinates: missing support is not zero.
            lo, hi = (.01, .045) if name == "Garmin" else (.05, .085)
            for indices in runs(frame[feature].isna(), frame.index):
                ax.axvspan(frame.index[indices[0]], frame.index[indices[-1]],
                           ymin=lo, ymax=hi, color=COLORS[name], alpha=.65)
        ax.axhline(0, lw=.5, color=".4")
        ax.set(ylabel=("Vertical rate" if feature == "climb30" else "Path gradient") + " (" + unit + ")",
               xlabel="Local time (Europe/Brussels, UTC+02:00)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZONE))
        ax.set_xlim(reference.index[0], reference.index[-1])
    axes[2].set_title(
        "3. Centered 30-sample climbing rate; bottom ribbons = unavailable support (incl. stops)")
    axes[3].set_title(
        "4. Distance-window gradient (at least 50m); dashed ±20% = inspection screen, not truth")
    for value in (-20, 20):
        axes[3].axhline(value, color=".4", ls="--", lw=.7)
    for ax in axes:
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(alpha=.2)
    fig.suptitle("October 16 MTB: Strava-exported elevation versus original Garmin\n"
                 "Terrain diagnostics only — no independent altitude ground truth", fontsize=14)
    fig.savefig(path, dpi=155)
    plt.close(fig)


def write_report(sources, gpx, xml, messages, stops, path, protected):
    """Write reproducible measured comparisons and qualified recommendations."""
    garmin, strava = sources.values()
    difference = strava.altitude_raw - garmin.altitude_raw
    separation = gps_distance(garmin.position_lat, garmin.position_long,
                              gpx.lat * 2 ** 31 / 180, gpx.lon * 2 ** 31 / 180)
    closure = gps_distance(garmin.position_lat.iloc[0], garmin.position_long.iloc[0],
                           garmin.position_lat.iloc[-1], garmin.position_long.iloc[-1])
    closure_gpx = gps_distance(gpx.lat.iloc[0] * 2 ** 31 / 180,
                               gpx.lon.iloc[0] * 2 ** 31 / 180,
                               gpx.lat.iloc[-1] * 2 ** 31 / 180,
                               gpx.lon.iloc[-1] * 2 ** 31 / 180)
    summary = {name: altitude_summary(frame)
               for name, frame in sources.items()}
    features = feature_summary(sources)
    stop_rows = stop_summary(sources, stops)
    session = messages["session_mesgs"][0]
    stop_better = sum(r["Strava"][1] < r["Garmin"][1] - .05 for r in stop_rows)
    stop_worse = sum(r["Strava"][1] > r["Garmin"][1] + .05 for r in stop_rows)
    lines = [
        "# Strava versus Garmin elevation — October 16, 2026", "",
        "## Decision", "",
        "The GPX contains a materially different elevation profile, consistent with the user's "
        "Strava correction/export workflow. It is not a copy or constant offset of Garmin altitude. "
        "It resolves two strong internal inconsistencies: route closure and the 20:34 jump. "
        "The export has no correction flag, algorithm, terrain-model provenance or independent truth, "
        "so the precise correction mechanism and absolute accuracy cannot be authenticated.", "",
        "Prefer the exported Strava profile with explicit quality masks for a first terrain sensitivity "
        "experiment, retaining filtered Garmin as an alternative. This preference concerns gross drift "
        "and discontinuities, not proven local climbing-rate or gradient accuracy. Neither local "
        "feature is validated; do not build a switching hybrid yet. A limited physiological feasibility "
        "experiment is supportable with terrain ablation and uncertainty flags, not reliable second-by-second effort claims.", "",
        "## Sources, export and alignment", "",
        "- Garmin: `data/mtb_edge.fit`, official FIT SDK integrity-checked decoding; enhanced altitude "
        "in metres, speed in m/s, distance in metres. Original speed/distance/time remain the reference.",
        f"- Export: `data/VTT_Apero.gpx`; GPX {xml.get('version')}, creator `{xml.get('creator')}`, "
        f"{len(xml.findall('g:trk', NS))} track, {gpx.segment.nunique()} segment; elevation `<ele>` in GPX metres, "
        "latitude/longitude in degrees, UTC `<time>`; temperature/HR extensions present but unused.",
        f"- Both: {len(garmin):,} points, {garmin.index[0].isoformat()}–{garmin.index[-1].isoformat()} "
        f"({garmin.index[0].tz_convert(ZONE).strftime('%H:%M:%S')}–"
        f"{garmin.index[-1].tz_convert(ZONE).strftime('%H:%M:%S')} Brussels, UTC+02:00); "
        f"{(garmin.index[-1]-garmin.index[0]).total_seconds()/60:.2f} min, "
        f"{garmin.distance.iloc[-1]/1000:.3f} km recorded distance.",
        f"- Exact UTC timestamp matching: {len(gpx):,}/{len(garmin):,} (100%); all increments 1s. "
        "No missing timestamps, elevations or coordinates, duplicate times, omitted records, "
        "resampling or clipped endpoints observed. There is no timestamp shift, interpolation or extrapolation.",
        f"- Corresponding GPS separation median {np.median(separation):.3f}m, maximum {np.max(separation):.3f}m. "
        "Small export coordinate differences do not prevent exact time alignment; they may still affect terrain lookup.",
        f"- GPS bounds: latitude {gpx.lat.min():.6f}–{gpx.lat.max():.6f}, "
        f"longitude {gpx.lon.min():.6f}–{gpx.lon.max():.6f}. Point counts/time coverage exclude temporal "
        "simplification, but cannot establish the spatial resolution of the elevation source.", "",
        "### What did ‘200m to 400m’ mean?", "",
        "The displayed field and its original/corrected values cannot be established from these files. "
        "GPX `<ele>` values are absolute altitude; it supplies no Strava cumulative-gain summary. "
        "Here the export peaks at 235.2m, while standardized cumulative gain is 472.9m. "
        "Thus a displayed value near 400m is consistent with gain, but is not evidence of a 400m absolute "
        "altitude. Confirming the user's exact numbers requires the labelled activity summary or screenshot; "
        "do not equate them with this calculation. Garmin FIT's ascent summary is 317m, not ~200m.", "",
        "## Altitude comparison", "",
        "Standardized gain/descent = sum of positive/negative adjacent finite altitude increments at "
        "consecutive 1s timestamps, including stops, with no threshold or smoothing. Both use the identical "
        "10,408 intervals. This deliberately exposes noise/drift and is not a physical ground truth or "
        "Strava's proprietary gain calculation. Gaps and segment boundaries would not be summed across.", "",
        "| Source | Start m | Finish m | Min m | Max m | Gain m | Descent m | >2m/1s steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary.items():
        lines.append("| " + name + " | " + " | ".join(f"{v:.1f}" for v in values[:-1])
                     + f" | {values[-1]} |")
    lines += ["", f"Garmin session summary: ascent {session['total_ascent']}m, descent "
              f"{session['total_descent']}m; distinct from increment sums. Strava summary unavailable. "
              "Larger gain alone does not demonstrate improvement.", "",
              f"Matched altitude difference (Strava minus Garmin): mean {difference.mean():.2f}m, "
              f"median {difference.median():.2f}m, range {difference.min():.2f} to {difference.max():.2f}m. "
              f"Only {difference.abs().le(.11).sum()} points agree within 0.11m; "
              f"adjacent altitude-change correlation {strava.altitude_raw.diff().corr(garmin.altitude_raw.diff()):.3f}.", "",
              "## Known problems and new cautions", "",
              "- **20:34:13–16 Brussels:** Garmin 135.2→148.6m (+13.4m), export 151.6→152.2m (+0.6m). "
              "Recorded distance 10316.17→10326.31m (+10.14m): implied rise/path-distance 132.15% versus "
              "5.92%; three-second vertical change 4.47 versus 0.20m/s. The obvious discontinuity disappears, "
              "rather than moving to another >2m/1s step nearby. No pause or recording gap explains it. "
              "These ratios use recorded path distance, not certified horizontal survey distance.",
              f"- **Route closure:** endpoints separated by {closure:.1f}m (Garmin GPS), "
              f"{closure_gpx:.1f}m (export GPS). Garmin finish minus start +105.4m; export −0.8m. "
              "The points are nearby, not identical; a nonzero true elevation difference is possible, "
              "but the Garmin mismatch is strongly inconsistent with route closure.",
              f"- **Saved stops:** reused all {len(stop_rows)} probable intervals from "
              f"`data/processed/cycling_predictor_quality_summary.csv`, totalling {stops.duration_s.sum():.0f}s "
              "(56m21s). No ravito labels inferred. Low speed, ≤3m distance range and ≤8m GPS radius "
              "are moderate-confidence evidence, with dependent recording channels. "
              f"Export altitude range improves at {stop_better} stops, worsens at {stop_worse}, "
              f"and ties within 0.05m at {len(stop_rows)-stop_better-stop_worse}.",
              "- At 22:08:22–22:11:40, Garmin drifts +8.2m with unchanged distance; export range 0.0m. "
              "At 21:20:59–21:27:50, Garmin range 7.2m versus export 0.0m. "
              "Counterexamples: 21:39:16–21:40:56 range 0.6→1.5m and 21:40:58–21:44:04 "
              "range 1.6→2.7m (Garmin→export), despite probable stationary states. "
              "Mask terrain at stops for both sources.", "",
              "### Resolution, steep sections and stability", ""]
    for name, frame in sources.items():
        delta = frame.altitude_raw.diff()
        plateaus = runs(delta.abs().lt(1e-7) &
                        frame.speed_raw.gt(.3), frame.index)
        longest = max(plateaus, key=len)
        part = frame.iloc[longest]
        slope = 100 * delta / frame.distance.diff()
        valid = frame.dt.eq(
            1) & frame.speed_raw.gt(.3) & frame.distance.diff().gt(1)
        lines.append(f"- {name}: largest adjacent step {delta.abs().max():.1f}m; "
                     f"{100*delta.iloc[1:].eq(0).mean():.1f}% unchanged adjacent elevations. "
                     f"Longest moving unchanged-step run: {len(longest)} records starting "
                     f"{part.index[0].tz_convert(ZONE).strftime('%H:%M:%S')}, "
                     f"{part.distance.iloc[-1]-part.distance.iloc[0]:.1f}m between run endpoints. "
                     f"{int((slope.abs().gt(40)&valid).sum())} instantaneous path slopes exceed ±40% "
                     "where Δdistance>1m; these are inspection flags, not confirmed terrain errors.")
    same_position = gpx.lat.diff().eq(0) & gpx.lon.diff().eq(0)
    lines += [f"- Export elevation is quantized to 0.1m; {int((same_position & strava.altitude_raw.diff().abs().gt(.05)).sum())} "
              "adjacent identical exported GPS positions have changing elevation (maximum 0.1m). "
              "Temporal processing/rounding may contribute; this is not proof of a pure position-to-height lookup.",
              "- Around 19:54 the exported 50m gradient reaches roughly −25%, with locally larger "
              "altitude changes than Garmin. This may be real steep descent, GPS/terrain mismatch or "
              "export processing: without surveyed terrain it cannot be classified as an improvement. "
              "Plateaus/stair steps exist in both, but neither native DEM spacing nor forest cover is "
              "encoded. Do not label sections forested or technical from elevation alone.", "",
              "## Terrain features on common support", "",
              "Reuse `audit_cycling_predictors.terrain_features`: centered 5-sample median (diagnostic only), "
              "30-sample altitude/time least-squares trend (29s endpoint span), and altitude/distance "
              "least-squares gradient bracketing ±25m, at least 50m support and at most 120s duration. "
              "Gradient is rise/path-distance, not necessarily rise/horizontal-distance. "
              "Both sources use identical Garmin movement state, speed and distance. "
              "Full finite support is required, with splits at every recording gap, GPX segment boundary "
              "and low-movement transition; no smoothing or regression crosses a stop. "
              "Steps >2m/1s trigger the previous 30s caution margin plus feature support. "
              "These are screening criteria, not proof that unflagged data are valid. "
              "Source-specific flags explain coverage differences; numeric comparisons below use the intersection.", "",
              "| Source | Feature (units) | Available n / all % | Common n | Common P95 abs | Common P95 1s change | Common max abs |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in features:
        unit = "m/s" if row["feature"] == "climb30" else "%"
        lines.append(f"| {row['source']} | {row['feature']} ({unit}) | {row['n']} / "
                     f"{row['coverage']:.2f} | {row['common']} | {row['p95']:.4f} | "
                     f"{row['step']:.4f} | {row['max']:.4f} |")
    lines += ["", "P95 1s changes require both adjacent common timestamps; no change is measured across "
              "a support gap. Export gradients exceed ±20% at "
              f"{features[3]['extreme']} common timestamps versus {features[2]['extreme']} for Garmin. "
              "The threshold is an inspection screen, not a validated maximum. "
              "The exported profile has higher local variability despite removal of gross drift; "
              "smoothness is neither the selection criterion nor an accuracy metric.", "",
              "Sign/context checks below use the common Garmin-sign groups (±0.05m/s screen only), "
              "not independently verified climbs/descents. Values are median climbing rates:", ""]
    common = garmin.climb30.notna() & strava.climb30.notna()
    for label, mask in (("positive trend", garmin.climb30.gt(.05)),
                        ("negative trend", garmin.climb30.lt(-.05))):
        mask &= common
        lines.append(f"- {label}, {mask.sum()} timestamps: Garmin "
                     f"{garmin.loc[mask, 'climb30'].median():.3f}, export "
                     f"{strava.loc[mask, 'climb30'].median():.3f}m/s; "
                     f"{100*(np.sign(garmin.loc[mask,'climb30']) == np.sign(strava.loc[mask,'climb30'])).mean():.1f}% sign agreement.")
    lines += ["", "For sustained examples, select the longest continuous common-support run in each "
              "Garmin sign group (same ±0.05m/s screen), requiring at least 60s. "
              "This rule does not select by source agreement or appearance:", ""]
    for label, mask in (("climbing", garmin.climb30.gt(.05)),
                        ("descending", garmin.climb30.lt(-.05))):
        groups = runs(common & mask, garmin.index)
        group = max(groups, key=len)
        part = garmin.iloc[group]
        duration = (part.index[-1] - part.index[0]).total_seconds()
        assert duration >= 60
        lines.append(f"- {label}, {part.index[0].tz_convert(ZONE).strftime('%H:%M:%S')}–"
                     f"{part.index[-1].tz_convert(ZONE).strftime('%H:%M:%S')} ({duration:.0f}s): "
                     f"median Garmin {part.climb30.median():.3f}, "
                     f"export {strava.loc[part.index, 'climb30'].median():.3f}m/s.")
    lines += ["", "Both features are deliberately unavailable at stops and until complete support after "
              "restarts. The plot's bottom ribbons show unavailable support, not zero climbing. "
              "A five-sample median cannot correct Garmin drift or establish Strava terrain resolution.", "",
              "## Modelling choices and final answers", "",
              "| Strategy | Advantages | Failure modes and coverage |",
              "|---|---|---|",
              "| Garmin with quality filtering | Native synchronized recording; small local numerical changes | Gross drift and residual slope bias remain; flags remove jump regions; stops unavailable |",
              "| Strava with quality filtering | Corrects closure and obvious jump; more supported terrain timestamps | Larger local slopes/variability, quantization and possible map/GPS errors; stops still masked; 1s export does not prove 1s terrain resolution |",
              "| Explicit-quality hybrid | Could reject a source with clear discontinuities | No independent rule chooses correct local slopes; switching offsets can create false climbs. Not implemented or justified |", "",
              "1. **Corrected values present?** A genuinely altered Strava-exported elevation series is present, "
              "consistent with correction; the export does not authenticate its internal algorithm.",
              "2. **Profile differences?** Closure is −0.8m versus +105.4m; exported range 66.3–235.2m "
              "versus Garmin 64.4–304.8m; standardized gain is higher, not evidence of accuracy alone.",
              "3. **20:34 anomaly?** The 13.4m rise becomes 0.6m over the same 10.14m of movement.",
              "4. **Stops?** Usually improved, but two probable stops have greater exported variation; "
              "neither source should generate terrain workload while stationary.",
              "5. **Climbing rate?** Strava is the more defensible provisional source for gross-profile "
              "consistency; local 30-sample accuracy remains unresolved. Retain both in sensitivity tests.",
              "6. **Gradient?** No validated local winner: Strava fixes the obvious jump but produces "
              "more large 50m gradients on common support. Treat it as a provisional input with flags, "
              "and compare Garmin alternatives; neither is a pedaling-power estimate.",
              "7. **Strava less reliable?** Two stationary counterexamples and the larger ~19:54 descent "
              "are warning regions; the latter is unresolved rather than a proven error.",
              "8. **Future source?** Prefer quality-masked exported Strava for the primary terrain "
              "sensitivity arm; retain Garmin and a no-terrain arm. No switching hybrid yet.",
              "9. **Ready for feasibility?** Yes for a limited, explicitly uncertain physiological "
              "feasibility experiment with terrain ablation; no for claiming reliable second-by-second "
              "workload or HR. No measured power/cadence, independent altitude reference, verified "
              "stop ground truth or fine terrain resolution is available. Coasting, braking and "
              "technical effort weaken speed/gradient-to-workload inference. No HR model was built.", "",
              "## Reproduction and verification", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe scripts/compare_strava_garmin_elevation.py", "```", "",
              f"Figure: `docs/images/{STEM}.png`. Script: `scripts/compare_strava_garmin_elevation.py`. "
              "No dense intermediate files or new dependencies.", "",
              "Executed: FIT integrity/decoding; GPX field completeness and degree/metre scale checks; "
              "unique ordered UTC times; exact 1s alignment and GPS consistency; existing stop-boundary "
              "agreement; stationary feature masking; an injected missing-value/segment-boundary support probe; independent "
              "polynomial refits of sampled real 50m gradient windows; minimum "
              "50m distance and maximum 120s support checks; common-timestamp metrics; gain-minus-descent "
              f"equals endpoint change; all {len(protected)} protected source/prior-output SHA-256 hashes unchanged; "
              "non-empty report/figure. Visual readability is checked separately after generation. "
              "No external APIs, source edits, HR estimates, model fitting, commits or uploads.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run only the target MTB comparison and verify protected files."""
    prior = list((ROOT / "data").rglob("*"))
    prior += list((ROOT / "docs").rglob("*"))
    prior += list((ROOT / "scripts").glob("*.py"))
    prior += [ROOT / "README.md", ROOT / "AGENTS.md"]
    protected = [p for p in prior if p.is_file() and STEM not in p.name
                 and p.name != "compare_strava_garmin_elevation.py"]
    before = hashes(protected)
    timeline, messages = load_records("mtb_edge")
    detected = pd.DataFrame(identify_stops(timeline))
    stops = pd.read_csv(
        ROOT / "data/processed/cycling_predictor_quality_summary.csv")
    stops = stops.loc[stops.record_type.eq("mtb_stop")
                      & stops.classification.eq("probable_stop")].copy()
    for column in ("start", "end"):
        stops[column] = pd.to_datetime(stops[column], utc=True)
        actual = detected.loc[detected.classification.eq(
            "probable_stop"), column]
        assert list(stops[column]) == list(actual)
    gpx, xml = load_gpx(ROOT / "data/VTT_Apero.gpx")
    if not gpx.index.equals(timeline.index):
        raise ValueError(
            "Exact full time alignment unavailable; inspect before comparing")
    assert gpx[["altitude_raw", "lat", "lon"]].notna().all().all()
    assert timeline.dt.iloc[1:].eq(1).all()
    assert gpx.lat.between(-90, 90).all() and gpx.lon.between(-180, 180).all()
    delta = gpx.altitude_raw - timeline.altitude_raw
    if delta.std() < .11:
        raise ValueError(
            "No materially different profile: obtain an export explicitly containing corrected elevation before comparison")
    sources = {
        "Garmin": source_features(timeline, timeline.altitude_raw,
                                  pd.Series(0, index=timeline.index)),
        "Strava": source_features(timeline, gpx.altitude_raw, gpx.segment),
    }
    verify_support(timeline, sources)
    for frame in sources.values():
        values = altitude_summary(frame)
        assert np.isclose(values[4] - values[5], values[1] - values[0])
    image = ROOT / "docs/images" / (STEM + ".png")
    report = ROOT / "docs" / (STEM + ".md")
    plot_comparison(sources, image)
    assert hashes(
        protected) == before, "Protected source or prior analysis changed"
    write_report(sources, gpx, xml, messages, stops, report, protected)
    assert hashes(protected) == before
    assert image.stat().st_size > 0 and report.stat().st_size > 0
    print(pd.DataFrame(feature_summary(sources)).to_string(index=False))
    print("Wrote", report.relative_to(ROOT), "and", image.relative_to(ROOT))
    print("Protected hashes unchanged:", len(protected))


if __name__ == "__main__":
    main()
