"""Evaluate existing interpolation on deterministic sparse samples of 3 rides.

Run: .venv/Scripts/python scripts/evaluate_hr_interpolation.py
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

from diagnose_watch_anchors import load_activity, reference_values
from reconstruct_hr_baseline import METHODS, interpolate_watch, validate


ROOT = Path(__file__).resolve().parents[1]
STEM = "hr_reconstruction_validation_3rides"
RIDES = ("road_ride_1", "road_ride_2", "virtual_road_ride_1")
PHASES = (0, 20, 40)
ZONE = pytz.timezone("Europe/Brussels")


def sampling_times(start, end, pattern, phase):
    """Replay the actual interval sequence cyclically in wall-clock time."""
    seconds = [phase]
    i = 0
    duration = (end - start).total_seconds()
    while seconds[-1] + pattern[i % len(pattern)] <= duration:
        seconds.append(seconds[-1] + pattern[i % len(pattern)])
        i += 1
    return start + pd.to_timedelta(seconds, unit="s")


def continuous_blocks(reference):
    """Split at every absent second, including recording pauses."""
    group = reference.index.to_series().diff().dt.total_seconds().ne(1).cumsum()
    return [part for _, part in reference.groupby(group.to_numpy())]


def reconstruct(anchors, grid, block_bounds):
    """Call the existing interpolator with sparse anchors only, per block."""
    result = pd.DataFrame(index=grid, columns=METHODS, dtype=float)
    for a, b in block_bounds:
        sparse = anchors.loc[a:b]
        if sparse.empty:
            continue
        targets = grid[(grid >= a) & (grid <= b)]
        if len(sparse) == 1:
            estimates = pd.DataFrame(index=targets, columns=METHODS,
                                     dtype=float)
            estimates.loc[sparse.index, :] = sparse.iloc[0]
        else:
            estimates = interpolate_watch(sparse.copy(), targets)
        validate(estimates, sparse)
        result.loc[targets, list(METHODS)] = estimates.values
    for method in METHODS:
        assert np.array_equal(result.loc[anchors.index, method], anchors.values)
    return result


def evaluate(reference, estimates, common, method):
    """Score matching observed timestamps, including fixed diagnostic subsets."""
    actual = reference.reindex(estimates.index)[common]
    predicted = estimates.loc[common, method]
    assert actual.notna().all() and predicted.notna().all()
    error = predicted - actual
    high = actual >= reference.quantile(.9)
    # Changes require adjacent recorded seconds, not differences over pauses.
    steps = reference.diff().where(
        reference.index.to_series().diff().dt.total_seconds().eq(1))
    rapid = steps.reindex(actual.index).abs().ge(2)
    peak_time = actual.idxmax()
    missed_time = error.idxmin()
    return {
        "evaluated_n": len(actual),
        "evaluated_percent": 100 * len(actual) / len(reference),
        "mae": error.abs().mean(),
        "rmse": np.sqrt(np.mean(error ** 2)),
        "bias": error.mean(),
        "average_hr_error": predicted.mean() - actual.mean(),
        "reference_mean": actual.mean(), "estimated_mean": predicted.mean(),
        "high_n": int(high.sum()), "high_bias": error[high].mean(),
        "high_mae": error[high].abs().mean(),
        "rapid_n": int(rapid.sum()), "rapid_mae": error[rapid].abs().mean(),
        "peak_time_utc": peak_time, "peak_reference": actual.loc[peak_time],
        "peak_estimate": predicted.loc[peak_time],
        "missed_time_utc": missed_time,
        "missed_reference": actual.loc[missed_time],
        "missed_estimate": predicted.loc[missed_time],
        "max_abs_error": error.abs().max(),
    }


def draw_figure(panels, path):
    """Show the midpoint 20 minutes of each longest uninterrupted HR block."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)
    for ax, panel in zip(axes, panels):
        name, reference, anchors, estimates, blocks = panel
        longest = max(blocks, key=len)
        if (longest.index[-1] - longest.index[0]).total_seconds() < 1200:
            raise ValueError(f"{name}: no continuous 20-minute plot interval")
        midpoint = longest.index[0] + (
            longest.index[-1] - longest.index[0]) / 2
        start = (midpoint - pd.Timedelta(minutes=10)).floor("s")
        end = start + pd.Timedelta(minutes=20)
        visible = reference.loc[start:end]
        ax.plot(visible.index, visible, color=".55", lw=.9,
                label="Recorded HR reference")
        for method, color, style in (("linear", "#e17416", "-"),
                                      ("pchip", "#7844a6", "--")):
            ax.plot(estimates.loc[start:end].index,
                    estimates.loc[start:end, method], color=color, ls=style,
                    lw=1.8, label=f"{method.upper()} estimate")
        sparse = anchors.loc[start:end]
        ax.scatter(sparse.index, sparse, s=30, color="#007a91", zorder=5,
                   label="Simulated point anchors")
        ax.set_xlim(start, end)
        date = start.tz_convert(ZONE).strftime("%Y-%m-%d")
        ax.set_title(f"{name} | {date} | fixed phase 0s")
        ax.set_ylabel("HR (BPM)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZONE))
        ax.grid(alpha=.2)
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.set_xlabel("Local time — Europe/Brussels")
    fig.suptitle("Sparse HR interpolation: three complete activity exports\n"
                 "Point sampling; midpoint of longest continuous block; "
                 "curves are offline estimates", fontsize=14)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(characteristics, metrics, schedules, path):
    """Summarize per-ride results and all phase variability without pooling."""
    lines = ["# HR interpolation evaluation — three cycling rides", "",
             "## Datasets", "",
             "UTC is used internally; displayed start/end times are "
             "Europe/Brussels. Durations below span the first/last HR record. "
             "A one-second record is not necessarily a fresh independent "
             "physiological observation.", "",
             "| Ride | Date, local HR start–end | Span | Valid HR | Coverage of elapsed seconds | Recording gaps / longest | Repeated adjacent BPM |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for c in characteristics:
        lines.append(
            f"| {c['ride']} | {c['start']}–{c['end']} | "
            f"{c['span_minutes']:.2f} min | {c['n']} | "
            f"{c['coverage']:.2f}% | {c['gaps']} / "
            f"{c['max_gap']:.0f}s | {c['repeat_percent']:.2f}% |")
    lines += ["", "Outdoor activities identify an Edge 840 recorder and an "
              "ANT+ heart-rate device labeled `OHR`; the metadata does not "
              "establish its exact optical sensor model or physiological "
              "accuracy. Their timer pauses and recording gaps prevent "
              "continuous reference evaluation over the whole elapsed ride. "
              "HR is stored at 1s intervals except those discontinuities: "
              "17,714 adjacent 1s intervals in road ride 1, 15,432 in road "
              "ride 2 and 5,114 in the virtual ride. "
              "All recorded HR values pass 0<BPM<255 validity checks; no "
              "physiological outliers are removed. Each raw inter-record gap "
              "is treated as an unsupported boundary.", "",
              "The virtual activity is the confirmed FR265 optical sensor → "
              "Bluetooth → MyWhoosh path. Its developer metadata identifies "
              "MyWhoosh despite an Edge product ID in the FIT header. It has "
              "no recording gaps. Held values, latency and optical errors "
              "remain possible in all references. The road and virtual "
              "references therefore have different recording paths.", "",
              "## Simulation and comparison", "",
              "Replay the exact October 16 watch interval sequence from "
              "`data/processed/watch_hr_ride_2026-10-16.csv`: 158 anchors, "
              "142×60s and 15×120s intervals (172-minute span). Start at each "
              "activity's first recorded HR time +0/+20/+40 seconds; cycle "
              "the sequence for longer rides and truncate for shorter ones. "
              "This deterministic schedule uses elapsed UTC time through "
              "pauses, never row indices. Longer/shorter rides accordingly "
              "have different anchor counts; prefix proportions need not "
              "equal the full-pattern proportions exactly.", "",
              "A: point anchors equal the exact recorded HR at scheduled "
              "timestamps. B: averaged anchors are means of all 60 observed "
              "integer-second values [t−30s, t+29s]. If any required record "
              "is absent/invalid, drop that anchor; never fill reference HR. "
              "Averaging is an alternative simulation assumption, not an "
              "established Garmin storage rule. Its 60 source samples are "
              "intentionally used to define an anchor, then only that scalar "
              "anchor is supplied to reconstruction.", "",
              "Reuse `interpolate_watch` and its verification from the "
              "October 16 baseline: linear/PCHIP, maximum 180s anchor gap, "
              "no extrapolation. Additionally break at every missing reference "
              "second to avoid interpolating through known recording gaps. "
              "This structural use of gap locations does not expose hidden "
              "HR values. No filtering, calibration or time shifts are used.", "",
              "Within each ride/phase, all four scenario/method combinations "
              "use the identical intersection of supported timestamps with "
              "valid reference HR. Counts/percentages use all valid HR rows "
              "in that ride as denominator. All eligible errors are retained, "
              "including anchor timestamps. Point-sampled anchor accuracy is "
              "therefore partly guaranteed by design. Estimates are offline "
              "and use anchors on both sides. No independent holdout is "
              "needed to fit these fixed interpolation rules; the reference "
              "is used only to construct declared simulated anchors and score.",
              "", "| Ride | Phase | Scheduled anchors (60s/120s intervals) | Point / averaged valid anchors | Evaluated n (%) |",
              "|---|---:|---:|---:|---:|"]
    for row in schedules:
        lines.append(f"| {row['ride']} | {row['phase']}s | "
                     f"{row['scheduled']} ({row['n60']}/{row['n120']}) | "
                     f"{row['point']} / {row['averaged']} | "
                     f"{row['evaluated_n']} ({row['evaluated_percent']:.2f}%) |")
    lines += ["", "## Accuracy and phase variability", "",
              "Values are the equal-weight mean across all three phases "
              "[minimum, maximum]; phases are not independent rides. Bias is "
              "estimate minus reference. The average-HR error equals bias "
              "on these same evaluation timestamps, rather than comparing "
              "a supported estimate with a full-ride reference mean.", "",
              "| Ride | Anchors | Method | MAE (BPM) | RMSE (BPM) | Mean-HR error / bias (BPM) |",
              "|---|---|---|---:|---:|---:|"]
    for (ride, scenario, method), rows in metrics.groupby(
            ["ride", "scenario", "method"], sort=False):
        def value(column):
            s = rows[column]
            return f"{s.mean():.2f} [{s.min():.2f}, {s.max():.2f}]"
        lines.append(f"| {ride} | {scenario} | {method} | "
                     f"{value('mae')} | {value('rmse')} | "
                     f"{value('average_hr_error')} |")
    paired = metrics.pivot_table(index=["ride", "scenario", "phase"],
                                 columns="method", values="mae")
    pchip_wins = int((paired.pchip < paired.linear).sum())
    lines += ["", f"PCHIP has lower MAE in {pchip_wins}/{len(paired)} "
              "ride/scenario/phase combinations; linear wins "
              f"{int((paired.linear < paired.pchip).sum())}/{len(paired)}. "
              "This descriptive count does not establish superiority or "
              "statistical significance. Full phase-level metrics, paired "
              "means and evaluation counts are in the compact CSV. Across "
              "phases, point-sampled PCHIP has slightly lower mean MAE but "
              "higher mean RMSE than linear on each ride. With averaged "
              "anchors, PCHIP has lower mean MAE and RMSE on each ride. "
              "Averaging reduces RMSE for both methods on all three rides; "
              "linear MAE changes little, while PCHIP MAE improves.", "",
              "## High HR and rapid changes", "",
              "High HR is the ride's upper decile of recorded BPM (ties "
              "included). Rapid change is an absolute ≥2 BPM difference "
              "between consecutive observed seconds, never across a gap. "
              "These diagnostic subsets do not replace the primary scores.", "",
              "| Ride | Anchors | Method | High-HR bias, mean over phases | Rapid-change MAE, mean over phases |",
              "|---|---|---|---:|---:|"]
    for key, rows in metrics.groupby(["ride", "scenario", "method"], sort=False):
        lines.append(f"| {' | '.join(key)} | {rows.high_bias.mean():.2f} | "
                     f"{rows.rapid_mae.mean():.2f} |")
    lines += ["", "Examples at phase 0, point scenario: each method's largest "
              "underestimate within the scored region. These deliberately "
              "show worst cases rather than typical errors; all timestamps "
              "remain in the primary metrics.", "",
              "| Ride | Method | Local timestamp | Recorded | Estimate |",
              "|---|---|---|---:|---:|"]
    for ride in RIDES:
        rows = metrics[(metrics.ride == ride) & (metrics.phase == 0)
                       & (metrics.scenario == "point")].set_index("method")
        for method in METHODS:
            row = rows.loc[method]
            t = pd.Timestamp(row.missed_time_utc).tz_convert(ZONE).isoformat()
            lines.append(f"| {ride} | {method} | {t} | "
                         f"{row.missed_reference:g} | "
                         f"{row.missed_estimate:.2f} |")
    lines += ["", "## Implications and reproducibility", "",
              "Compare the per-ride table before drawing outdoor/virtual "
              "conclusions: steadier virtual HR and its recording path can "
              "make interpolation easier, while outdoor changes, pauses and "
              "optical artifacts can increase error. Averaging anchors can "
              "change both peak errors and mean HR without restoring missing "
              "rapid fluctuations. Low average-HR bias does not imply low "
              "instantaneous error. Point anchors assume perfect agreement "
              "with the reference at sampled times, unlike the actual "
              "October 16 watch/Edge recording paths.", "",
              "These three rides evaluate the information lost by sparse "
              "sampling under two assumptions. They do not identify Garmin's "
              "wellness algorithm, calibrate the MTB recording paths, or "
              "prove generalization to MTB. Keep the original October 16 "
              "12.78/17.18 BPM linear MAE/RMSE as separate evidence. Temporal "
              "correlation and only three rides preclude strong population "
              "claims. Synthetic anchors are simulated observations; curves "
              "remain estimates. No ML or new reconstruction method is used.", "",
              "The figure uses point anchors at phase 0 and the central "
              "20 minutes of each ride's longest uninterrupted HR block. "
              "This rule uses timestamp availability, not errors or HR shape.",
              "",
              "```powershell",
              ".\\.venv\\Scripts\\python.exe scripts/evaluate_hr_interpolation.py",
              "```", "",
              f"- Figure: `docs/images/{STEM}.png`.",
              f"- Report: `docs/{STEM}.md`.",
              f"- All 36 metrics rows: `data/processed/{STEM}_metrics.csv`.",
              "",
              "The real-file run verifies intended anchor definitions, "
              "fully observed windows, exact anchors, no extrapolation, "
              "gap/value bounds and matching evaluation timestamps. FIT "
              "and existing October 16 output hashes are checked unchanged. "
              "No dependencies, test suite, raw-data copies, commits or "
              "uploads are created.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run all 36 fixed experiments and create only the requested artifacts."""
    protected = [p for p in ROOT.rglob("*") if p.is_file()
                 and ".venv" not in p.parts and ".git" not in p.parts
                 and (p.suffix.lower() == ".fit" or "2026-10-16" in p.name)]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    watch = pd.read_csv(ROOT / "data/processed/watch_hr_ride_2026-10-16.csv")
    timestamps = pd.to_datetime(watch.timestamp, utc=True).sort_values()
    pattern = np.diff(pd.DatetimeIndex(timestamps).asi8) // 1_000_000_000
    assert len(timestamps) == 158
    assert (pattern == 60).sum() == 142 and (pattern == 120).sum() == 15
    metrics_rows, characteristics, schedules, panels = [], [], [], []
    for name in RIDES:
        reference, records, messages, cleaning = load_activity(
            ROOT / "data" / f"{name}.fit")
        start, end = reference.index[[0, -1]]
        grid = pd.date_range(start, end, freq="s")
        blocks = continuous_blocks(reference)
        block_bounds = [tuple(block.index[[0, -1]]) for block in blocks]
        spacing = reference.index.to_series().diff().dt.total_seconds()
        characteristics.append({
            "ride": name, "start": start.tz_convert(ZONE).isoformat(),
            "end": end.tz_convert(ZONE).strftime("%H:%M:%S"),
            "n": len(reference), "span_minutes": (end - start).total_seconds() / 60,
            "coverage": 100 * len(reference) / len(grid),
            "gaps": int(spacing.gt(1).sum()),
            "max_gap": spacing[spacing.gt(1)].max() if spacing.gt(1).any() else 0,
            "repeat_percent": 100 * reference.diff()[spacing.eq(1)].eq(0).mean(),
        })
        for phase in PHASES:
            times = sampling_times(start, end, pattern, phase)
            anchors = {
                "point": reference.reindex(times).dropna(),
                "averaged": reference_values(
                    reference, times, ("centered", 60, 0)).dropna(),
            }
            assert anchors["point"].equals(
                reference.reindex(anchors["point"].index))
            for t, bpm in anchors["averaged"].items():
                window = reference.reindex(pd.date_range(
                    t - pd.Timedelta(seconds=30),
                    t + pd.Timedelta(seconds=29), freq="s"))
                assert len(window) == 60 and window.notna().all()
                assert np.isclose(bpm, window.mean())
            # This is the sole interpolation call path; no dense HR values
            # are passed as anchor values or used to tune either method.
            results = {scenario: reconstruct(sparse, grid, block_bounds)
                       for scenario, sparse in anchors.items()}
            common = reference.reindex(grid).notna()
            for estimates in results.values():
                common &= estimates.notna().all(axis=1)
            assert common.any()
            for scenario, estimates in results.items():
                for method in METHODS:
                    score = evaluate(reference, estimates, common, method)
                    assert score["evaluated_n"] == int(common.sum())
                    assert np.isclose(score["bias"], score["average_hr_error"])
                    metrics_rows.append({
                        "ride": name, "scenario": scenario, "phase": phase,
                        "method": method, "anchors": len(anchors[scenario]),
                        "reference_n": len(reference), **score,
                    })
            schedules.append({
                "ride": name, "phase": phase, "scheduled": len(times),
                "n60": int((np.diff(times.asi8) / 1e9 == 60).sum()),
                "n120": int((np.diff(times.asi8) / 1e9 == 120).sum()),
                "point": len(anchors["point"]),
                "averaged": len(anchors["averaged"]),
                "evaluated_n": int(common.sum()),
                "evaluated_percent": 100 * common.sum() / len(reference),
            })
            if phase == 0:
                panels.append((name, reference, anchors["point"],
                               results["point"], blocks))
    metrics = pd.DataFrame(metrics_rows)
    assert len(metrics) == 36 and set(metrics.ride) == set(RIDES)
    csv = ROOT / "data/processed" / f"{STEM}_metrics.csv"
    metrics.to_csv(csv, index=False)
    figure = ROOT / "docs/images" / f"{STEM}.png"
    report = ROOT / "docs" / f"{STEM}.md"
    draw_figure(panels, figure)
    write_report(characteristics, metrics, schedules, report)
    assert all(p.stat().st_size > 0 for p in (csv, figure, report))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h
               for p, h in hashes.items())
    print(pd.DataFrame(characteristics).to_string(index=False))
    print(metrics.groupby(["ride", "scenario", "method"])[
        ["mae", "rmse", "average_hr_error"]].mean().to_string())
    print("Verified all 36 cases: anchor definitions/preservation, gaps, "
          "bounds, no extrapolation, matching pairs, nonempty outputs, "
          "unchanged FITs and October 16 outputs.")


if __name__ == "__main__":
    main()
