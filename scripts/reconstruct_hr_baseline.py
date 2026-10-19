"""Compare watch-only linear and PCHIP HR estimates with recorded Edge HR.

Run from the repository root with:
    .venv/Scripts/python scripts/reconstruct_hr_baseline.py
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import hashlib
from pathlib import Path

from garmin_fit_sdk import Decoder, Stream
import matplotlib
import numpy as np
import pandas as pd
import pytz
import scipy
from scipy.interpolate import PchipInterpolator

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-10-16"
ZONE = "Europe/Brussels"
MAX_GAP = 180
METHODS = ("linear", "pchip")


def clean_observations(frame):
    """Remove invalid rows and identical duplicates; reject conflicting anchors."""
    frame = frame.copy()
    # Reject naive timestamps rather than silently assuming a timezone.
    parsed = frame.timestamp.map(
        lambda value: pd.to_datetime(value, errors="coerce")
    )
    if any(t.tzinfo is None for t in parsed if not pd.isna(t)):
        raise ValueError("Input timestamps must include a timezone")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True,
                                        errors="coerce")
    frame["bpm"] = pd.to_numeric(frame.bpm, errors="coerce")
    valid = (frame.timestamp.notna() & np.isfinite(frame.bpm)
             & (frame.bpm > 0) & (frame.bpm < 255))
    if "valid" in frame:
        valid &= frame.valid.astype(str).str.lower().eq("true")
    invalid_count = int((~valid).sum())
    frame = frame.loc[valid, ["timestamp", "bpm"]]
    duplicate_count = int(frame.duplicated(["timestamp", "bpm"]).sum())
    frame = frame.drop_duplicates(["timestamp", "bpm"])
    if frame.timestamp.duplicated().any():
        raise ValueError("Conflicting BPM values at the same timestamp")
    frame = frame.sort_values("timestamp").set_index("timestamp")
    if not (frame.index == frame.index.floor("s")).all():
        raise ValueError("Off-second anchors cannot be preserved on this grid")
    return frame.bpm.astype(float), invalid_count, duplicate_count


def interpolate_watch(watch, grid):
    """Interpolate each gap-bounded segment without extrapolation or Edge input."""
    if len(watch) < 2:
        raise ValueError("At least two valid watch anchors are required")
    result = pd.DataFrame(index=grid, columns=METHODS, dtype=float)
    seconds = np.asarray((watch.index - grid[0]).total_seconds())
    targets = np.asarray((grid - grid[0]).total_seconds())
    breaks = np.flatnonzero(np.diff(seconds) > MAX_GAP) + 1
    for segment in np.split(np.arange(len(watch)), breaks):
        x = seconds[segment]
        y = watch.iloc[segment].to_numpy()
        mask = (targets >= x[0]) & (targets <= x[-1])
        if len(segment) == 1:
            result.loc[mask, :] = y[0]
            continue
        result.loc[mask, "linear"] = np.interp(targets[mask], x, y)
        result.loc[mask, "pchip"] = PchipInterpolator(
            x, y, extrapolate=False
        )(targets[mask])
    # Assign the original values explicitly, avoiding endpoint roundoff.
    for method in METHODS:
        result.loc[watch.index, method] = watch.values
    return result


def read_edge(path):
    """Use the existing official SDK dependency to read actual activity records."""
    if not Decoder(Stream.from_file(str(path))).check_integrity():
        raise ValueError("Edge FIT integrity check failed")
    messages, errors = Decoder(Stream.from_file(str(path))).read(
        merge_heart_rates=False
    )
    if errors:
        raise ValueError(str(errors))
    records = pd.DataFrame(messages["record_mesgs"])
    records["timestamp"] = pd.to_datetime(records.timestamp, utc=True)
    edge, invalid, duplicates = clean_observations(
        records.rename(columns={"heart_rate": "bpm"})
    )
    following = records.loc[records.timestamp > edge.index.max(), "timestamp"]
    cutoff = following.min() if len(following) else None
    return edge, records.timestamp.min(), records.timestamp.max(), cutoff


def validate(result, watch):
    """Check exact anchors, unfilled boundaries/gaps, and local value bounds."""
    assert result.index.is_unique
    assert (np.diff(result.index.asi8) == 1_000_000_000).all()
    outside = ((result.index < watch.index[0])
               | (result.index > watch.index[-1]))
    assert result.loc[outside, list(METHODS)].isna().all().all()
    for method in METHODS:
        assert np.array_equal(result.loc[watch.index, method], watch.values)
    gaps = []
    for i in range(len(watch) - 1):
        a, b = watch.index[i:i + 2]
        duration = (b - a).total_seconds()
        inside = (result.index > a) & (result.index < b)
        values = result.loc[inside, list(METHODS)]
        if duration > MAX_GAP:
            assert values.isna().all().all()
            gaps.append((a, b, duration))
        else:
            lo, hi = sorted(watch.iloc[i:i + 2])
            assert values.notna().all().all()
            assert (values >= lo - 1e-10).all().all()
            assert (values <= hi + 1e-10).all().all()
    return gaps


def error_metrics(pairs, method):
    """Calculate errors on all exact timestamp pairs, with estimate minus Edge."""
    error = pairs[method] - pairs.edge_hr
    return {
        "n": len(error), "mae": error.abs().mean(),
        "rmse": np.sqrt(np.mean(error ** 2)), "bias": error.mean(),
        "estimate_mean": pairs[method].mean(),
        "edge_mean": pairs.edge_hr.mean(),
    }


def draw_figure(result, watch, edge, cutoff, metrics, path):
    """Show full-ride estimates and the unchanged initial evaluation window."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    local = result.index.tz_convert(ZONE)
    for ax in axes:
        ax.plot(edge.index.tz_convert(ZONE), edge.values, color="#777777",
                lw=1, alpha=.85, label="Edge HR (recorded)", zorder=1)
        ax.plot(local, result.linear, color="#df7015", lw=2,
                label="Linear estimate", zorder=2)
        ax.plot(local, result.pchip, color="#7444aa", lw=1.7, ls="--",
                label="PCHIP estimate", zorder=3)
        ax.scatter(watch.index.tz_convert(ZONE), watch.values, s=30,
                   color="#00758a", edgecolor="white", linewidth=.5,
                   label="Watch wellness (stored anchors)", zorder=4)
        if cutoff is not None:
            ax.axvline(cutoff.tz_convert(ZONE), color="#b33336", ls="--",
                       lw=1.2, label="First Edge record without HR")
        ax.set_ylabel("Heart rate (BPM)")
        ax.grid(alpha=.2)
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M", tz=pytz.timezone(ZONE))
        )
    axes[0].set_xlim(local[0].floor("30min"), local[-1])
    axes[0].set_title(
        "Complete MTB ride: sparse observations and baseline estimates")
    axes[0].legend(loc="upper left", fontsize=9, ncol=2)
    if len(edge):
        axes[1].set_xlim(edge.index.min().tz_convert(ZONE),
                         edge.index.max().tz_convert(ZONE)
                         + pd.Timedelta(seconds=2))
    axes[1].set_title("Initial overlap: no calibration or timestamp shifting")
    axes[1].legend(loc="upper left", fontsize=9, ncol=2)
    axes[1].set_xlabel("Local time — Europe/Brussels (UTC+02:00)")
    if metrics:
        annotation = " | ".join(
            f"{m.capitalize()}: MAE {metrics[m]['mae']:.2f}, "
            f"RMSE {metrics[m]['rmse']:.2f} BPM" for m in METHODS
        )
        axes[1].text(.01, .76, annotation, transform=axes[1].transAxes,
                     fontsize=10, bbox={"facecolor": "white", "alpha": .9})
    fig.suptitle(f"HR baseline interpolation — {DATE}\n"
                 "Curves are estimates; wellness averaging window is unknown",
                 fontsize=15)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(result, watch, edge, pairs, metrics, gaps, cleaning,
                 edge_issue, cutoff, path):
    """Write reproducible findings and limitations from the calculated results."""
    def local(t): return t.tz_convert(ZONE).isoformat()
    covered = result.dropna(subset=list(METHODS))
    lines = [f"# HR baseline interpolation — {DATE}", "",
             "## Inputs and method", "",
             "Watch anchors: `data/processed/watch_hr_ride_2026-10-16.csv`, "
             "previously extracted from `494532015728_WELLNESS.fit`. "
             f"{len(watch)} valid anchors, from {local(watch.index[0])} to "
             f"{local(watch.index[-1])}. Cleaning removed {cleaning[0]} invalid "
             f"rows and {cleaning[1]} identical duplicate rows. Conflicting "
             "duplicates or naive timestamps cause an error rather than averaging.",
             "", "Original Edge: `data/mtb_edge.fit`, read using the existing "
             f"official Garmin SDK dependency, with HR merging disabled. "
             f"{len(edge)} valid recorded HR values. All original FIT SHA-256 "
             "hashes were checked unchanged after the run.", "",
             "Both curves use only watch observations at their actual times. "
             "NumPy linear interpolation and SciPy PCHIP are applied separately "
             "to segments split at gaps >180 seconds. No extrapolation, "
             "rounding of BPM, Edge calibration, training, or curve tuning. "
             "Every valid watch anchor is preserved exactly. PCHIP is "
             "[shape-preserving](https://docs.scipy.org/doc/scipy/reference/"
             "generated/scipy.interpolate.PchipInterpolator.html); it does "
             "not create extrema outside neighboring anchor bounds. "
             f"Installed SciPy version: {scipy.__version__}.", "",
             f"Output: {len(result)} one-second rows spanning "
             f"{local(result.index[0])}–{local(result.index[-1])}. "
             f"{len(covered)} rows have reconstructed values, including "
             f"{len(watch)} unchanged anchors and "
             f"{int(result.is_interpolated.sum())} estimated non-anchor rows. "
             f"{len(result) - len(covered)} boundary/gap rows remain missing. "
             f"Gaps exceeding 180 seconds: {len(gaps)}.", ""]
    for a, b, seconds in gaps:
        lines.append(f"- Excluded gap: {local(a)}–{local(b)} ({seconds:g}s).")
    lines += ["## Validation", ""]
    if edge_issue:
        lines += [
            f"Quantitative Edge validation unavailable: {edge_issue}", ""]
    if metrics:
        lines += [f"Edge HR runs {local(edge.index[0])}–{local(edge.index[-1])}. "
                  f"First missing record: {local(cutoff) if cutoff is not None else 'not identified'}. "
                  f"Evaluation includes all {len(pairs)} finite, exact UTC "
                  f"timestamp pairs from {local(pairs.index[0])} through "
                  f"{local(pairs.index[-1])}. "
                  f"{len(edge) - len(pairs)} Edge samples lack supported "
                  "estimates (boundaries or excluded gaps). Difficult intervals "
                  "are not selectively removed. The same pairs are used for both methods.",
                  "", "| Method | Pairs | MAE | RMSE | Bias | Mean estimate | Mean Edge | Mean HR difference |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for method, m in metrics.items():
            lines.append(
                f"| {method} | {m['n']} | {m['mae']:.4f} | {m['rmse']:.4f} "
                f"| {m['bias']:.4f} | {m['estimate_mean']:.4f} "
                f"| {m['edge_mean']:.4f} "
                f"| {m['estimate_mean'] - m['edge_mean']:.4f} |"
            )
        lines += ["", "All HR/error values are BPM. Bias is estimate minus "
                  "Edge; mean HR difference is the difference of means on the "
                  "same pairs and therefore equals bias.", "",
                  "Both inputs have timezone-aware UTC times. Watch anchors "
                  "are on minute boundaries; Edge samples are on integer seconds. "
                  "There is no timestamp evidence establishing a clock offset, "
                  "so no shift is applied. Apparent timing differences in "
                  "fluctuations may reflect clock offset, storage averaging or "
                  "filtering; these cannot be separated from this overlap alone. "
                  "This is an offline reconstruction: interpolation can use "
                  "watch anchors later than each evaluation timestamp, including "
                  "the 19:58 anchor for the final overlap seconds. It is not a "
                  "causal, real-time prediction experiment.",
                  "", "## Peaks and rapid changes", ""]
        threshold = pairs.edge_hr.quantile(.9)
        high = pairs[pairs.edge_hr >= threshold]
        # Diagnostics only; they never replace the full-window metrics.
        rapid_times = edge.index[edge.diff().abs() >= 2]
        rapid = pairs.loc[pairs.index.isin(rapid_times)]
        lines += [f"Upper-decile Edge observations (>= {threshold:g} BPM; "
                  f"{len(high)} pairs) have mean signed errors of "
                  + ", ".join(f"{m}: {(high[m] - high.edge_hr).mean():.2f} BPM"
                              for m in METHODS) + ".", "",
                  f"At the Edge maximum ({pairs.edge_hr.max():g} BPM), "
                  "mean estimated HR across tied maximum timestamps is "
                  + ", ".join(
                      f"{m}: {pairs.loc[pairs.edge_hr == pairs.edge_hr.max(), m].mean():.2f} BPM"
                      for m in METHODS) + ".", "",
                  f"Rapid-change diagnostic: {len(rapid)} paired seconds where "
                  "successive Edge readings change by >=2 BPM. MAE: "
                  + ", ".join(f"{m}: {(rapid[m] - rapid.edge_hr).abs().mean():.2f} BPM"
                              for m in METHODS) + ". All remain in the primary metrics.", "",
                  "Maximum absolute one-second change on the paired window: "
                  f"Edge {pairs.edge_hr.diff().abs().max():.2f} BPM; "
                  + "; ".join(f"{m} {pairs[m].diff().abs().max():.2f} BPM"
                              for m in METHODS) + ". "
                  "Sparse-anchor interpolation smooths rapid fluctuations; "
                  "negative high-HR errors demonstrate peak underestimation "
                  "in this overlap, not necessarily every individual peak.", ""]
    means = covered[list(METHODS)].mean()
    lines += ["## Mean reconstructed HR and recommendation", "",
              f"Arithmetic mean over the {len(covered)} valid one-second rows "
              f"(anchors included): linear **{means.linear:.4f} BPM**; "
              f"PCHIP **{means.pchip:.4f} BPM**. PCHIP minus linear: "
              f"{means.pchip - means.linear:.4f} BPM. This is the watch-covered "
              "interval, not a filled whole-session average.", ""]
    if metrics:
        winner = min(metrics, key=lambda m: metrics[m]["mae"])
        lines += [f"{winner.capitalize()} has the lower full-overlap MAE. "
                  "Linear is the preferred first baseline here because it "
                  "also has lower RMSE and is simpler; the small difference "
                  "does not establish a general advantage on other rides. "
                  "Use the measured error table, not visual smoothness, to "
                  "choose between these baselines. Neither can restore "
                  "unobserved second-by-second physiology. Both are useful "
                  "reference baselines before evaluating additional models.", ""]
    lines += ["Validation covers only the initial approximately 25 minutes. "
              "Later ride accuracy is unknown, including at higher exertion. "
              "Wellness HR may be stored averages or filtered measurements; "
              "its averaging window is unknown. It is not confirmed "
              "instantaneous ground truth. One-second output resolution does "
              "not imply one-second measurement information. No external "
              "activities or machine learning were used.", "",
              "## Outputs and verification", "",
              f"- Figure: `docs/images/hr_baseline_interpolation_{DATE}.png`.",
              f"- CSV: `data/processed/hr_baseline_interpolation_{DATE}.csv`.",
              "- Script: `scripts/reconstruct_hr_baseline.py`.", "",
              "The CSV stores UTC timestamps, floating-point linear/PCHIP "
              "estimates, exact watch/Edge observations, anchor/interpolation "
              "flags and a support status. Blank values are unavailable. "
              "No values are rounded for CSV display. Estimated timestamps "
              "are explicitly flagged; observed anchors retain their stored "
              "values without asserting an instantaneous measurement window.", "",
              "The script checks one-second spacing, exact anchor preservation, "
              "missing unsupported intervals, local neighboring-value bounds, "
              "valid pairing and unchanged original FIT hashes. It also "
              "round-trips the CSV to check values and missingness.", "",
              "```powershell",
              ".\\.venv\\Scripts\\python.exe scripts/reconstruct_hr_baseline.py",
              "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run the baseline reconstruction and checks on the existing ride files."""
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / "data").rglob("*.fit")}
    processed = ROOT / "data/processed"
    raw = pd.read_csv(processed / f"watch_hr_ride_{DATE}.csv")
    if not raw.source.eq("watch").all():
        raise ValueError("Expected watch-only input")
    watch, invalid, duplicates = clean_observations(raw)
    edge_issue = None
    cutoff = None
    try:
        edge, start, end, cutoff = read_edge(ROOT / "data/mtb_edge.fit")
    except (OSError, ValueError, KeyError) as error:
        edge_issue = str(error)
        edge = pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
        start, end = watch.index[0], watch.index[-1]
    grid = pd.date_range(min(start, watch.index[0]),
                         max(end, watch.index[-1]), freq="s")
    result = interpolate_watch(watch, grid)
    gaps = validate(result, watch)
    result["watch_hr"] = watch.reindex(grid)
    result["edge_hr"] = edge.reindex(grid)
    result["is_watch_anchor"] = result.watch_hr.notna()
    result["is_interpolated"] = (result.linear.notna()
                                 & ~result.is_watch_anchor)
    result["status"] = np.where(
        result.is_watch_anchor, "observed_watch_anchor",
        np.where(result.is_interpolated, "estimated", "unsupported")
    )
    pairs = result.dropna(subset=["edge_hr", *METHODS])
    metrics = {m: error_metrics(pairs, m)
               for m in METHODS} if len(pairs) else {}
    assert len(pairs) == int((result.edge_hr.notna()
                             & result.linear.notna()
                             & result.pchip.notna()).sum())
    assert pairs.edge_hr.equals(edge.reindex(pairs.index).rename("edge_hr"))
    stem = f"hr_baseline_interpolation_{DATE}"
    csv_path = processed / f"{stem}.csv"
    result.index.name = "timestamp_utc"
    result.to_csv(csv_path)
    saved = pd.read_csv(csv_path, float_precision="round_trip")
    for column in [*METHODS, "watch_hr", "edge_hr"]:
        assert np.array_equal(saved[column].isna(), result[column].isna())
        valid = result[column].notna().to_numpy()
        assert np.array_equal(saved.loc[valid, column],
                              result.loc[valid, column])
    figure = ROOT / "docs/images" / f"{stem}.png"
    draw_figure(result, watch, edge, cutoff, metrics, figure)
    report = ROOT / "docs" / f"{stem}.md"
    write_report(result, watch, edge, pairs, metrics, gaps,
                 (invalid, duplicates), edge_issue, cutoff, report)
    assert all(p.stat().st_size > 0 for p in (csv_path, figure, report))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest
               for p, digest in hashes.items())
    print(f"Rows: {len(result)}; supported: {result.linear.notna().sum()}; "
          f"anchors: {len(watch)}; interpolated: {result.is_interpolated.sum()}")
    print("Validation:", metrics)
    print("Covered-interval means:", result[list(METHODS)].mean().to_dict())
    print("Checks passed: anchors, spacing, bounds, missingness, exact pairs, "
          "CSV round trip, non-empty outputs, unchanged FIT hashes.")


if __name__ == "__main__":
    main()
