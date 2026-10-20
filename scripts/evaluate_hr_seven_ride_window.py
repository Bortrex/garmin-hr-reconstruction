"""Establish the seven-ride Window v2 benchmark with exactly three fits.

Run: .venv/Scripts/python scripts/evaluate_hr_seven_ride_window.py
Prior results are read as context, never refitted or overwritten.
"""

import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import evaluate_hr_ml_windows as window
from evaluate_hr_mtb_training import markdown_table
from audit_cycling_predictors import runs

base = window.base
ROOT, ZONE = base.ROOT, base.ZONE
ROADS = tuple("road_ride_" + str(i) for i in range(1, 6))
MTB = ("mtb_ride_1", "mtb_ride_2")
RIDES = ROADS + MTB
COLUMNS = window.BASE_COLUMNS + window.WINDOW_COLUMNS
METHODS = ("linear", "seven_ride")
STEM = "hr_seven_ride_window_benchmark"
METRICS = ROOT / "data/processed/hr_seven_ride_window_metrics.csv"
CANDIDATE = ROOT / "data/processed/hr_seven_ride_candidate_2026-10-16.csv"
REPORT = ROOT / "docs" / (STEM + ".md")
FIGURE = ROOT / "docs/images" / (STEM + ".png")


def fit(names, samples, fits):
    """Fit once on declared rides with the unchanged unweighted selection."""
    x = pd.concat([samples[name][0] for name in names])
    y = pd.concat([samples[name][1] for name in names])
    assert x.index.is_unique and x.index.equals(y.index)
    assert len(COLUMNS) == 50 and list(x.columns) == COLUMNS
    model = HistGradientBoostingRegressor(**base.PARAMETERS).fit(x, y)
    assert model.n_iter_ == 120 and not model.do_early_stopping_
    road_n = sum(len(samples[n][0]) for n in names if n in ROADS)
    fits.append([" + ".join(names), len(x), f"{100*road_n/len(x):.2f}"])
    return {"models": {"seven_ride": model}, "training": names}


def evaluate(ride, packages, fitted, domain, checks):
    """Score common hidden timestamps using the established metric definitions."""
    assert ride["name"] not in fitted["training"]
    masks, peaks, _ = window.diagnostic_masks(ride)
    phases = base.PHASES if domain == "mtb_holdout" else (0,)
    conditions = ("point", "averaged") if domain == "mtb_holdout" else (
        "actual_watch",)
    rows = []
    for phase in phases:
        predictions = {condition: base.predictions(packages[phase, condition], fitted)
                       for condition in conditions}
        support = ride["reference"].reindex(ride["frame"].index).notna()
        for values in predictions.values():
            assert values.seven_ride.notna().equals(values.linear.notna())
            support &= values[list(METHODS)].notna().all(axis=1)
        common = support.copy()
        for condition in conditions:
            common.loc[packages[phase, condition]["anchors"].index] = False
        digest = hashlib.sha256(
            common.index.asi8[common].tobytes()).hexdigest()
        for condition, values in predictions.items():
            package = packages[phase, condition]
            for name in METHODS:
                estimate = values[name]
                metrics = window.score(ride, estimate, common, masks, peaks)
                error = (
                    estimate - ride["reference"].reindex(estimate.index)).where(common)
                for label, mask in (("stationary", ride["frame"].stationary_evidence),
                                    ("restart", package["x"].restart_age_s.le(30))):
                    selected = error[mask].dropna()
                    metrics[label + "_n"] = len(selected)
                    metrics[label + "_mae"] = selected.abs().mean()
                metrics.update(base.extra_metrics(estimate, values.linear,
                                                  ride["reference"], support, peaks))
                finite = estimate.dropna()
                assert finite.gt(0).all() and finite.lt(255).all()
                rows.append({"domain": domain, "ride": ride["name"], "phase": phase,
                             "condition": condition, "model": name,
                             "training": "none" if name == "linear" else "+".join(fitted["training"]),
                             **metrics, "average_hr_error": metrics["bias"],
                             "supported_n": len(finite), "supported_percent": 100 * len(finite) / len(values),
                             "prediction_min": finite.min(), "prediction_max": finite.max(),
                             "evaluation_timestamp_sha256": digest, "metric_origin": "executed"})
    # Keep declared anchors fixed while replacing every hidden HR value.
    package = next(iter(packages.values()))
    replacement = ride["reference"] * 0 + 200
    rebuilt = base.examples(ride["frame"], package["anchors"], replacement.index,
                            base.activity_features(ride["frame"]))
    rebuilt["x"] = rebuilt["x"].join(package["x"][window.WINDOW_COLUMNS])
    pd.testing.assert_frame_equal(rebuilt["x"], package["x"])
    pd.testing.assert_frame_equal(base.predictions(
        rebuilt, fitted), base.predictions(package, fitted))
    checks.append(ride["name"] + ": held out entirely; hidden-HR substitution leaves features/predictions identical; exact anchors, numeric range and common timestamp support verified.")
    return rows


def saved_comparison(current):
    """Check the unchanged linear protocol before importing saved four-ride rows."""
    previous = pd.read_csv(ROOT / "data/processed/hr_mtb_training_metrics.csv")
    keys = ["domain", "ride", "phase", "condition"]
    linear = current.loc[current.model.eq("linear")].set_index(keys)
    old_linear = previous.loc[previous.model.eq(
        "linear")].set_index(keys).loc[linear.index]
    columns = old_linear.select_dtypes(
        include=[np.number]).columns.intersection(linear.columns)
    assert len(linear) == 13
    assert np.allclose(linear[columns], old_linear[columns],
                       rtol=1e-10, atol=1e-10, equal_nan=True)
    old = previous.loc[previous.model.eq("B")].copy()
    assert len(old) == 13
    pd.testing.assert_index_equal(old.set_index(
        keys).index.sort_values(), linear.index.sort_values())
    assert np.array_equal(old.set_index(keys).loc[linear.index, "n"], linear.n)
    old["model"], old["metric_origin"] = "four_ride", "saved_four_ride"
    return pd.concat([current, old], ignore_index=True)


def reconstruct_target(ride, watch, fitted, windows):
    """Save one new candidate, preserving recorded Edge and separate watch HR."""
    frame = ride["frame"]
    edge = ride["reference"].reindex(frame.index)
    cutoff = ride["reference"].index[-1]
    anchors = pd.concat([pd.Series(edge.loc[cutoff], index=pd.DatetimeIndex([cutoff])),
                         watch.loc[watch.index > cutoff]])
    package = base.examples(frame, anchors, frame.index,
                            base.activity_features(frame))
    package["x"] = package["x"].join(windows)
    predicted = base.predictions(package, fitted)
    table = pd.DataFrame({"edge_hr": edge, "watch_hr": watch.reindex(frame.index),
                          "interpolation_baseline": predicted.linear,
                          "model_estimate": predicted.seven_ride,
                          "final_hr": predicted.seven_ride.where(edge.isna(), edge)})
    table["estimated"] = table.final_hr.notna() & edge.isna()
    table["edge_observed"] = edge.notna()
    table["watch_observed"] = table.watch_hr.notna()
    table["source_status"] = "missing_unsupported"
    table.loc[table.estimated, "source_status"] = "seven_ride_window_estimate"
    table.loc[table.edge_observed, "source_status"] = "recorded_edge"
    table["quality_indicator"] = np.where(
        frame.climb30.notna(), "terrain_supported", "terrain_missing")
    table.loc[frame.stationary_evidence,
              "quality_indicator"] += ";stationary_evidence"
    table.index.name = "timestamp_utc"
    assert np.array_equal(table.loc[edge.notna(), "final_hr"], edge.dropna())
    assert np.array_equal(table.loc[watch.index, "watch_hr"], watch)
    future = watch.loc[watch.index > cutoff]
    assert np.array_equal(table.loc[future.index, "final_hr"], future)
    assert table.loc[table.index > watch.index[-1], "final_hr"].isna().all()
    finite = table.final_hr.dropna()
    assert finite.gt(0).all() and finite.lt(255).all()
    table.to_csv(CANDIDATE)
    saved = pd.read_csv(CANDIDATE)
    assert np.array_equal(
        saved.loc[table.edge_observed.values, "final_hr"], edge.dropna())
    assert np.array_equal(
        saved.loc[table.watch_observed.values, "watch_hr"], watch)
    assert pd.DatetimeIndex(pd.to_datetime(
        saved.timestamp_utc, utc=True)).equals(frame.index)
    return table


def figure(ride, table, metrics):
    """Show the target reconstruction and directly comparable held-out errors."""
    plt = window.plt
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.5, 1]})
    ax = axes[0]
    estimated = table.final_hr.where(table.estimated)
    ax.plot(table.index, estimated, color="#c35b2d", lw=1.25,
            label="Seven-ride HR estimate (not measured)")
    ax.plot(table.index, table.edge_hr, color="#237495",
            lw=1.1, label="Recorded Edge HR")
    watch = table.watch_hr.dropna()
    ax.scatter(watch.index, watch, color=".15", s=13, zorder=5,
               label="Recorded watch wellness observations")
    cutoff = ride["reference"].index[-1]
    ax.axvline(cutoff, color="#237495", lw=1,
               ls="--", label="Last recorded Edge HR")
    for i, indices in enumerate(runs(ride["frame"].probable_stop, ride["frame"].index)):
        times = ride["frame"].index[indices]
        ax.axvspan(times[0], times[-1], color=".5", alpha=.09,
                   label="Probable sustained stops" if i == 0 else None)
    ax.set(title="A — October 16 MTB: recorded observations and one-second estimates",
           xlabel="Local time — Europe/Brussels", ylabel="HR (BPM)",
           xlim=(table.index[0], table.index[-1]))
    ax.xaxis.set_major_formatter(window.mdates.DateFormatter("%H:%M", tz=ZONE))
    ax.legend(ncol=2, fontsize=9, loc="lower right")
    ax.grid(alpha=.2)
    primary = metrics.loc[metrics.domain.eq(
        "mtb_holdout") & metrics.condition.eq("point")]
    labels = {"linear": "Linear interpolation",
              "four_ride": "Four rides (saved)", "seven_ride": "Seven rides"}
    colors = ("#72804a", "#257c9b", "#c35b2d")
    for i, (model, label) in enumerate(labels.items()):
        groups = [primary.loc[primary.ride.eq(
            name) & primary.model.eq(model)] for name in MTB]
        means = np.array([g.mae.mean() for g in groups])
        low, high = np.array([g.mae.min() for g in groups]), np.array(
            [g.mae.max() for g in groups])
        bars = axes[1].bar(np.arange(2) + (i - 1) * .24, means, .22, color=colors[i], label=label,
                           yerr=[means - low, high - means], capsize=3)
        for bar, group in zip(bars, groups):
            axes[1].annotate(f"MAE {group.mae.mean():.2f}\nRMSE {group.rmse.mean():.2f}",
                             (bar.get_x() + bar.get_width() /
                              2, group.mae.max() + .05),
                             ha="center", va="bottom", fontsize=9)
    axes[1].set(xticks=[0, 1], xticklabels=list(MTB), ylabel="MAE (BPM)",
                title="B — MTB holdouts: phase-mean MAE and min–max; labels include mean RMSE")
    axes[1].set_ylim(0, primary.mae.max() + 1)
    axes[1].legend(ncol=3, loc="lower center", fontsize=9)
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("Seven-Ride Window v2 — principal experimental benchmark\nFive road + two MTB rides; fixed 50 features and model settings", fontsize=15)
    fig.savefig(FIGURE, dpi=170)
    plt.close(fig)


def interpretation(metrics):
    """Describe the actual per-ride comparison without selecting a new experiment."""
    data = metrics.loc[metrics.domain.eq(
        "mtb_holdout") & metrics.condition.eq("point")]
    statements, improves = [], []
    for ride in MTB:
        mean = data.loc[data.ride.eq(ride)].groupby(
            "model").mean(numeric_only=True)
        old, new = mean.loc["four_ride"], mean.loc["seven_ride"]
        improves.append(new.mae < old.mae and new.rmse < old.rmse)
        statements.append(f"{ride}: seven-ride MAE/RMSE {new.mae:.3f}/{new.rmse:.3f} BPM "
                          f"versus four-ride {old.mae:.3f}/{old.rmse:.3f} "
                          f"(changes {new.mae-old.mae:+.3f}/{new.rmse-old.rmse:+.3f}).")
    summary = ("MAE and RMSE improve on both held-out MTB rides." if all(improves)
               else "MAE and RMSE do not both improve on both held-out MTB rides.")
    return summary, statements


def write_report(metrics, inventory, packages, samples, fits, table, checks, protected_n):
    """Document the new fixed benchmark, saved comparison and actual candidate."""
    summary, statements = interpretation(metrics)
    finite = table.final_hr.dropna()
    steps = table.final_hr.diff().abs().where(
        table.estimated & table.estimated.shift(fill_value=False))
    lines = ["# Seven-ride Window v2 benchmark", "", "**Principal experimental benchmark for future comparisons.** " + summary,
             " This status does not establish true second-by-second HR or overwrite prior reconstructions.", "",
             "## Fixed experiment", "",
             "Exactly three HGB fits: hold out mtb_ride_1, hold out mtb_ride_2, then all seven rides. Each holdout trains five roads plus the other MTB. No virtual activity, other fit, weighting, resampling, search or parameter change. Frozen fold models serve all phases and both observation assumptions. Four-ride results are imported from `data/processed/hr_mtb_training_metrics.csv`, model B; no previous model is refitted.", "",
             "Reuse `prepare`, `window_features`, `add_windows`, `training_rows`, `examples`, `predictions`, `score` and `extra_metrics`. Exactly 50 ordered inputs (20 context/movement/terrain + 30 window summaries), four half-open regions [-60,-30), [-30,0), [0,30), [30,60). Existing speed and centered terrain preprocessing retain their underlying support. Missing feature means remain NaN; coverage stays explicit. Fixed parameters: `" + str(
                 base.PARAMETERS) + "`. Same linear residual target, centered five-sample residual mean and 4f(1−f) anchor taper.", "",
             "UTC internally; Europe/Brussels for displayed dates and plots. Replay the exact October 16 pattern: 142×60s and 15×120s, 158 observations across 172min, cycling/truncating by elapsed time at phases 0/20/40s. Point anchors equal observed HR; centered averages require all 60 recorded samples [t−30,t+29]. Averaging is an alternative assumption, not established Garmin behavior. Train on point anchors only using the unchanged deterministic five-second policy and cyclic phase assignment, excluding anchors; no duplicate targets. No gap >180s interpolation or extrapolation; recording gaps split all support. Future non-HR data are permitted offline, never hidden HR.", "",
             "## Input checks and training distribution", ""]
    lines += markdown_table(["Activity", "Local start/end", "Span min", "HR rows / elapsed coverage %",
                            "Gaps / max spacing s", "Longest block s", "Speed/GPS/distance/elevation valid", "Stationary rows"], inventory)
    lines += ["All seven files have distinct SHA-256 and disjoint recorded timestamps; each contributes eligible training rows. New roads 3/4/5 contain valid 1Hz HR within recorded segments. Road 5 has only about 60% elapsed coverage and an 89-minute gap; its long continuous recorded blocks remain useful, while all missing sections stay unsupported. Missing speed/GPS on roads 3/5 remain missing; no new altitude step >2m was found in these three road files. Cadence is present on roads 3/5 and temperature on all three, but neither enters the 50 predictors; no new power input. Native training elevation uses existing quality masks. Target elevation uses the existing aligned Strava GPX.", ""]
    rows = []
    for name in RIDES:
        x = samples[name][0]
        complete = x[[label + "_record_coverage" for label,
                      _, _ in window.REGIONS]].eq(1).all(axis=1)
        terrain = x[[label + "_terrain_coverage" for label,
                     _, _ in window.REGIONS]].eq(1).all(axis=1)
        anchors = [len(packages[name][phase, "point"]["anchors"])
                   for phase in base.PHASES]
        averaged = [len(packages[name][phase, "averaged"]["anchors"])
                    for phase in base.PHASES]
        rows.append([name, len(x), "/".join(map(str, anchors)), "/".join(map(str, averaged)),
                     f"{100*x.isna().mean().mean():.2f}", f"{100*complete.mean():.2f}", f"{100*terrain.mean():.2f}"])
    lines += markdown_table(["Ride", "Training rows", "Point anchors 0/20/40", "Average anchors 0/20/40",
                            "Missing feature cells %", "Complete record windows %", "Complete terrain windows %"], rows)
    lines += markdown_table(["Training partition",
                            "Total rows", "Road %"], fits)
    lines += ["Proportions describe the unchanged equal-per-example policy; road dominance is not corrected by weighting. Window coverage is on selected training rows. Stops invalidate terrain support but do not erase recorded HR or valid stationary features.", "",
              "## Comparable held-out results", "",
              "All methods and both anchor conditions score the same hidden timestamps per ride/phase, excluding anchors in either condition. Every one of the 13 recomputed linear rows reproduces saved numeric metrics (including coverage, event counts and roughness) within 1e-10. The unmodified four-ride evaluator used the same FITs, deterministic schedule and common-mask rule, and enforced model support equal to linear. Together these establish protocol-equivalent timestamps; the old CSV did not persist timestamp hashes, so direct old-mask hash comparison is unavailable. The new executed rows persist SHA-256 of evaluated UTC timestamps for future comparisons. Saved four-ride rows are labelled `saved_four_ride`; no historical timestamps are invented.", ""]
    lines += [summary, ""] + [statement + "\n" for statement in statements]
    held = metrics.loc[metrics.domain.eq("mtb_holdout")]
    rows = []
    for (ride, condition, phase), group in held.groupby(["ride", "condition", "phase"]):
        g = group.set_index("model")
        rows.append([ride, condition, phase] + [f"{g.loc[m,'mae']:.3f}/{g.loc[m,'rmse']:.3f}" for m in ("linear", "four_ride", "seven_ride")] + [
                    f"{g.loc['seven_ride','bias']:+.3f}", f"{int(g.loc['seven_ride','n'])} ({g.loc['seven_ride','coverage_percent']:.2f}%)"])
    lines += markdown_table(["Held out", "Anchors", "Phase", "Linear MAE/RMSE",
                            "Four-ride saved", "Seven-ride", "New bias", "Common n (%)"], rows)
    rows = []
    for (ride, condition, model), group in held.groupby(["ride", "condition", "model"]):
        rows.append([ride, condition, model, f"{group.mae.mean():.3f} [{group.mae.min():.3f}, {group.mae.max():.3f}]",
                    f"{group.rmse.mean():.3f} [{group.rmse.min():.3f}, {group.rmse.max():.3f}]"])
    lines += markdown_table(["Ride", "Anchors", "Model",
                            "Mean MAE [phase range]", "Mean RMSE [phase range]"], rows)
    lines += ["### High HR, peaks, recovery and stops", "",
              "Established score-only definitions: high HR = reference upper decile; rise/recovery = ≥5/≤−5 BPM change over 30s within a continuous block. True peaks require ≥10 BPM prominence and ≥120s separation. Peak amplitude/timing use a fully scored ±15s local maximum; timing is bounded by this window, not unrestricted matching. Unmatched predicted peaks lack a reference peak within ±30s. Roughness includes consecutive supported anchor timestamps. Stationary and ≤30s restart errors use only recorded reference HR. CSV supplies all phase/scenario subset counts and errors; a missing event metric is NaN, never zero.", ""]
    primary = held.loc[held.condition.eq("point")]
    rows = []
    for (ride, model), group in primary.groupby(["ride", "model"]):
        rows.append([ride, model, f"{group.high_mae.mean():.2f}", f"{group.rise_mae.mean():.2f}/{group.recovery_mae.mean():.2f}", f"{group.peak_amplitude_mae.mean():.2f}/{group.peak_timing_mae_s.mean():.2f}",
                    f"{group.stationary_mae.mean():.2f}/{group.restart_mae.mean():.2f}", f"{group.unmatched_peaks.mean():.1f}", f"{group.step_p99.mean():.3f}/{group.max_step.max():.3f}", f"{group.steps_gt5_percent.max():.3f}"])
    lines += markdown_table(["Ride", "Model", "High MAE", "Rise/recovery MAE", "Peak amp/timing s",
                            "Stationary/restart MAE", "Unmatched mean", "Step p99 mean/max", ">5 BPM/s % max"], rows)
    for ride in MTB:
        g = primary.loc[primary.ride.eq(ride)].groupby(
            "model").mean(numeric_only=True)
        old, new = g.loc["four_ride"], g.loc["seven_ride"]
        lines += [f"{ride}: seven-minus-four high/rise/recovery MAE {new.high_mae-old.high_mae:+.3f}/{new.rise_mae-old.rise_mae:+.3f}/{new.recovery_mae-old.recovery_mae:+.3f} BPM; peak amplitude/timing {new.peak_amplitude_mae-old.peak_amplitude_mae:+.3f} BPM/{new.peak_timing_mae_s-old.peak_timing_mae_s:+.3f}s. These diagnostics must be considered alongside mean errors.", ""]
    missing = metrics.loc[metrics.model.eq("seven_ride"), ["high_mae", "rise_mae", "recovery_mae",
                                                           "transition_mae", "peak_amplitude_mae", "stationary_mae", "restart_mae"]].isna().sum()
    lines += ["Unavailable seven-ride metric rows: " + ", ".join(f"{name}={value}" for name, value in missing.items()) + ".", "",
              "## October 16 secondary overlap and reconstruction", "",
              "The frozen final model uses all seven training rides. Its scored Edge overlap uses only 21 real watch observations; no withheld Edge HR is an input. The already examined overlap is not a pristine independent test and does not tune the model.", ""]
    rows = []
    for _, r in metrics.loc[metrics.domain.eq("mtb_overlap")].iterrows():
        rows.append([r.model, f"{r.mae:.3f}/{r.rmse:.3f}", f"{r.bias:+.3f}", f"{r.high_mae:.2f}", f"{r.recovery_mae:.2f}",
                    f"{r.peak_amplitude_mae:.2f}/{r.peak_timing_mae_s:.2f}", f"{int(r.n)} ({r.coverage_percent:.2f}%)"])
    lines += markdown_table(["Model", "MAE/RMSE", "Bias", "High MAE",
                            "Recovery MAE", "Peak amp/timing", "Common n (%)"], rows)
    lines += ["Candidate construction follows the prior handoff policy: preserve all original Edge HR, use its last recorded value as a boundary only for later estimates, and use subsequent real watch anchors. This known handoff value never enters the withheld-overlap diagnostic. All 158 watch summaries are preserved separately in `watch_hr`; `final_hr` retains Edge priority where both exist. Watch pseudo-point constraints may differ from instantaneous HR. No unsupported final timestamp is filled.", ""]
    lines += markdown_table(["Supported mean BPM", "Supported max BPM", "Recorded Edge seconds", "Estimated seconds", "Supported n (%)", "Unsupported seconds", "Estimated step p99/max BPM/s"],
                            [[f"{finite.mean():.3f}", f"{finite.max():.3f}", int(table.edge_observed.sum()), int(table.estimated.sum()), f"{len(finite)} ({100*len(finite)/len(table):.2f}%)", int(table.final_hr.isna().sum()), f"{steps.quantile(.99):.3f}/{steps.max():.3f}"]])
    lines += ["The supported mean uses recorded and estimated values; the maximum can be an estimate and does not establish the true missing peak. No score is claimed for the unrecorded target HR. Source/quality flags distinguish Edge records, model estimates and missing support; watch observations retain their own column.", "",
              "## Limitations and benchmark use", "",
              "This is the new experimental benchmark to beat, regardless of whether every metric improves. Seven training rides do not create seven independent MTB validation rides: only the two held-out MTBs test transfer, and phases/seconds are correlated. MTB 1 has OHR metadata and MTB 2 an HRM 200; road recording paths, seasons and HR artifacts differ. Native training versus Strava target terrain and frequent MTB stops create domain shifts. Sparse wellness observations, unknown exact averaging/latency, uncertain short peaks, sensor errors and absent target power/cadence remain limitations. Neither this reconstruction nor a lower exploratory error proves true second-by-second physiology.", "",
              "![Seven-ride benchmark](images/hr_seven_ride_window_benchmark.png)", "",
              "Panel A distinguishes recorded Edge/watch observations from estimates and shades probable sustained stops; the dashed marker is the last recorded Edge HR (observed disconnection boundary). Panel B uses protocol-comparable point-anchor scores. Bars show phase means, error bars the dependent phase min–max, not confidence intervals; labels include RMSE.", "",
              "```powershell", ".venv\\Scripts\\python.exe scripts/evaluate_hr_seven_ride_window.py", "```", "",
              "Outputs: this report, `docs/images/hr_seven_ride_window_benchmark.png`, `data/processed/hr_seven_ride_window_metrics.csv`, `data/processed/hr_seven_ride_candidate_2026-10-16.csv`; implementation `scripts/evaluate_hr_seven_ride_window.py`. README presents this as the principal experimental benchmark; historical outputs remain intact.", ""]
    lines += ["- " + item for item in checks]
    lines += [f"- SHA-256 preservation checks pass for {protected_n} previous files; README is the only existing file intentionally updated.",
              "- Exactly three fits with fixed random state; no repeat fit, search, new dependency or separate suite. Nonempty artifacts and README link are checked; figure readability is inspected after generation.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def update_readme(metrics, table):
    """Present two result-based paragraphs first while retaining historical text."""
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    heading = "## Seven-Ride Window-Enhanced HR Reconstruction"
    if heading in text:
        start = text.index(heading)
        end = text.find("\n## ", start + len(heading))
        text = text[:start] + (text[end + 1:] if end >= 0 else "")
    first_line, remainder = text.split("\n", 1)
    summary, _ = interpretation(metrics)
    primary = metrics.loc[metrics.domain.eq(
        "mtb_holdout") & metrics.condition.eq("point")]
    means = primary.groupby(["ride", "model"]).mean(numeric_only=True)
    results = []
    for i, ride in enumerate(MTB, 1):
        old, new = means.loc[(ride, "four_ride")
                             ], means.loc[(ride, "seven_ride")]
        results.append(f"MTB {i} MAE/RMSE: {new.mae:.3f}/{new.rmse:.3f} BPM "
                       f"(four-ride {old.mae:.3f}/{old.rmse:.3f}).")
    gains = {metric: sum(means.loc[(ride, "seven_ride"), metric] <
                         means.loc[(ride, "four_ride"), metric] for ride in MTB)
             for metric in ("high_mae", "peak_amplitude_mae", "recovery_mae")}
    finite = table.final_hr.dropna()
    paragraph = (summary + " " + " ".join(results) + " "
                 f"High-HR, peak-amplitude and recovery MAE improve on {gains['high_mae']}/2, "
                 f"{gains['peak_amplitude_mae']}/2 and {gains['recovery_mae']}/2 rides respectively. "
                 f"The October 16 estimate has supported mean HR {finite.mean():.2f} BPM, "
                 f"peak {finite.max():.2f} BPM and {100*len(finite)/len(table):.2f}% coverage. "
                 "Garmin Edge/watch observations remain recorded data; reconstructed seconds are estimates. "
                 "Sparse watch observations, uncertain short HR peaks, sensor errors and limited independent "
                 "MTB validation remain limitations. This is the new experimental benchmark to beat, "
                 "not proof of true second-by-second HR.")
    section = (heading + "\n\n"
               "The principal experimental benchmark combines five outdoor road rides and two MTB rides. "
               "Window v2 uses 50 temporal/context features and gradient boosting to predict corrections "
               "to linear HR interpolation. Its fixed model is evaluated with two MTB holdouts before "
               "one final fit on all seven rides; previous reconstructions remain available below.\n\n"
               + paragraph + "\n\n"
               "![Seven-ride window-enhanced heart-rate reconstruction](docs/images/hr_seven_ride_window_benchmark.png)\n\n"
               "See the [technical report](docs/hr_seven_ride_window_benchmark.md) and "
               "[candidate CSV](data/processed/hr_seven_ride_candidate_2026-10-16.csv).\n\n"
               "```powershell\n.venv\\Scripts\\python.exe scripts/evaluate_hr_seven_ride_window.py\n```\n\n"
               "*Historical sections below retain their original datasets and model status.*\n\n")
    path.write_text(first_line + "\n\n" + section +
                    remainder.lstrip("\n"), encoding="utf-8")
    assert FIGURE.is_file() and REPORT.is_file()


def main():
    """Validate all inputs, execute three fits and produce the new benchmark."""
    outputs = {ROOT / "scripts/evaluate_hr_seven_ride_window.py",
               METRICS, CANDIDATE, REPORT, FIGURE}
    protected = [p for folder in ("data", "docs", "scripts") for p in (ROOT / folder).rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and p not in outputs]
    before = base.fingerprints(protected)
    files = [ROOT / "data" / (name + ".fit") for name in RIDES]
    digests = list(base.fingerprints(files).values())
    assert len(set(digests)) == 7
    rides, packages, samples, inventory = {}, {}, {}, []
    watch, _, _ = window.clean_observations(pd.read_csv(
        ROOT / "data/processed/watch_hr_ride_2026-10-16.csv"))
    pattern = np.diff(watch.index.asi8) // 1_000_000_000
    assert len(watch) == 158 and (pattern == 60).sum(
    ) == 142 and (pattern == 120).sum() == 15
    window.verify_windows()
    # No fitting until every requested activity passes usability/duplicate checks.
    for name in RIDES:
        ride = window.prepare(name)
        reference, frame = ride["reference"], ride["frame"]
        for previous in rides.values():
            if len(reference.index.intersection(previous["reference"].index)):
                raise ValueError(
                    "Overlapping activity timestamps: inspect duplicates before fitting")
        blocks = base.continuous_blocks(reference)
        longest = max(map(len, blocks))
        if len(reference) < 1201 or longest < 1201:
            raise ValueError(
                name + ": insufficient usable HR; no models fitted")
        gaps = np.diff(reference.index.asi8) / 1e9
        span = (reference.index[-1] - reference.index[0]).total_seconds()
        fields = [frame.speed_raw.count(), frame[["position_lat", "position_long"]].notna(
        ).all(axis=1).sum(), frame.distance.count(), frame.altitude_raw.count()]
        inventory.append([name, reference.index[0].tz_convert(ZONE).isoformat() + " / " + reference.index[-1].tz_convert(ZONE).isoformat(),
                         f"{span/60:.2f}", f"{len(reference)} / {100*len(reference)/(span+1):.2f}", f"{int((gaps>1).sum())} / {gaps.max():.0f}", longest, "/".join(map(str, fields)), int(frame.stationary_evidence.sum())])
        packs = window.add_windows(base.synthetic(
            ride, pattern, base.activity_features(frame)), window.window_features(frame))
        for package in packs.values():
            assert list(package["x"].columns) == COLUMNS
            for left, right in package["spans"]:
                assert (right - left).total_seconds() <= 180
                assert np.all(
                    np.diff(frame.loc[left:right].index.asi8) == 1_000_000_000)
        samples[name] = window.training_rows([(ride, packs)])
        if len(samples[name][0]) < 200:
            raise ValueError(
                name + ": insufficient eligible training examples; no fits run")
        rides[name], packages[name] = ride, packs
        print("Prepared", name, "HR", len(reference),
              "training examples", len(samples[name][0]), flush=True)
    checks = ["All seven official FIT decodes/integrity checks pass; file hashes distinct, recorded timestamps disjoint, all references usable and all seven contribute eligible unique targets before fitting.",
              "Reused exact window-boundary/gap/stop/terrain checks; 50 ordered predictors unchanged; declared point and complete 60-s average anchors verified; <=180s spans remain in continuous record blocks."]
    fits, rows = [], []
    for held in MTB:
        names = tuple(name for name in RIDES if name != held)
        print("One fit, holding out", held, flush=True)
        fitted = fit(names, samples, fits)
        rows.extend(
            evaluate(rides[held], packages[held], fitted, "mtb_holdout", checks))
    print("One final fit on all seven rides", flush=True)
    fitted = fit(RIDES, samples, fits)
    target = window.prepare("mtb_edge")
    target["frame"] = target["strava"]
    windows = window.window_features(target["frame"])
    overlap = watch.loc[target["reference"].index[0]                        :target["reference"].index[-1]]
    original = base.examples(
        target["frame"], overlap, target["reference"].index, base.activity_features(target["frame"]))
    target_packages = window.add_windows(
        {(0, "actual_watch"): original}, windows)
    rows.extend(evaluate(target, target_packages,
                fitted, "mtb_overlap", checks))
    metrics = saved_comparison(pd.DataFrame(rows))
    assert len(metrics) == 39 and len(fits) == 3
    assert metrics.groupby(["domain", "ride", "phase"]).n.nunique().eq(1).all()
    checks.append("All 13 linear comparison rows reproduce saved numeric metrics/support within 1e-10; four-ride rows imported without refitting. Executed rows store evaluated timestamp hashes.")
    table = reconstruct_target(target, watch, fitted, windows)
    checks.append("Candidate CSV read back: original Edge HR and all 158 separate watch observations exact, future anchors exact, UTC timeline preserved, final unsupported timestamps missing and numeric HR valid.")
    metrics.to_csv(METRICS, index=False)
    figure(target, table, metrics)
    assert base.fingerprints(protected) == before
    write_report(metrics, inventory, packages, samples,
                 fits, table, checks, len(protected))
    update_readme(metrics, table)
    assert all(path.stat().st_size > 0 for path in outputs)
    assert base.fingerprints(protected) == before
    print(metrics.loc[metrics.condition.isin(["point", "actual_watch"])].groupby(["domain", "ride", "model"])[
          ["mae", "rmse", "bias", "high_mae", "recovery_mae"]].mean().round(4).to_string())
    print("Candidate mean/max/coverage:", table.final_hr.mean(),
          table.final_hr.max(), 100 * table.final_hr.notna().mean())
    print("Training rows:", {name: len(x) for name, (x, _) in samples.items(
    )}, "; exactly three fits; protected files unchanged:", len(protected))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
