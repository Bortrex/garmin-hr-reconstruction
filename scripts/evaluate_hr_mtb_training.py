"""Compare MTB-only and mixed training with unchanged Window v2 processing.

Run: .venv/Scripts/python scripts/evaluate_hr_mtb_training.py
Exactly six fits: two MTB holdouts and one final fit per configuration.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import evaluate_hr_ml_windows as window
from audit_cycling_predictors import load_records

base = window.base
ROOT, ZONE = base.ROOT, base.ZONE
MTB = ("mtb_ride_1", "mtb_ride_2")
ROADS = ("road_ride_1", "road_ride_2")
COLUMNS = window.BASE_COLUMNS + window.WINDOW_COLUMNS
METHODS = ("linear", "A", "B")
LABELS = {"linear": "Linear interpolation", "A": "MTB only", "B": "Road + MTB"}
COLORS = {"linear": "#72804a", "A": "#257c9b", "B": "#c35b2d"}
STEM = "hr_mtb_training_comparison"


def fit_models(members, samples, partitions):
    """Fit only the two requested unweighted partitions with fixed settings."""
    models, training = {}, {}
    for name, rides in (("A", members), ("B", ROADS + members)):
        x = pd.concat([samples[ride][0] for ride in rides])
        y = pd.concat([samples[ride][1] for ride in rides])
        assert x.index.is_unique and x.index.equals(y.index)
        assert list(x.columns) == COLUMNS and len(COLUMNS) == 50
        model = HistGradientBoostingRegressor(**base.PARAMETERS).fit(x, y)
        assert model.n_iter_ == 120 and not model.do_early_stopping_
        models[name], training[name] = model, rides
        road_n = sum(len(samples[r][0]) for r in rides if r in ROADS)
        partitions.append([name, "+".join(rides), len(x),
                           f"{100*road_n/len(x):.2f}"])
    return {"models": models, "training": training}


def evaluate(ride, packages, fitted, domain, checks):
    """Reuse established errors on common hidden timestamps and keep anchors exact."""
    assert all(
        ride["name"] not in members for members in fitted["training"].values())
    masks, peaks, _ = window.diagnostic_masks(ride)
    phases = base.PHASES if domain == "mtb_holdout" else (0,)
    conditions = ("point", "averaged") if domain == "mtb_holdout" else (
        "actual_watch",)
    rows, cached = [], {}
    for phase in phases:
        predictions = {condition: base.predictions(packages[phase, condition], fitted)
                       for condition in conditions}
        support = ride["reference"].reindex(ride["frame"].index).notna()
        for values in predictions.values():
            support &= values[list(METHODS)].notna().all(axis=1)
            assert values.A.notna().equals(values.B.notna())
        common = support.copy()
        for condition in conditions:
            common.loc[packages[phase, condition]["anchors"].index] = False
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
                             "training": "none" if name == "linear" else "+".join(fitted["training"][name]),
                             **metrics, "average_hr_error": metrics["bias"],
                             "supported_n": len(finite),
                             "supported_percent": 100 * len(finite) / len(values),
                             "prediction_min": finite.min(), "prediction_max": finite.max()})
            cached[phase, condition] = (package, values, common)
    package, expected, _ = next(iter(cached.values()))
    replaced_hr = ride["reference"] * 0 + 200
    rebuilt = base.examples(ride["frame"], package["anchors"], replaced_hr.index,
                            base.activity_features(ride["frame"]))
    rebuilt["x"] = rebuilt["x"].join(package["x"][window.WINDOW_COLUMNS])
    pd.testing.assert_frame_equal(rebuilt["x"], package["x"])
    pd.testing.assert_frame_equal(base.predictions(rebuilt, fitted), expected)
    checks.append(
        ride["name"] + ": held out from both fits; replacing hidden HR leaves inputs/predictions identical; shared support and exact anchors verified.")
    return rows, cached


def candidates(ride, watch, fitted, windows):
    """Preserve measured Edge HR and separate watch summaries in two new files."""
    frame = ride["frame"]
    edge = ride["reference"].reindex(frame.index)
    cutoff = ride["reference"].index[-1]
    anchors = pd.concat([pd.Series(edge.loc[cutoff], index=pd.DatetimeIndex([cutoff])),
                         watch.loc[watch.index > cutoff]])
    package = base.examples(frame, anchors, frame.index,
                            base.activity_features(frame))
    package["x"] = package["x"].join(windows)
    curves = base.predictions(package, fitted)
    summaries, saved = [], {}
    for name, slug in (("A", "mtb_only"), ("B", "mixed")):
        table = pd.DataFrame({"edge_hr": edge, "watch_hr": watch.reindex(frame.index),
                              "interpolation_baseline": curves.linear,
                              "model_estimate": curves[name],
                              "final_hr": curves[name].where(edge.isna(), edge)})
        table["estimated"] = table.final_hr.notna() & edge.isna()
        table["edge_observed"] = edge.notna()
        table["watch_observed"] = table.watch_hr.notna()
        table["source_status"] = "missing_unsupported"
        table.loc[table.estimated, "source_status"] = "unapproved_" + \
            slug + "_candidate"
        table.loc[edge.notna(), "source_status"] = "recorded_edge"
        table["quality_indicator"] = np.where(
            frame.climb30.notna(), "terrain_supported", "terrain_missing")
        table.loc[frame.stationary_evidence,
                  "quality_indicator"] += ";stationary_evidence"
        table.index.name = "timestamp_utc"
        assert np.array_equal(
            table.loc[edge.notna(), "final_hr"], edge.dropna())
        assert np.array_equal(table.loc[watch.index, "watch_hr"], watch)
        future = watch.index[watch.index > cutoff]
        assert np.array_equal(table.loc[future, "final_hr"], watch.loc[future])
        assert table.loc[table.index >
                         watch.index[-1], "final_hr"].isna().all()
        finite = table.final_hr.dropna()
        assert finite.gt(0).all() and finite.lt(255).all()
        steps = table.final_hr.diff().abs().where(frame.dt.eq(1))
        predicted_steps = steps.where(
            table.estimated & table.estimated.shift(fill_value=False))
        path = ROOT / "data/processed" / \
            ("hr_" + slug + "_candidate_2026-10-16.csv")
        table.to_csv(path)
        reread = pd.read_csv(path)
        assert np.array_equal(
            reread.loc[table.edge_observed.values, "final_hr"], edge.dropna())
        assert np.array_equal(
            reread.loc[table.watch_observed.values, "watch_hr"], watch)
        summaries.append([name, len(finite), f"{100*len(finite)/len(table):.2f}",
                          f"{finite.mean():.3f}", f"{finite.max():.2f}",
                          f"{predicted_steps.quantile(.99):.3f}/{predicted_steps.max():.3f}",
                          f"{steps.max():.2f}", int(table.final_hr.isna().sum())])
        saved[name] = table
    assert saved["A"].final_hr.notna().equals(saved["B"].final_hr.notna())
    difference = saved["B"].final_hr - saved["A"].final_hr
    return summaries, difference[saved["A"].estimated].abs()


def plot_comparison(panels, metrics, path):
    """Show fixed central 20-minute intervals and phase ranges, not chosen errors."""
    plt = window.plt
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
    for ax, (ride, cached) in zip(axes[:2], panels):
        package, values, _ = cached[0, "point"]
        block = max(base.continuous_blocks(ride["reference"]), key=len)
        center = block.index[0] + (block.index[-1] - block.index[0]) / 2
        start = (center - pd.Timedelta(minutes=10)).floor("s")
        end = start + pd.Timedelta(minutes=20)
        actual = ride["reference"].loc[start:end]
        ax.plot(actual.index, actual, color=".65",
                lw=1.2, label="Recorded HR reference")
        for name in METHODS:
            part = values.loc[start:end, name]
            ax.plot(part.index, part, color=COLORS[name], lw=1.3,
                    ls="--" if name == "linear" else "-", label=LABELS[name])
        sparse = package["anchors"].loc[start:end]
        ax.scatter(sparse.index, sparse, color=".15", s=20,
                   zorder=5, label="Simulated point observations")
        ax.set(title=ride["name"] + " — held out; phase 0, fixed central 20 minutes",
               ylabel="HR (BPM)", xlabel="Local time — Europe/Brussels", xlim=(start, end))
        ax.xaxis.set_major_formatter(
            window.mdates.DateFormatter("%H:%M", tz=ZONE))
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=.2)
    primary = metrics.loc[metrics.domain.eq(
        "mtb_holdout") & metrics.condition.eq("point")]
    for i, name in enumerate(METHODS):
        groups = [primary.loc[primary.ride.eq(
            ride) & primary.model.eq(name)] for ride in MTB]
        mean = np.array([g.mae.mean() for g in groups])
        low, high = np.array([g.mae.min() for g in groups]), np.array(
            [g.mae.max() for g in groups])
        axes[2].bar(np.arange(2) + (i - 1) * .23, mean, .21, color=COLORS[name],
                    label=LABELS[name], yerr=[mean - low, high - mean], capsize=4)
    axes[2].set(xticks=[0, 1], xticklabels=list(MTB), ylabel="MAE (BPM)",
                title="MTB holdouts — phase mean and min–max (0 / 20 / 40s)")
    axes[2].legend(ncol=3)
    axes[2].grid(axis="y", alpha=.2)
    fig.suptitle(
        "Window v2: MTB-only versus road + MTB training\nUnchanged 50 features, fixed model and residual postprocessing", fontsize=14)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def markdown_table(headers, rows):
    """Render compact Markdown tables with existing dependencies only."""
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + [
        "| " + " | ".join(str(v) for v in row) + " |" for row in rows] + [""]


def recommend(metrics):
    """Require consistent held-out benefits; never select from target overlap alone."""
    primary = metrics.loc[metrics.domain.eq(
        "mtb_holdout") & metrics.condition.eq("point")]
    means = primary.groupby(["ride", "model"]).mean(numeric_only=True)
    # A practical consistency screen, not a statistical significance test.
    for candidate, other in (("B", "A"), ("A", "B")):
        passes = []
        for ride in MTB:
            chosen, alternative, linear = means.loc[(ride, candidate)], means.loc[(
                ride, other)], means.loc[(ride, "linear")]
            passes += [chosen.mae < linear.mae, chosen.rmse < linear.rmse,
                       chosen.mae <= alternative.mae, chosen.rmse <= alternative.rmse]
            for metric in ("high_mae", "rise_mae", "recovery_mae", "stationary_mae", "restart_mae"):
                if np.isfinite(alternative[metric]):
                    passes.append(chosen[metric] <= alternative[metric] + .5)
            passes += [chosen.peak_amplitude_mae <= alternative.peak_amplitude_mae + 1,
                       chosen.peak_timing_mae_s <= alternative.peak_timing_mae_s + 2]
        sensitivity = metrics.loc[metrics.domain.eq("mtb_holdout") & metrics.condition.eq(
            "averaged")].groupby(["ride", "model"]).mean(numeric_only=True)
        passes += [(sensitivity.loc[(ride, candidate), ["mae", "rmse"]] <=
                    sensitivity.loc[(ride, other), ["mae", "rmse"]] + .1).all() for ride in MTB]
        if all(bool(value) for value in passes):
            return candidate
    return "C"


def report(metrics, inventory, packages, samples, partitions, candidate_summary,
           difference, checks, protected_n):
    """Keep individual rides, phases, sensor differences and uncertainty visible."""
    letter = recommend(metrics)
    choices = {"A": "MTB-only training.", "B": "Combined road + MTB training.",
               "C": "Retain the existing Window v2 because neither new configuration is sufficiently convincing."}
    lines = ["# MTB-only versus mixed training for Window v2", "",
             "**Recommendation: " + letter + ". " +
             choices[letter] +
             "** No automatic replacement; both new files are unapproved candidates.", "",
             "## Design and reference quality", "",
             "Exactly six HGB fits: two leave-one-MTB-out folds for each configuration, followed by final A on both MTBs and final B on road_ride_1 + road_ride_2 + both MTBs. No road holdouts, other road files, virtual data, search, weighting, balancing or extra refits. A and B share all settings and examples from their common training rides. Only training membership changes.", ""]
    lines += markdown_table(["MTB reference", "Local HR start/end", "Span min", "Valid HR / elapsed %", "Gaps / longest continuous s",
                            "Speed/distance/altitude valid", "Stationary / probable-stop rows", "Altitude steps >2m", "Other sensors"], inventory)
    lines += ["Both FITs pass integrity/decoding, unique timezone-aware UTC timestamps and HR validity. Both supply uninterrupted 1Hz references, including stationary intervals. MTB 1 records Edge 840 + OHR metadata; MTB 2 records Edge 840 + HRM 200. This difference in reference sensors can confound transfer; neither is independent physiological ground truth. Temperature is available; power/cadence are absent. No new sensor enters prediction. The first ride has one absent speed value and the second has fourteen, left missing.", "",
              "Reuse `prepare`, native elevation, stop classification and `source_features`; no separate elevation investigation. Existing quality masks exclude unsupported centered climbing rates at stops, restarts, recording gaps and altitude jumps. MTB 2's five >2m adjacent altitude steps justify retaining masks; no steps exceed this threshold in MTB 1, which does not prove its altitude is accurate. Target MTB uses the existing aligned Strava terrain. Display Europe/Brussels; all computations UTC.", "",
              "## Fixed model and sampling", "",
              "Exactly the current 50 ordered predictors: 20 anchor/movement/terrain plus 30 temporal summaries. Regions [-60,-30), [-30,0), [0,30), [30,60); missing means remain NaN with explicit coverage. Existing speed/centered terrain preprocessing retains its own support. Future non-HR features are permitted offline; no hidden HR, ride ID, power/cadence or coordinates enter predictors. Keep linear residual targets, five-sample residual smoothing, 4f(1−f) taper and missing-feature handling. Fixed HGB parameters: `" + str(
                  base.PARAMETERS) + "`.", "",
              "Replay the October 16 watch's actual ordered 142×60s + 15×120s intervals (158 observations over 172 minutes), cyclically extending or truncating by elapsed UTC time at phases 0/20/40s. Point anchors equal the exact recorded HR. Average anchors require all 60 samples [t−30,t+29]; this is a sensitivity assumption, not verified Garmin behavior. Train on point anchors only, with the same deterministic five-second timestamp selection and cyclic phase assignment, excluding that phase's anchors. No repeated targets. Maximum interpolation gap 180s, no extrapolation, no anchors from missing HR.", ""]
    training_rows, anchor_rows = [], []
    for name, packs in packages.items():
        x = samples[name][0]
        complete = x[[label + "_record_coverage" for label,
                      _, _ in window.REGIONS]].eq(1).all(axis=1)
        terrain = x[[label + "_terrain_coverage" for label,
                     _, _ in window.REGIONS]].eq(1).all(axis=1)
        training_rows.append([name, len(x), f"{100*x.isna().mean().mean():.2f}", f"{100*complete.mean():.2f}",
                             f"{100*terrain.mean():.2f}", f"{100*x.climb_up.notna().mean():.2f}"])
        for phase in base.PHASES:
            anchor_rows.append([name, phase, len(packs[phase, "point"]["anchors"]), len(
                packs[phase, "averaged"]["anchors"])])
    lines += markdown_table(["Activity", "Eligible training rows", "Missing feature cells %",
                            "All record windows complete %", "All terrain windows complete %", "Current terrain valid %"], training_rows)
    lines += markdown_table(["Activity", "Phase s",
                            "Point anchors", "Average anchors"], anchor_rows)
    lines += markdown_table(["Config", "Training partition",
                            "Total examples", "Road %"], partitions)
    lines += ["Coverage above is on selected training examples, not all elapsed seconds. A complete terrain window requires every rate in all four regions. Stop-related missing terrain remains missing; it is not fabricated as flat terrain. Larger rides contribute more under the unchanged unweighted policy; the road proportions above quantify this rather than silently balancing activities.", "",
              "## Held-out MTB results", "",
              "linear is a fixed interpolation reference, not a third fitted training experiment. Within each ride/phase all three methods AND both anchor conditions score exactly the same hidden timestamps, excluding anchors in either condition. Coverage denominator is all valid HR; `supported_n` in the CSV additionally describes each full curve before common-mask scoring. Bias equals mean-HR error on matching timestamps. Independent units are two rides, not seconds or phases.", ""]
    held = metrics.loc[metrics.domain.eq("mtb_holdout")]
    rows = []
    for (ride, condition, phase), group in held.groupby(["ride", "condition", "phase"]):
        g = group.set_index("model")
        rows.append([ride, condition, phase] + [f"{g.loc[n,'mae']:.3f}/{g.loc[n,'rmse']:.3f}" for n in METHODS] + [
                    f"{g.loc['A','bias']:+.3f}/{g.loc['B','bias']:+.3f}", f"{int(g.loc['A','n'])} ({g.loc['A','coverage_percent']:.2f}%)"])
    lines += markdown_table(["Held out", "Anchors", "Phase", "Linear MAE/RMSE",
                            "A MAE/RMSE", "B MAE/RMSE", "A/B bias", "Common n (%)"], rows)
    rows = []
    for (ride, condition, model), group in held.groupby(["ride", "condition", "model"]):
        rows.append([ride, condition, model, f"{group.mae.mean():.3f} [{group.mae.min():.3f}, {group.mae.max():.3f}]",
                    f"{group.rmse.mean():.3f} [{group.rmse.min():.3f}, {group.rmse.max():.3f}]"])
    lines += markdown_table(["Ride", "Anchors", "Model",
                            "Mean MAE [range]", "Mean RMSE [range]"], rows)
    lines += ["### Dynamics, stops and continuity", "",
              "Reuse established score-only masks: high HR = reference upper decile; rise/recovery = ≥5/≤−5 BPM over 30s in a continuous reference block. Reference peaks: prominence ≥10 BPM, spacing ≥120s. Amplitude/timing use a supported ±15s local maximum; timing is bounded by that window, not unrestricted peak alignment. Unmatched predicted peaks lack a reference peak within ±30s. Roughness uses consecutive supported seconds including anchors. Stationary and ≤30s restart errors below use recorded reference HR only. CSV contains every phase/scenario metric and subset count.", ""]
    primary = held.loc[held.condition.eq("point")]
    rows = []
    for (ride, model), group in primary.groupby(["ride", "model"]):
        rows.append([ride, model, f"{group.high_mae.mean():.2f}", f"{group.rise_mae.mean():.2f}/{group.recovery_mae.mean():.2f}", f"{group.peak_amplitude_mae.mean():.2f}/{group.peak_timing_mae_s.mean():.2f}",
                    f"{group.stationary_mae.mean():.2f}/{group.restart_mae.mean():.2f}", f"{group.unmatched_peaks.mean():.1f}", f"{group.step_p99.mean():.3f}/{group.max_step.max():.3f}", f"{group.steps_gt5_percent.max():.3f}"])
    lines += markdown_table(["Ride", "Model", "High MAE", "Rise/recovery MAE", "Peak amp/timing",
                            "Stationary/restart MAE", "Unmatched peaks mean", "Step p99 mean/max", ">5 BPM/s % max"], rows)
    for ride in MTB:
        means = primary.loc[primary.ride.eq(ride)].groupby(
            "model").mean(numeric_only=True)
        a, b = means.loc["A"], means.loc["B"]
        lines += [f"{ride}: adding road examples changes phase-mean MAE/RMSE by {b.mae-a.mae:+.3f}/{b.rmse-a.rmse:+.3f} BPM, peak-amplitude MAE by {b.peak_amplitude_mae-a.peak_amplitude_mae:+.3f} BPM, rise/recovery MAE by {b.rise_mae-a.rise_mae:+.3f}/{b.recovery_mae-a.recovery_mae:+.3f} BPM and stationary/restart MAE by {b.stationary_mae-a.stationary_mae:+.3f}/{b.restart_mae-a.restart_mae:+.3f} BPM.", ""]
    lines += ["## October 16 target diagnostic and candidates", "",
              "Frozen final fits use real wellness anchors and quality-masked Strava terrain. The Edge overlap diagnostic uses only its 21 actual watch anchors, never withheld Edge values. This matches the prior benchmark's overlap protocol; it is secondary and has been examined repeatedly. Synthetic point anchors do not reproduce the real watch/Edge observation mismatch.", ""]
    rows = []
    for _, r in metrics.loc[metrics.domain.eq("mtb_overlap")].iterrows():
        rows.append([r.model, f"{r.mae:.3f}/{r.rmse:.3f}", f"{r.bias:+.3f}", f"{r.high_mae:.2f}", f"{r.recovery_mae:.2f}", f"{r.peak_amplitude_mae:.2f}/{r.peak_timing_mae_s:.2f}",
                    f"{r.stationary_mae:.2f}/{r.restart_mae:.2f}", f"{int(r.n)} ({r.coverage_percent:.2f}%)", f"{r.step_p99:.3f}/{r.max_step:.3f}"])
    lines += markdown_table(["Model", "MAE/RMSE", "Bias", "High MAE", "Recovery MAE",
                            "Peak amp/timing", "Stationary/restart MAE", "Scored n (%)", "Step p99/max"], rows)
    old = pd.read_csv(ROOT / "data/processed/hr_ml_window_metrics.csv")
    prior = old.loc[old.model.eq(
        "Window") & old.domain.eq("mtb_overlap")].iloc[0]
    lines += [f"Stored road-only Window v2 context: outdoor road 1 MAE/RMSE 3.50/5.14, road 2 3.34/4.97 BPM; October 16 overlap {prior.mae:.3f}/{prior.rmse:.3f} BPM. These are different holdouts, not evidence of how that model would score on the new MTB rides; it is not refitted or scored here.", "",
              "Candidate construction follows the existing recorded/estimated handoff policy: preserve every Edge observation, use the last recorded Edge value solely as the observed boundary for subsequent reconstruction, then use actual later watch anchors. This boundary is never supplied to the withheld-overlap evaluation. Preserve all 158 wellness summaries separately in `watch_hr`; within the Edge overlap they need not equal instantaneous Edge HR, so `final_hr` retains Edge priority. No unsupported trailing timestamps are filled. Future watch anchors are preserved exactly.", ""]
    lines += markdown_table(["Candidate", "Supported n", "Coverage %", "Mean HR", "Max HR",
                            "Estimated step p99/max", "Full-series max step", "Unsupported n"], candidate_summary)
    lines += [f"On estimated seconds only, absolute A/B difference: mean {difference.mean():.3f}, 95th percentile {difference.quantile(.95):.3f}, maximum {difference.max():.3f} BPM. Both preserve the same recorded observations and support. A different mean or smoother curve does not establish accuracy outside the measured overlap. Full-series jumps include original Edge changes; estimated-step statistics require both adjacent seconds to be estimated.", "",
              "## Decision, limitations and reproduction", "",
              "Conservative decision rule fixed before fitting: prefer a configuration only if it improves over linear on both rides and has nonworsening mean point MAE/RMSE versus the other configuration on both rides. Allow at most +0.5 BPM high/rise/recovery/stationary/restart MAE, +1 BPM peak-amplitude MAE and +2s timing. Averaged-anchor mean MAE/RMSE may worsen at most 0.1 BPM. Otherwise choose C. These are descriptive practical screens, not significance tests; settings are never tuned against them, and target overlap does not select the winner.", "",
              "Only two held-out MTB rides, from different sensor paths and seasons, are available. Phases and 1Hz records are correlated; held optical values and sensor artifacts remain possible. Native training altitude and Strava target elevation have different error structures. Speed/terrain do not establish pedaling effort, especially during technical riding and stops without power/cadence. Recommendation remains exploratory, and no candidate automatically replaces Window v2 or v1.", "",
              "![MTB comparison](images/hr_mtb_training_comparison.png)", "",
              "Figure uses phase 0 and the central 20 minutes of the longest continuous reference block, selected without inspecting model errors. Bottom bars show means and phase ranges, not confidence intervals.", "",
              "```powershell", ".venv\\Scripts\\python.exe scripts/evaluate_hr_mtb_training.py", "```", "",
              "Outputs: this report, `docs/images/hr_mtb_training_comparison.png`, `data/processed/hr_mtb_training_metrics.csv`, `data/processed/hr_mtb_only_candidate_2026-10-16.csv`, `data/processed/hr_mixed_candidate_2026-10-16.csv`; script `scripts/evaluate_hr_mtb_training.py`.", ""]
    lines += ["- " + item for item in checks]
    lines += [f"- SHA-256 preservation check passes for {protected_n} pre-existing files; no prior model/report/reconstruction or raw data overwritten.",
              "- Exactly six requested fits; no separate suite or dependencies. Actual script checks output sizes; figure readability is inspected after generation.", "", letter + ". " + choices[letter], ""]
    (ROOT / "docs" / (STEM + ".md")).write_text("\n".join(lines), encoding="utf-8")
    return letter


def main():
    """Validate reference suitability first, then run the six necessary fits once."""
    output_paths = [ROOT / "scripts/evaluate_hr_mtb_training.py", ROOT / "docs" / (STEM + ".md"),
                    ROOT / "docs/images" /
                    (STEM + ".png"), ROOT /
                    "data/processed/hr_mtb_training_metrics.csv",
                    ROOT / "data/processed/hr_mtb_only_candidate_2026-10-16.csv", ROOT / "data/processed/hr_mixed_candidate_2026-10-16.csv"]
    protected = [p for folder in ("data", "docs", "scripts") for p in (ROOT / folder).rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and p not in output_paths]
    before = base.fingerprints(protected)
    rides, inventory = {}, []
    # Both references must pass before any fitting, including the first fold.
    for name in MTB:
        raw, messages = load_records(name)
        ride = window.prepare(name)
        rides[name] = ride
        reference, frame = ride["reference"], ride["frame"]
        span = (reference.index[-1] - reference.index[0]).total_seconds()
        gaps = np.diff(reference.index.asi8) / 1e9
        longest = max(map(len, base.continuous_blocks(reference)))
        if len(reference) / (span + 1) < .8 or longest < 1201:
            raise ValueError(
                name + ": inadequate reference coverage/continuous HR; no fits run")
        fields = "/".join(str(raw[c].count())
                          for c in ("speed_raw", "distance", "altitude_raw"))
        additional = ", ".join(c for c in (
            "temperature", "power", "cadence") if c in raw)
        inventory.append([name, reference.index[0].tz_convert(ZONE).isoformat() + " / " + reference.index[-1].tz_convert(ZONE).isoformat(),
                          f"{span/60:.2f}", f"{len(reference)} / {100*len(reference)/(span+1):.2f}",
                          f"{int((gaps>1).sum())} / {longest}", fields,
                          f"{int(frame.stationary_evidence.sum())}/{int(frame.probable_stop.sum())}",
                          int((raw.altitude_raw.diff().abs().gt(2) & raw.dt.eq(1)).sum()), additional])
        assert messages.get("record_mesgs") and reference.index.tz is not None
        print("Suitable reference:", name, len(reference),
              "HR rows; longest block", longest, flush=True)
    for name in ROADS:
        rides[name] = window.prepare(name)
    watch, _, _ = window.clean_observations(pd.read_csv(
        ROOT / "data/processed/watch_hr_ride_2026-10-16.csv"))
    pattern = np.diff(watch.index.asi8) // 1_000_000_000
    assert len(watch) == 158 and (pattern == 60).sum(
    ) == 142 and (pattern == 120).sum() == 15
    window.verify_windows()
    packages, samples = {}, {}
    for name, ride in rides.items():
        frame = ride["frame"]
        assert "heart_rate" not in frame
        packages[name] = window.add_windows(base.synthetic(
            ride, pattern, base.activity_features(frame)), window.window_features(frame))
        for package in packages[name].values():
            assert list(package["x"].columns) == COLUMNS and len(COLUMNS) == 50
            for left, right in package["spans"]:
                assert (right - left).total_seconds() <= 180
                assert np.all(
                    np.diff(frame.loc[left:right].index.asi8) == 1_000_000_000)
        samples[name] = window.training_rows([(ride, packages[name])])
        assert len(samples[name][0]) >= 200
    checks = ["Both MTB FITs pass decoding/integrity and continuous-reference gate before fitting; native missing measurements remain missing.",
              "Reused exact window-boundary, gap, stop and missing-terrain checks; 50 ordered features unchanged, no duplicate targets, complete point/averaged anchors and <=180s gap-isolated interpolation verified."]
    rows, panels, partitions = [], [], []
    print("Training examples:", {n: len(x)
          for n, (x, _) in samples.items()}, flush=True)
    for held, training in ((MTB[0], MTB[1]), (MTB[1], MTB[0])):
        print("Two fits; holding out", held, flush=True)
        fitted = fit_models((training,), samples, partitions)
        values, cached = evaluate(
            rides[held], packages[held], fitted, "mtb_holdout", checks)
        rows.extend(values)
        panels.append((rides[held], cached))
    print("Two final fits; target diagnostic and separate candidates", flush=True)
    fitted = fit_models(MTB, samples, partitions)
    mtb = window.prepare("mtb_edge")
    mtb["frame"] = mtb["strava"]
    windows = window.window_features(mtb["frame"])
    overlap = watch.loc[mtb["reference"].index[0]:mtb["reference"].index[-1]]
    original = base.examples(
        mtb["frame"], overlap, mtb["reference"].index, base.activity_features(mtb["frame"]))
    overlap_packages = window.add_windows(
        {(0, "actual_watch"): original}, windows)
    values, _ = evaluate(mtb, overlap_packages, fitted, "mtb_overlap", checks)
    rows.extend(values)
    candidate_summary, difference = candidates(mtb, watch, fitted, windows)
    checks.append("Both candidate CSVs read back: original Edge and separate watch HR exact, future watch anchors exact, matching support, unsupported tail missing, valid numeric HR.")
    metrics = pd.DataFrame(rows)
    assert len(metrics) == 39 and len(partitions) == 6
    assert metrics.groupby(["domain", "ride", "phase"]).n.nunique().eq(1).all()
    assert np.isfinite(metrics[["mae", "rmse", "bias"]]).all().all()
    metrics.to_csv(output_paths[3], index=False)
    plot_comparison(panels, metrics, output_paths[2])
    assert base.fingerprints(protected) == before
    letter = report(metrics, inventory, packages, samples, partitions, candidate_summary,
                    difference, checks, len(protected))
    assert all(p.stat().st_size > 0 for p in output_paths)
    assert base.fingerprints(protected) == before
    print(metrics.loc[metrics.condition.isin(["point", "actual_watch"])].groupby(["domain", "ride", "model"])[
          ["mae", "rmse", "bias", "recovery_mae", "stationary_mae", "restart_mae"]].mean().round(4).to_string())
    print("Candidates:", candidate_summary,
          "estimated mean/max difference", difference.mean(), difference.max())
    print("Recommendation:", letter,
          "; six fits; protected files unchanged:", len(protected))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
