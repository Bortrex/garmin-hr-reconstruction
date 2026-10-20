"""Compare fixed Window v2 fits with and without virtual training examples.

Run: .venv/Scripts/python scripts/evaluate_hr_virtual_augmentation.py
Existing models, observations and reconstructions are never overwritten.
"""

import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import evaluate_hr_ml_windows as window
from diagnose_watch_anchors import decode

base = window.base
ROOT, ZONE = base.ROOT, base.ZONE
STEM = "hr_virtual_augmentation"
NAMES = {"A": "Outdoor only", "B": "Outdoor + virtual"}
COLORS = {"A": "#257c9b", "B": "#c35b2d"}
COLUMNS = window.BASE_COLUMNS + window.WINDOW_COLUMNS


def fit_configs(training, samples, repeat=False):
    """Change only training membership; give every example equal weight."""
    models, partitions = {}, {}
    for name, members in (("A", training),
                          ("B", training + ("virtual_road_ride_1",))):
        x = pd.concat([samples[ride][0] for ride in members])
        target = pd.concat([samples[ride][1] for ride in members])
        assert x.index.is_unique and x.index.equals(target.index)
        assert list(x.columns) == COLUMNS and len(COLUMNS) == 50
        model = HistGradientBoostingRegressor(**base.PARAMETERS).fit(x, target)
        assert model.n_iter_ == 120 and not model.do_early_stopping_
        if repeat and name == "B":
            second = HistGradientBoostingRegressor(
                **base.PARAMETERS).fit(x, target)
            assert np.array_equal(model.predict(x), second.predict(x))
        models[name], partitions[name] = model, members
    return {"models": models, "partitions": partitions}


def evaluate(ride, packages, fitted, domain, checks):
    """Use existing metrics on identical hidden seconds across configurations."""
    for members in fitted["partitions"].values():
        assert ride["name"] not in members
    masks, peaks, _ = window.diagnostic_masks(ride)
    rows, cached = [], {}
    phases = base.PHASES if domain == "outdoor" else (0,)
    conditions = ("point", "averaged") if domain == "outdoor" else (
        "actual_watch",)
    for phase in phases:
        results = {condition: base.predictions(packages[phase, condition], fitted)
                   for condition in conditions}
        support = ride["reference"].reindex(ride["frame"].index).notna()
        for result in results.values():
            support &= result[["A", "B", "linear"]].notna().all(axis=1)
            assert result.A.notna().equals(result.B.notna())
        common = support.copy()
        for condition in conditions:
            common.loc[packages[phase, condition]["anchors"].index] = False
        for condition, result in results.items():
            package = packages[phase, condition]
            for name in NAMES:
                prediction = result[name]
                metrics = window.score(ride, prediction, common, masks, peaks)
                metrics["average_hr_error"] = metrics["bias"]
                error = (prediction -
                         ride["reference"].reindex(result.index)).where(common)
                for label, mask in (
                        ("stationary", ride["frame"].stationary_evidence),
                        ("restart", package["x"].restart_age_s.le(30)),
                        ("uphill", ride["frame"].climb30.gt(.05)),
                        ("downhill", ride["frame"].climb30.lt(-.05))):
                    selected = error[mask].dropna()
                    metrics[label + "_n"] = len(selected)
                    metrics[label + "_mae"] = selected.abs().mean()
                metrics.update(base.extra_metrics(prediction, result.linear,
                                                  ride["reference"], support, peaks))
                finite = prediction.dropna()
                assert finite.gt(0).all() and finite.lt(255).all()
                rows.append({"domain": domain, "ride": ride["name"], "phase": phase,
                             "condition": condition, "model": name,
                             "training": "+".join(fitted["partitions"][name]),
                             **metrics, "prediction_min": finite.min(),
                             "prediction_max": finite.max(),
                             "supported_n": len(finite),
                             "supported_percent": 100 * len(finite) / len(result)})
            cached[phase, condition] = (package, result, common)
    package, expected, _ = next(iter(cached.values()))
    hidden_replaced = ride["reference"] * 0 + 200
    rebuilt = base.examples(ride["frame"], package["anchors"], hidden_replaced.index,
                            base.activity_features(ride["frame"]))
    rebuilt["x"] = rebuilt["x"].join(window.window_features(ride["frame"]))
    pd.testing.assert_frame_equal(rebuilt["x"], package["x"])
    pd.testing.assert_frame_equal(base.predictions(rebuilt, fitted), expected)
    checks.append(
        ride["name"] + ": held out entirely; hidden-HR substitution leaves inputs/predictions identical; exact anchors, matching support, finite valid BPM")
    return rows, cached


def check_benchmark(metrics):
    """Reproduce saved Window v2 numeric results without changing the benchmark."""
    old = pd.read_csv(ROOT / "data/processed/hr_ml_window_metrics.csv")
    keys = ["domain", "ride", "phase", "condition"]
    current = metrics.loc[metrics.model.eq("A")].set_index(keys)
    previous = old.loc[old.model.eq("Window")].set_index(
        keys).loc[current.index]
    columns = previous.select_dtypes(
        include=[np.number]).columns.intersection(current.columns)
    assert np.allclose(current[columns], previous[columns],
                       atol=1e-10, rtol=1e-10, equal_nan=True)
    return len(current)


def decision(metrics):
    """Apply a fixed descriptive screen, never optimize settings on holdouts.

    Prefer B only for >=0.1 BPM mean MAE improvement on both roads,
    nonworsening RMSE in every point phase and both averaged means,
    and no material dynamics/MTB regression. These are practical screens,
    not confidence intervals or a statistical claim of generalization.
    """
    tests, worse = [], []
    for ride in base.OUTDOOR:
        data = metrics.loc[metrics.ride.eq(ride)]
        point = data.loc[data.condition.eq("point")]
        a = point.loc[point.model.eq("A")].set_index("phase")
        b = point.loc[point.model.eq("B")].set_index("phase")
        tests += [(ride + ": mean MAE improvement >=0.1 BPM", a.mae.mean() - b.mae.mean() >= .1),
                  (ride + ": every point phase MAE/RMSE nonworsening", ((b[["mae", "rmse"]] <= a[["mae", "rmse"]]).all().all()))]
        worse.append(b.mae.mean() > a.mae.mean()
                     and b.rmse.mean() > a.rmse.mean())
        averaged = data.loc[data.condition.eq("averaged")].groupby(
            "model").mean(numeric_only=True)
        tests.append((ride + ": averaged-anchor mean MAE/RMSE nonworsening",
                      (averaged.loc["B", ["mae", "rmse"]] <= averaged.loc["A", ["mae", "rmse"]]).all()))
        for metric, tolerance in (("high_mae", .5), ("rise_mae", .5),
                                  ("recovery_mae", .5), ("peak_amplitude_mae", 1),
                                  ("peak_timing_mae_s", 2), ("unmatched_peaks", 2)):
            tests.append((ride + ": " + metric + " no material regression",
                          b[metric].mean() <= a[metric].mean() + tolerance))
    mtb = metrics.loc[metrics.domain.eq("mtb_overlap")].set_index("model")
    for metric in ("rmse", "high_mae", "stationary_mae", "restart_mae"):
        tests.append(("MTB " + metric + " no >0.5 BPM regression",
                      mtb.loc["B", metric] <= mtb.loc["A", metric] + .5))
    tests.append(("No >5 BPM/s jumps or invalid numeric HR",
                 metrics.steps_gt5_percent.eq(0).all() and metrics.invalid_hr_n.eq(0).all()))
    letter = "B" if all(bool(v)
                        for _, v in tests) else ("A" if all(worse) else "C")
    return letter, tests


def plot_comparison(panels, metrics, path):
    """Keep v2's central 20 minutes of the longest continuous reference block."""
    plt, mdates = window.plt, window.mdates
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), constrained_layout=True)
    for ax, (ride, cached) in zip(axes[:2], panels):
        package, predicted, _ = cached[0, "point"]
        block = max(base.continuous_blocks(ride["reference"]), key=len)
        midpoint = block.index[0] + (block.index[-1] - block.index[0]) / 2
        start = (midpoint - pd.Timedelta(minutes=10)).floor("s")
        end = start + pd.Timedelta(minutes=20)
        actual = ride["reference"].loc[start:end]
        ax.plot(actual.index, actual, color=".65", lw=1.3,
                label="Recorded hidden HR reference")
        for name in NAMES:
            part = predicted.loc[start:end, name]
            ax.plot(part.index, part, color=COLORS[name], lw=1.4,
                    ls="--" if name == "B" else "-", label=NAMES[name] + " estimate")
        sparse = package["anchors"].loc[start:end]
        ax.scatter(sparse.index, sparse, s=24, color=".15", zorder=5,
                   label="Simulated sparse point observations")
        ax.set(title=ride["name"] + " — held out, phase 0; fixed central 20-minute segment",
               xlabel="Local time — Europe/Brussels", ylabel="HR (BPM)", xlim=(start, end))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZONE))
        ax.legend(ncol=2, fontsize=9)
        ax.grid(alpha=.2)
    primary = metrics.loc[metrics.domain.eq(
        "outdoor") & metrics.condition.eq("point")]
    for i, name in enumerate(NAMES):
        groups = [primary.loc[primary.ride.eq(
            ride) & primary.model.eq(name)] for ride in base.OUTDOOR]
        means = np.array([group.mae.mean() for group in groups])
        low = np.array([group.mae.min() for group in groups])
        high = np.array([group.mae.max() for group in groups])
        axes[2].bar(np.arange(2) + (i - .5) * .3, means, .28, color=COLORS[name],
                    label=NAMES[name], yerr=[means - low, high - means], capsize=4)
    axes[2].set(xticks=[0, 1], xticklabels=list(base.OUTDOOR), ylabel="MAE (BPM)",
                title="Outdoor holdouts: phase mean and min–max (0 / 20 / 40 seconds)")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=.2)
    fig.suptitle(
        "Window v2: adding virtual training data only\nFixed 50 predictors, model settings and residual postprocessing", fontsize=15)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def table(headers, rows):
    """Render small report tables without an additional dependency."""
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + [
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows] + [""]


def write_report(metrics, rides, packages, samples, checks, protected, letter,
                 tests, full_target):
    """Report individual folds, dependent phase variation and domain limitations."""
    recommendations = {"A": "Use outdoor-only Window v2.",
                       "B": "Use Window v2 trained with outdoor + virtual data.",
                       "C": "Retain the outdoor-only model because the evidence is mixed."}
    lines = ["# Virtual training augmentation for Window v2", "",
             "**Recommendation: " + letter + ". " +
             recommendations[letter] + "**", "",
             "Only training membership changes. A uses the other outdoor ride; B adds all eligible MyWhoosh rows with equal per-example weight. No oversampling, weights, optional virtual validation, parameter search or promotion. The virtual ride trains B and is not an independent validation set. Existing v1 and Window v2 outputs remain intact; no new full-MTB reconstruction is generated.", "",
             "## Data and provenance", "",
             "Sources: `data/road_ride_1.fit`, `data/road_ride_2.fit`, `data/virtual_road_ride_1.fit`; target `data/mtb_edge.fit`, quality-masked `data/VTT_Apero.gpx` and `data/processed/watch_hr_ride_2026-10-16.csv`. FIT integrity/decoder checks run. UTC internally, Europe/Brussels below and in the figure.", "",
             "The virtual FIT has MyWhoosh developer application ID `mywhoosh.whooshg` and a virtual-cycling session despite an Edge 840 product header; no device-info messages establish the optical sensor. FR265 optical HR over Bluetooth is rider-confirmed provenance, not inferred from that header. Power/cadence exist but never enter features. Virtual speed, distance, position and elevation describe the platform, not physical outdoor motion. Outdoor Edge metadata identifies ANT+ OHR without establishing physiological accuracy.", ""]
    rows = []
    for ride in rides:
        ref = ride["reference"]
        gaps = np.diff(ref.index.asi8) / 1e9
        rows.append([ride["name"], ref.index[0].tz_convert(ZONE).isoformat(),
                     ref.index[-1].tz_convert(ZONE).isoformat(), len(ref),
                     f"{(ref.index[-1]-ref.index[0]).total_seconds()/60:.2f}",
                     int((gaps > 1).sum()), len(samples[ride["name"]][0])])
    lines += table(["Ride", "Local start", "Local end", "HR rows",
                   "Span min", "Gaps >1s", "Training rows"], rows)
    rows = []
    for training in (("road_ride_2",), ("road_ride_1",), base.OUTDOOR):
        outdoor_n = sum(len(samples[name][0]) for name in training)
        virtual_n = len(samples["virtual_road_ride_1"][0])
        rows.append([" + ".join(training), outdoor_n, outdoor_n + virtual_n,
                     f"{100*virtual_n/(outdoor_n+virtual_n):.2f}"])
    lines += table(["Outdoor training partition", "A n",
                   "B n", "Virtual % of B"], rows)
    lines += ["All per-ride training timestamps are unique; one sample every five elapsed seconds is assigned cyclically to exactly one phase, excluding that phase's anchors and unsupported baseline seconds. No virtual rows are replicated. HR residual targets use training HR only.", "",
              "## Unchanged protocol and inputs", "",
              "Replay the actual October 16 sequence (142×60s and 15×120s, 158 observations over 172min), cycling for longer rides and truncating shorter ones, at phases 0/20/40s. Point anchors equal the recorded value exactly. Averaged anchors require all 60 observed samples [t−30,t+29]; these are an alternative assumption, not established Garmin behavior. Both fits train on point anchors only. No anchor comes from missing HR. All gaps in reference recording split support; maximum anchor gap 180s; no extrapolation.", "",
              "Keep exactly the same 50 ordered predictors: 20 anchor/movement/terrain plus 30 window summaries, including coverage/missingness. Four half-open regions [-60,-30), [-30,0), [0,30), [30,60); no hidden HR, identity, power, cadence or coordinates as predictors. Future non-HR measurements are permitted offline; existing centered terrain/speed preprocessing retains its own support. Incomplete means remain NaN and coverage remains explicit; HGB handles missing features. No smoothing, masks, feature definition or stopping rule was changed.", "",
              "Fixed HistGradientBoostingRegressor parameters: `" + str(base.PARAMETERS) + "`. Same linear residual baseline, centered five-sample residual smoothing and 4f(1−f) taper to preserve anchors. No model tuning, early stopping split or scaling. Python " + sys.version.split()[0] + ", scikit-learn " + sklearn.__version__ + ".", ""]
    rows = []
    for ride, packs in zip(rides, packages):
        for phase in base.PHASES:
            rows.append([ride["name"], phase, len(packs[phase, "point"]["anchors"]),
                         len(packs[phase, "averaged"]["anchors"])])
    lines += table(["Ride", "Phase s", "Valid point anchors",
                   "Valid average anchors"], rows)
    lines += ["### Feature coverage and domain shift", "",
              "Coverage below is on selected training rows. Complete windows means all four 30s record regions are present; complete window terrain means all four contain 30 valid climbing rates. A gap/stop can therefore reduce terrain support without discarding the training example.", ""]
    rows = []
    for ride in rides:
        x = samples[ride["name"]][0]
        record = x[[label + "_record_coverage" for label,
                    _, _ in window.REGIONS]].eq(1).all(axis=1)
        terrain = x[[label + "_terrain_coverage" for label,
                     _, _ in window.REGIONS]].eq(1).all(axis=1)
        rows.append([ride["name"], f"{100*record.mean():.2f}", f"{100*terrain.mean():.2f}",
                     f"{100*x.climb_up.notna().mean():.2f}", f"{100*x.isna().mean().mean():.2f}"])
    lines += table(["Ride", "Complete record windows %", "Complete terrain windows %",
                   "Current terrain valid %", "Missing feature cells %"], rows)
    rows = []
    for ride in rides:
        frame = ride["frame"]
        speed = frame.speed5.quantile([.05, .5, .95])
        climb = frame.climb30.quantile([.05, .5, .95])
        rows.append([ride["name"], "/".join(f"{v:.3f}" for v in speed),
                     "/".join(f"{v:.3f}" for v in climb),
                     f"{100*frame.stationary_evidence.mean():.2f}",
                     f"{100*frame.speed_raw.gt(.3).mean():.2f}",
                     f"{100*frame.climb30.notna().mean():.2f}"])
    lines += table(["Ride", "Speed5 p5/50/95 m/s", "Climb30 p5/50/95 m/s",
                   "Stationary %", "Moving >0.3m/s %", "Full-record terrain valid %"], rows)
    lines += ["Existing preprocessing does not infer physical stops from virtual coordinates: its virtual stationary flag is zero by design, rather than a measured absence of physical rests. Platform terrain/speed relationships need not transfer to outdoor effort. Missing terrain and window coverage also differ by domain. These non-HR differences can alter fitted splits even without an explicit activity label. HR held values, optical error and different recording paths remain possible; 1Hz records are not independent physiological measurements.", "",
              "## Outdoor results", "",
              "A = outdoor only; B = outdoor + virtual. Errors in BPM, timing in seconds. All configurations AND both observation conditions use the same supported hidden timestamps within each ride/phase, excluding anchors in either condition. Bias equals average-HR error on these same seconds. Coverage denominator is all valid reference HR, not elapsed gaps. Phases are dependent simulations, not six independent rides.", ""]
    outdoor = metrics.loc[metrics.domain.eq("outdoor")]
    rows = []
    for (ride, condition, phase), group in outdoor.groupby(["ride", "condition", "phase"]):
        indexed = group.set_index("model")
        a, b = indexed.loc["A"], indexed.loc["B"]
        rows.append([ride, condition, phase, f"{a.mae:.3f}/{a.rmse:.3f}", f"{b.mae:.3f}/{b.rmse:.3f}",
                     f"{b.mae-a.mae:+.3f}", f"{a.bias:+.3f}/{b.bias:+.3f}",
                     f"{int(a.n)} ({a.coverage_percent:.2f}%)"])
    lines += table(["Held out", "Anchors", "Phase", "A MAE/RMSE",
                   "B MAE/RMSE", "B−A MAE", "A/B bias", "Common n (%)"], rows)
    rows = []
    for (ride, condition, model), group in outdoor.groupby(["ride", "condition", "model"]):
        rows.append([ride, condition, model,
                     f"{group.mae.mean():.3f} [{group.mae.min():.3f}, {group.mae.max():.3f}]",
                     f"{group.rmse.mean():.3f} [{group.rmse.min():.3f}, {group.rmse.max():.3f}]",
                     f"{group.bias.mean():+.3f}"])
    lines += table(["Held out", "Anchors", "Model", "Mean MAE [phase range]",
                   "Mean RMSE [phase range]", "Mean bias"], rows)
    for ride in base.OUTDOOR:
        means = outdoor.loc[outdoor.ride.eq(ride)].groupby(
            ["condition", "model"]).mean(numeric_only=True)
        a, b = means.loc[("point", "A")], means.loc[("point", "B")]
        aa, ab = means.loc[("averaged", "A")], means.loc[("averaged", "B")]
        lines += [f"{ride}: adding virtual training changes mean point MAE by "
                  f"{b.mae-a.mae:+.3f} BPM and RMSE by {b.rmse-a.rmse:+.3f} BPM. "
                  f"Under averaged anchors the changes are {ab.mae-aa.mae:+.3f} "
                  f"and {ab.rmse-aa.rmse:+.3f} BPM respectively.", ""]
    lines += ["Road 1 deteriorates in all three point phases; road 2 has a tiny mean MAE benefit but worse RMSE in every phase. Averaged anchors do not rescue the result: both rides' mean MAE and RMSE increase. The outdoor evidence does not support replacing Window v2.", ""]
    lines += ["### HR fluctuations, peaks and continuity", "",
              "Existing score-only definitions: high HR = reference upper decile; rise/recovery = ≥5/≤−5 BPM change over 30s within a continuous block. True peaks have ≥10 BPM prominence and ≥120s separation. Peak amplitude/timing compare a local maximum within ±15s only where the entire 31s window is evaluated; timing is bounded by this window and is not unrestricted peak matching. Unmatched predicted peaks lack an observed peak within ±30s. Roughness uses consecutive supported seconds including anchors. Full phase/scenario metrics and subset counts are in the CSV.", ""]
    rows = []
    for (ride, model), group in outdoor.loc[outdoor.condition.eq("point")].groupby(["ride", "model"]):
        rows.append([ride, model, f"{group.high_mae.mean():.2f}",
                     f"{group.rise_mae.mean():.2f}/{group.recovery_mae.mean():.2f}",
                     f"{group.peak_amplitude_mae.mean():.2f}/{group.peak_timing_mae_s.mean():.2f}",
                     f"{group.unmatched_peaks.mean():.1f}",
                     f"{group.step_p99.mean():.3f}/{group.max_step.max():.3f}",
                     f"{group.steps_gt5_percent.max():.3f}"])
    lines += table(["Ride", "Model", "High MAE", "Rise/recovery MAE", "Peak amp MAE/timing s",
                   "Unmatched peaks mean", "Step p99 mean/max BPM/s", ">5 BPM/s % max"], rows)
    lines += ["Point-anchor high-HR MAE improves slightly on both rides, but recovery MAE and peak-amplitude MAE worsen on both. Peak timing is nearly unchanged on road 1 and worse on road 2, with more unmatched peaks in phase means. Slightly lower step roughness therefore does not establish better fluctuation recovery. No estimate has a >5 BPM/s adjacent supported step or an invalid numeric HR range.", ""]
    lines += ["## Frozen pooled fits: target MTB overlap", "",
              "A trains both roads, B trains both roads plus MyWhoosh. Apply actual watch anchors within the initial Edge overlap to identical movement/stopping and quality-masked Strava features. No Edge HR enters inputs, no handoff anchor is injected and no settings are tuned from this diagnostic. This overlap has been inspected repeatedly and is not a pristine independent test. Its sparse watch observations may differ from Edge instantaneous HR; this domain mismatch is absent from perfect synthetic point anchors.", ""]
    rows = []
    for _, row in metrics.loc[metrics.domain.eq("mtb_overlap")].iterrows():
        rows.append([row.model, f"{row.mae:.3f}/{row.rmse:.3f}", f"{row.bias:+.3f}",
                     f"{row.high_mae:.2f}", f"{row.peak_amplitude_mae:.2f}/{row.peak_timing_mae_s:.2f}",
                     f"{row.stationary_mae:.2f} (n={int(row.stationary_n)})",
                     f"{row.restart_mae:.2f} (n={int(row.restart_n)})",
                     f"{int(row.n)} ({row.coverage_percent:.2f}%)",
                     f"{row.step_p99:.3f}/{row.max_step:.3f}", int(row.unmatched_peaks)])
    lines += table(["Model", "MAE/RMSE", "Bias", "High MAE", "Peak amp/timing", "Stationary MAE",
                   "Restart ≤30s MAE", "Scored n (%)", "Step p99/max", "Unmatched peaks"], rows)
    lines += [
        "MTB results are also mixed: B's MAE increases by 0.031 BPM while RMSE decreases by only 0.004 BPM. Bias, high-HR error, peak amplitude and restart error improve slightly; stationary error, recovery error (14.27→14.66 BPM) and peak timing worsen. Both configurations support 1,441 overlap timestamps including 21 anchors and score the same 1,420 hidden timestamps. This does not resolve the large watch/Edge disagreement.", ""]
    lines += ["### Full target support and continuity (unscored)", "",
              "Also apply both frozen fits across the entire MTB timeline with all 158 real watch observations. This separate continuity diagnostic supplies no Edge HR or handoff anchor and never merges observations into a final reconstruction. It leaves the first 38s and final 50s unsupported. The overlap scores above retain the original benchmark's 21-anchor protocol; full-ride estimates are not scored or saved as a candidate.", ""]
    lines += table(["Model", "Supported n (%)", "HR min/max BPM", "Step p99/max BPM/s",
                    "Supported stationary/restart n", ">5 BPM/s %"], full_target)
    lines += ["Stationary/restart counts describe available estimates, not their accuracy outside the recorded overlap. Real watch anchors remain exact; no unsupported boundary is filled. A continuous supported curve cannot establish recovery of unobserved physiological fluctuations.", ""]
    lines += ["## Decision and limitations", "",
              "The following descriptive screen was fixed before fitting: require ≥0.1 BPM mean point MAE improvement on both roads, nonworsening point MAE/RMSE in every phase and both averaged-anchor means. Allow at most +0.5 BPM for high/rise/recovery, +1 BPM peak amplitude, +2s peak timing and +2 unmatched peaks in phase means. Require no >0.5 BPM MTB RMSE/high/stationary/restart regression, invalid HR or >5 BPM/s jumps. These are conservative practical checks, not significance tests; no weight/settings are selected against them. If both outdoor mean MAE and RMSE deteriorate choose A; otherwise an incomplete benefit yields C.", ""]
    lines += ["- " + ("PASS: " if passed else "FAIL: ") +
              label for label, passed in tests]
    lines += ["", "Only two outdoor rides and one virtual training activity are available. Correlated seconds/phases do not establish population generalization or MTB physiological accuracy. Average bias can be small while rapid changes/peaks remain wrong. Missing power/cadence and technical MTB motion confound speed/terrain as effort proxies. Neither a smoother curve nor lower exploratory mean error establishes true second-by-second HR. Existing model status is unchanged; replacement needs explicit approval.", "",
              "## Outputs and verification", "",
              "![Fixed outdoor comparison](images/hr_virtual_augmentation.png)", "",
              "Figure: phase-0 central 20min of the longest continuous HR block, identical to v2 selection, never selected by errors. Bottom panel shows mean and phase range, not a confidence interval.", "",
              "Reproduce from repository root:", "", "```powershell",
              ".venv\\Scripts\\python.exe scripts/evaluate_hr_virtual_augmentation.py", "```", "",
              "Outputs: this report, `docs/images/hr_virtual_augmentation.png`, `data/processed/hr_virtual_augmentation_metrics.csv`; implementation `scripts/evaluate_hr_virtual_augmentation.py`.", ""]
    lines += ["- " + item for item in checks]
    lines += [f"- SHA-256 equality checked for all {protected} pre-existing data/report/script files (excluding bytecode and this experiment's outputs); no original or prior output altered.",
              "- Actual execution checks all three generated artifacts are non-empty. Figure readability is inspected after generation; no separate test suite or dependency added.", "",
              letter + ". " + recommendations[letter], ""]
    (ROOT / "docs" / (STEM + ".md")).write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run two outdoor folds, then the frozen MTB diagnostic, and save results."""
    output_names = {
        STEM + suffix for suffix in (".py", ".md", ".png", "_metrics.csv")}
    protected = [p for folder in ("data", "docs", "scripts") for p in (ROOT / folder).rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and p.name not in output_names]
    before = base.fingerprints(protected)
    messages = decode(ROOT / "data/virtual_road_ride_1.fit")
    application = bytes(
        messages["developer_data_id_mesgs"][0]["application_id"]).decode("ascii")
    assert application == "mywhoosh.whooshg"
    assert messages["session_mesgs"][0]["sub_sport"] == "virtual_activity"
    records = messages["record_mesgs"]
    times = pd.DatetimeIndex([r["timestamp"] for r in records])
    assert len(records) == 5115 and times.is_unique and times.tz is not None
    assert np.all(np.diff(times.asi8) == 1_000_000_000)
    assert all("power" in r and "cadence" in r for r in records)
    window.verify_windows()
    checks = ["Virtual FIT integrity/decoding, MyWhoosh developer metadata, virtual session, 5,115 unique UTC 1Hz records verified; power/cadence excluded.",
              "Reused focused window checks pass: exact half-open bounds, future alignment, gap isolation, stop fractions and missing terrain coverage."]
    watch, _, _ = window.clean_observations(pd.read_csv(
        ROOT / "data/processed/watch_hr_ride_2026-10-16.csv"))
    pattern = np.diff(watch.index.asi8) // 1_000_000_000
    assert len(watch) == 158 and (pattern == 60).sum(
    ) == 142 and (pattern == 120).sum() == 15
    rides = [window.prepare(name)
             for name in base.OUTDOOR + ("virtual_road_ride_1",)]
    packages, samples = [], {}
    for ride in rides:
        frame = ride["frame"]
        assert "heart_rate" not in frame and frame.index.tz is not None
        packs = window.add_windows(base.synthetic(ride, pattern, base.activity_features(frame)),
                                   window.window_features(frame))
        for package in packs.values():
            assert list(package["x"].columns) == COLUMNS
            for left, right in package["spans"]:
                times = frame.loc[left:right].index
                assert 0 < (right - left).total_seconds() <= 180
                assert np.all(np.diff(times.asi8) == 1_000_000_000)
        packages.append(packs)
        samples[ride["name"]] = window.training_rows([(ride, packs)])
    print("Training rows:", {name: len(x)
          for name, (x, _) in samples.items()}, flush=True)
    checks.append("Same 50 ordered columns and fixed settings; no HR/identity/power/cadence inputs; no duplicate training targets; exact point/complete averaged anchors and <=180s gap-isolated spans verified.")
    rows, panels = [], []
    for held, train in ((0, 1), (1, 0)):
        print("Holding out", rides[held]["name"], flush=True)
        fitted = fit_configs((rides[train]["name"],),
                             samples, repeat=(held == 0))
        values, cached = evaluate(
            rides[held], packages[held], fitted, "outdoor", checks)
        rows.extend(values)
        panels.append((rides[held], cached))
    checks.append(
        "Repeated first augmented fit produces bit-identical training predictions with fixed random state and two computation threads.")
    print("Frozen pooled MTB diagnostic", flush=True)
    pooled = fit_configs(base.OUTDOOR, samples)
    mtb = window.prepare("mtb_edge")
    mtb["frame"] = mtb["strava"]
    overlap = watch.loc[mtb["reference"].index[0]:mtb["reference"].index[-1]]
    original = base.examples(
        mtb["frame"], overlap, mtb["reference"].index, base.activity_features(mtb["frame"]))
    mtb_packages = window.add_windows(
        {(0, "actual_watch"): original}, window.window_features(mtb["frame"]))
    values, _ = evaluate(mtb, mtb_packages, pooled, "mtb_overlap", checks)
    rows.extend(values)
    # Full-ride continuity uses watch observations only, never the Edge HR.
    full = base.examples(mtb["frame"], watch, mtb["frame"].index,
                         base.activity_features(mtb["frame"]))
    full["x"] = full["x"].join(
        mtb_packages[0, "actual_watch"]["x"][window.WINDOW_COLUMNS])
    curves = base.predictions(full, pooled)
    assert curves.A.notna().equals(curves.B.notna())
    assert curves.loc[curves.index < watch.index[0],
                      ["A", "B"]].isna().all().all()
    assert curves.loc[curves.index > watch.index[-1],
                      ["A", "B"]].isna().all().all()
    full_target = []
    for name in NAMES:
        curve = curves[name]
        valid = curve.notna()
        assert curve[valid].gt(0).all() and curve[valid].lt(255).all()
        step = curve.diff().where(valid & valid.shift(fill_value=False)
                                  & curve.index.to_series().diff().dt.total_seconds().eq(1)).abs().dropna()
        stationary = (valid & mtb["frame"].stationary_evidence).sum()
        restart = (valid & full["x"].restart_age_s.le(30)).sum()
        full_target.append([name, f"{valid.sum()} ({100*valid.mean():.2f}%)",
                            f"{curve.min():.2f}/{curve.max():.2f}",
                            f"{step.quantile(.99):.3f}/{step.max():.3f}",
                            f"{stationary}/{restart}", f"{100*step.gt(5).mean():.3f}"])
    checks.append("Both frozen models applied to full MTB timeline: all 158 watch anchors exact, matching support, valid numeric HR and no extrapolation; Edge HR never supplied as an input. Full curves are unscored and not exported.")
    metrics = pd.DataFrame(rows)
    assert len(metrics) == 26
    assert metrics.groupby(
        ["domain", "ride", "phase", "condition"]).n.nunique().eq(1).all()
    assert metrics.groupby(["domain", "ride", "phase"]).n.nunique().eq(1).all()
    assert np.isfinite(metrics[["mae", "rmse", "bias"]]).all().all()
    count = check_benchmark(metrics)
    checks.append(
        f"All {count} saved Window v2 rows reproduce numeric metrics/support within 1e-10; matching timestamp scoring and bias/mean-HR identity verified.")
    letter, tests = decision(metrics)
    paths = [ROOT / "data/processed" / (STEM + "_metrics.csv"),
             ROOT / "docs/images" / (STEM + ".png"), ROOT / "docs" / (STEM + ".md")]
    metrics.to_csv(paths[0], index=False)
    plot_comparison(panels, metrics, paths[1])
    assert base.fingerprints(protected) == before
    write_report(metrics, rides, packages, samples,
                 checks, len(protected), letter, tests, full_target)
    assert all(path.stat().st_size > 0 for path in paths)
    assert base.fingerprints(protected) == before
    print(metrics.loc[metrics.condition.isin(["point", "actual_watch"])].groupby(
        ["domain", "ride", "model"])[["mae", "rmse", "bias", "high_mae", "recovery_mae"]].mean().round(4).to_string())
    print("Decision:", letter, "; protected files unchanged:", len(protected))
    print("Failed screens:", [label for label, passed in tests if not passed])


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
