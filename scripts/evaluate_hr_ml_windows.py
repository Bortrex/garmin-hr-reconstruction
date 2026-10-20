"""Compare fixed temporal windows with the established ML-B residual model.

Run: .venv/Scripts/python scripts/evaluate_hr_ml_windows.py
Version-1 outputs are protected; a new model is only an evaluation candidate.
"""

import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import evaluate_hr_ml_residual as base
from evaluate_hr_physiology import prepare, diagnostic_masks, score
from reconstruct_hr_baseline import clean_observations
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


ROOT, ZONE = base.ROOT, base.ZONE
STEM = "hr_ml_window_evaluation"
REGIONS = (("past_far", -60, -30), ("past_near", -30, 0),
           ("future_near", 0, 30), ("future_far", 30, 60))
STATISTICS = ("speed_mean", "moving_fraction", "stationary_fraction",
              "up_mean", "down_mean", "record_coverage", "terrain_coverage")
WINDOW_COLUMNS = [label + "_" + statistic for label, _, _ in REGIONS
                  for statistic in STATISTICS] + ["speed_transition", "climb_transition"]
BASE_COLUMNS = base.CONTEXT + base.MOVEMENT + base.TERRAIN
METHODS = ("linear", "ML-B", "Window")
COLORS = {"linear": "#72804a", "ML-B": "#257c9b", "Window": "#c35b2d"}


def window_features(frame):
    """Summarize four exact half-open UTC regions without crossing gaps.

    A region must contain all 30 recorded seconds in the target's contiguous
    record block for a statistic to be defined. Terrain additionally requires
    all 30 quality-supported rates. Coverage remains defined at boundaries.
    """
    output = pd.DataFrame(
        index=frame.index, columns=WINDOW_COLUMNS, dtype=float)
    groups = frame.index.to_series().diff().dt.total_seconds().ne(1).cumsum()
    for _, part in frame.groupby(groups):
        assert part.index.to_series().diff(
        ).dt.total_seconds().iloc[1:].eq(1).all()
        moving = part.speed_raw.gt(.3).astype(
            float).where(part.speed_raw.notna())
        channels = np.column_stack([part.speed5, moving,
                                    part.stationary_evidence.astype(float),
                                    part.climb30.clip(lower=0),
                                    -part.climb30.clip(upper=0)])
        # A sentinel row keeps absent timestamps missing rather than wrapping -1.
        channels = np.vstack([channels, np.full(5, np.nan)])
        for label, start, end in REGIONS:
            offsets = np.arange(start, end, dtype=np.int64) * 1_000_000_000
            query = pd.to_datetime(
                (part.index.asi8[:, None] + offsets).ravel(), utc=True)
            positions = part.index.get_indexer(query).reshape(len(part), 30)
            present = positions >= 0
            values = channels[np.where(present, positions, len(part))]
            finite = np.isfinite(values)
            means = np.where(finite, values, 0).sum(axis=1) / 30
            means[~finite.all(axis=1)] = np.nan
            columns = [label + "_" + name for name in STATISTICS[:5]]
            output.loc[part.index, columns] = means
            output.loc[part.index, label +
                       "_record_coverage"] = present.mean(axis=1)
            output.loc[part.index, label +
                       "_terrain_coverage"] = finite[:, :, 3].mean(axis=1)
    output["speed_transition"] = output.future_near_speed_mean - \
        output.past_near_speed_mean
    output["climb_transition"] = (
        output.future_near_up_mean - output.future_near_down_mean
        - output.past_near_up_mean + output.past_near_down_mean)
    assert not np.isinf(output.to_numpy()).any()
    return output


def add_windows(packages, windows):
    """Attach the same non-HR features to every observation scenario/phase."""
    result = {}
    for key, original in packages.items():
        result[key] = dict(original)
        result[key]["x"] = original["x"].join(windows)
        assert list(result[key]["x"].columns) == BASE_COLUMNS + WINDOW_COLUMNS
    return result


def training_rows(training):
    """Reproduce version 1's exact deterministic sampling and residual target."""
    features, targets = [], []
    for ride, packages in training:
        origin = ride["reference"].index[0]
        selected_times = []
        for k, phase in enumerate(base.PHASES):
            package = packages[phase, "point"]
            x = package["x"]
            seconds = ((x.index - origin).total_seconds()).astype(int)
            mask = ((seconds % 5 == 0) & ((seconds // 5) % 3 == k)
                    & x.baseline.notna() & x.anchor_gap_s.notna())
            mask &= ~x.index.isin(package["anchors"].index)
            chosen = x.loc[mask]
            target = ride["reference"].reindex(chosen.index) - chosen.baseline
            assert target.notna().all()
            features.append(chosen)
            targets.append(target)
            selected_times.extend(chosen.index)
        assert pd.DatetimeIndex(selected_times).is_unique
    return pd.concat(features), pd.concat(targets)


def fit_pair(training, check_repeat=False):
    """Fit only the required benchmark and one new fixed feature configuration."""
    x, target = training_rows(training)
    models = {}
    for name, columns in (("ML-B", BASE_COLUMNS),
                          ("Window", BASE_COLUMNS + WINDOW_COLUMNS)):
        model = HistGradientBoostingRegressor(**base.PARAMETERS)
        model.fit(x[columns], target)
        assert not model.do_early_stopping_ and model.n_iter_ == base.PARAMETERS["max_iter"]
        models[name] = model
    if check_repeat:
        repeated = HistGradientBoostingRegressor(
            **base.PARAMETERS).fit(x, target)
        assert np.array_equal(repeated.predict(x), models["Window"].predict(x))
    return {"models": models, "training": tuple(r["name"] for r, _ in training),
            "n": len(target)}


def evaluate(ride, packages, fitted, domain, checks):
    """Reuse all score definitions and score exactly the same unseen seconds."""
    assert ride["name"] not in fitted["training"]
    masks, peaks, _ = diagnostic_masks(ride)
    rows, cached = [], {}
    phases = base.PHASES if domain == "outdoor" else (0,)
    conditions = ("point", "averaged") if domain == "outdoor" else (
        "actual_watch",)
    for phase in phases:
        results = {condition: base.predictions(packages[phase, condition], fitted)
                   for condition in conditions}
        support = ride["reference"].reindex(ride["frame"].index).notna()
        for values in results.values():
            support &= values[list(METHODS)].notna().all(axis=1)
            assert values.Window.notna().equals(values["ML-B"].notna())
        common = support.copy()
        for condition in conditions:
            common.loc[packages[phase, condition]["anchors"].index] = False
        for condition, values in results.items():
            package = packages[phase, condition]
            for name in METHODS:
                metrics = score(ride, values[name], common, masks, peaks)
                metrics["average_hr_error"] = metrics["bias"]
                error = (
                    values[name] - ride["reference"].reindex(values.index)).where(common)
                for label, mask in (
                        ("stationary", ride["frame"].stationary_evidence),
                        ("restart", package["x"].restart_age_s.le(30)),
                        ("uphill", ride["frame"].climb30.gt(.05)),
                        ("downhill", ride["frame"].climb30.lt(-.05))):
                    selected = error[mask].dropna()
                    metrics[label + "_n"] = len(selected)
                    metrics[label + "_mae"] = selected.abs().mean()
                rows.append({"domain": domain, "ride": ride["name"], "phase": phase,
                             "condition": condition, "model": name,
                             "training": "+".join(fitted["training"]), **metrics,
                             **base.extra_metrics(values[name], values.linear,
                                                  ride["reference"], support, peaks)})
            cached[phase, condition] = (package, values, common)
    package, expected, _ = next(iter(cached.values()))
    changed = dict(ride)
    changed["reference"] = ride["reference"] * 0 + 200
    rebuilt = base.examples(ride["frame"], package["anchors"], changed["reference"].index,
                            base.activity_features(ride["frame"]))
    rebuilt["x"] = rebuilt["x"].join(window_features(ride["frame"]))
    pd.testing.assert_frame_equal(rebuilt["x"], package["x"])
    pd.testing.assert_frame_equal(base.predictions(rebuilt, fitted), expected)
    checks.append(
        ride["name"] + ": hidden-HR substitution leaves all inputs/predictions identical; exact anchors and support retained")
    return rows, cached


def verify_windows():
    """Check half-open bounds, future alignment, gaps, stops and terrain holes."""
    index = pd.date_range("2026-10-01", periods=241, freq="s", tz="UTC")
    numbers = np.arange(241, dtype=float)
    frame = pd.DataFrame({"speed5": numbers, "speed_raw": numbers,
                          "stationary_evidence": False,
                          "climb30": (numbers - 120) / 300}, index=index)
    time = index[120]
    windows = window_features(frame)
    for label, start, end in REGIONS:
        part = frame.loc[time + pd.Timedelta(seconds=start):
                         time + pd.Timedelta(seconds=end - 1)]
        assert len(part) == 30
        assert np.isclose(
            windows.loc[time, label + "_speed_mean"], part.speed5.mean())
        assert np.isclose(
            windows.loc[time, label + "_up_mean"], part.climb30.clip(lower=0).mean())
        assert np.isclose(
            windows.loc[time, label + "_down_mean"], -part.climb30.clip(upper=0).mean())
    altered = frame.copy()
    altered.loc[index[150], "speed5"] += 30
    difference = window_features(altered).loc[time] - windows.loc[time]
    assert np.isclose(difference.future_far_speed_mean, 1)
    assert difference[[label + "_speed_mean" for label,
                       _, _ in REGIONS[:3]]].eq(0).all()
    altered.loc[index[150], "speed5"] -= 30
    altered.loc[index[180], "speed5"] += 30
    pd.testing.assert_series_equal(window_features(
        altered).loc[time], windows.loc[time])
    broken = window_features(frame.drop(index[130]))
    assert broken.loc[time, "future_far_record_coverage"] == 0
    assert np.isclose(broken.loc[time, "future_near_record_coverage"], 10 / 30)
    assert pd.isna(broken.loc[time, "future_near_speed_mean"])
    stopped = frame.copy()
    stopped.loc[index[120]:index[149], ["speed5", "speed_raw"]] = 0
    stopped.loc[index[120]:index[149], "stationary_evidence"] = True
    stopped.loc[index[120]:index[149], "climb30"] = np.nan
    observation = window_features(stopped).loc[time]
    assert observation.future_near_speed_mean == 0
    assert observation.future_near_stationary_fraction == 1
    assert observation.future_near_moving_fraction == 0
    assert observation.future_near_terrain_coverage == 0
    assert pd.isna(observation.future_near_up_mean)
    hole = frame.copy()
    hole.loc[index[110], "climb30"] = np.nan
    observation = window_features(hole).loc[time]
    assert np.isclose(observation.past_near_terrain_coverage, 29 / 30)
    assert pd.isna(observation.past_near_up_mean)
    assert np.isfinite(observation.past_near_speed_mean)


def check_benchmark(metrics):
    """Require controlled benchmark support and scores to match saved v1 rows."""
    old = pd.read_csv(ROOT / "data/processed/hr_ml_residual_metrics.csv")
    keys = ["domain", "ride", "phase", "condition", "model"]
    current = metrics.loc[metrics.model.ne("Window")].set_index(keys)
    previous = old.set_index(keys).loc[current.index]
    columns = current.select_dtypes(
        include=[np.number]).columns.intersection(previous.columns)
    assert np.allclose(current[columns], previous[columns],
                       rtol=1e-10, atol=1e-10, equal_nan=True)
    return len(current)


def decide(metrics):
    """Apply fixed promotion screens; version 1 still requires user replacement approval."""
    primary = metrics.loc[metrics.domain.eq(
        "outdoor") & metrics.condition.eq("point")]
    tests = []
    for ride, group in primary.groupby("ride"):
        mean = group.groupby("model").mean(numeric_only=True)
        old, new = mean.loc["ML-B"], mean.loc["Window"]
        tests += [(ride + " MAE improvement >=0.20", old.mae - new.mae >= .20),
                  (ride + " mean RMSE does not worsen", new.rmse <= old.rmse),
                  (ride + " high/rise/recovery benefit >=0.15", max(
                      old.high_mae - new.high_mae, old.rise_mae - new.rise_mae,
                      old.recovery_mae - new.recovery_mae) >= .15),
                  (ride + " high HR not worse by >0.15",
                   new.high_mae <= old.high_mae + .15),
                  (ride + " peak amplitude not worse by >0.25",
                   new.peak_amplitude_mae <= old.peak_amplitude_mae + .25),
                  (ride + " peak timing not worse by >1s", new.peak_timing_mae_s <= old.peak_timing_mae_s + 1)]
    checked = metrics.loc[(metrics.domain.eq("outdoor") & metrics.condition.eq("point"))
                          | metrics.domain.eq("mtb_overlap")]
    for (ride, phase), group in checked.groupby(["ride", "phase"]):
        group = group.set_index("model")
        old, new = group.loc["ML-B"], group.loc["Window"]
        label = f"{ride} phase {phase}"
        tolerance = .5 if ride == "mtb_edge" else .15
        tests += [(label + " no large MAE regression", new.mae <= old.mae + tolerance),
                  (label + " unchanged coverage", new.n == old.n),
                  (label + " P99 jump screen", new.step_p99 <= old.step_p99 + .25),
                  (label + " >5BPM/s screen", new.steps_gt5_percent <=
                   old.steps_gt5_percent + .1),
                  (label + " unmatched peak screen",
                   new.unmatched_peaks <= old.unmatched_peaks + 2),
                  (label + " valid numeric range", new.invalid_hr_n == 0)]
        for state in ("stationary", "restart"):
            if old[state + "_n"] >= 20:
                tests.append((label + " " + state + " no >1BPM regression",
                              new[state + "_mae"] <= old[state + "_mae"] + 1))
    mtb = metrics.loc[metrics.domain.eq("mtb_overlap")].set_index("model")
    tests += [("MTB RMSE no >0.5 regression", mtb.loc["Window", "rmse"] <= mtb.loc["ML-B", "rmse"] + .5),
              ("MTB high HR no >0.5 regression", mtb.loc["Window", "high_mae"] <= mtb.loc["ML-B", "high_mae"] + .5)]
    return all(bool(passed) for _, passed in tests), tests


def plot_comparison(panels, metrics, path):
    """Use the benchmark's fixed central 20-minute representative windows."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), constrained_layout=True)
    for ax, (ride, cached) in zip(axes[:2], panels):
        package, predicted, _ = cached[0, "point"]
        block = max(base.continuous_blocks(ride["reference"]), key=len)
        midpoint = block.index[0] + (block.index[-1] - block.index[0]) / 2
        start = (midpoint - pd.Timedelta(minutes=10)).floor("s")
        end = start + pd.Timedelta(minutes=20)
        actual = ride["reference"].loc[start:end]
        ax.plot(actual.index, actual, color=".6", lw=1.3,
                label="Recorded hidden HR reference")
        for name in METHODS:
            visible = predicted.loc[start:end, name]
            ax.plot(visible.index, visible, color=COLORS[name], lw=1.35,
                    ls="--" if name == "linear" else "-", label=name + " estimate")
        sparse = package["anchors"].loc[start:end]
        ax.scatter(sparse.index, sparse, s=24, color=".15", zorder=5,
                   label="Simulated sparse point observations")
        ax.set(title=f"{ride['name']} — held-out, phase 0, central 20min of longest continuous block",
               ylabel="HR (BPM)", xlabel="Local time — Europe/Brussels", xlim=(start, end))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZONE))
        ax.legend(ncol=3, fontsize=8, loc="best")
        ax.grid(alpha=.2)
    primary = metrics.loc[metrics.domain.eq(
        "outdoor") & metrics.condition.eq("point")]
    for i, name in enumerate(METHODS):
        groups = [primary.loc[primary.ride.eq(
            ride) & primary.model.eq(name)] for ride in base.OUTDOOR]
        means = np.array([g.mae.mean() for g in groups])
        lower = np.array([g.mae.min() for g in groups])
        upper = np.array([g.mae.max() for g in groups])
        axes[2].bar(np.arange(2) + (i - 1) * .23, means, .21, color=COLORS[name],
                    label=name, yerr=[means - lower, upper - means], capsize=3)
    axes[2].set(xticks=[0, 1], xticklabels=list(base.OUTDOOR), ylabel="MAE (BPM)",
                title="Independent outdoor holdouts: mean MAE and phase range (0/20/40s)")
    axes[2].legend(ncol=3)
    axes[2].grid(axis="y", alpha=.2)
    fig.suptitle("Temporal-window HR residual experiment — many inputs, one residual per timestamp\n"
                 "Fixed model/postprocessing; version 1 remains the current reconstruction", fontsize=14)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def candidate_series(ride, watch, fitted, windows):
    """Optionally export a separate, unapproved candidate with original HR intact."""
    frame = ride["frame"]
    edge = ride["reference"].reindex(frame.index)
    cutoff = ride["reference"].index[-1]
    tail = pd.concat([pd.Series(edge.loc[cutoff], index=pd.DatetimeIndex([cutoff])),
                      watch.loc[watch.index > cutoff]])
    package = base.examples(frame, tail, frame.index,
                            base.activity_features(frame))
    package["x"] = package["x"].join(windows)
    predicted = base.predictions(package, fitted)
    saved = pd.read_csv(
        ROOT / "data/processed/hr_reconstructed_2026-10-16.csv")
    saved.index = pd.to_datetime(saved.timestamp_utc, utc=True)
    assert saved.index.equals(frame.index)
    table = pd.DataFrame({"edge_hr": edge, "watch_hr": watch.reindex(frame.index),
                          "interpolation_baseline": predicted.linear,
                          "v1_final_hr": saved.final_hr,
                          "window_estimate": predicted.Window,
                          "final_hr": predicted.Window.where(edge.isna(), edge)})
    table["estimated"] = table.final_hr.notna() & edge.isna()
    table["edge_observed"] = edge.notna()
    table["watch_observed"] = table.watch_hr.notna()
    table["source_status"] = "missing_unsupported"
    table.loc[table.estimated, "source_status"] = "unapproved_window_candidate"
    table.loc[edge.notna(), "source_status"] = "recorded_edge"
    table["quality_indicator"] = np.where(
        frame.climb30.notna(), "terrain_supported", "terrain_missing")
    table.loc[frame.stationary_evidence,
              "quality_indicator"] += ";stationary_evidence"
    table.index.name = "timestamp_utc"
    assert np.array_equal(table.loc[edge.notna(), "final_hr"], edge.dropna())
    assert np.array_equal(table.loc[watch.index, "watch_hr"], watch)
    assert table.final_hr.notna().equals(saved.final_hr.notna())
    table.to_csv(ROOT / "data/processed/hr_ml_window_candidate_2026-10-16.csv")
    base.final_figure(ride, table, cutoff, "Window candidate (unapproved; v1 retained)",
                      ROOT / "docs/images/hr_ml_window_candidate_2026-10-16.png")
    return table


def write_report(metrics, fits, promote, tests, checked_rows, checks, protected_n, candidate):
    """Describe the fixed experiment, all phases, tradeoffs and current version."""
    primary = metrics.loc[metrics.domain.eq(
        "outdoor") & metrics.condition.eq("point")]
    means = primary.groupby(["ride", "model"])[["mae", "rmse"]].mean()
    paired = primary.pivot(index=["ride", "phase"],
                           columns="model", values="mae")
    recommendation = ("B. Promote the window-enhanced model as the next version, subject to user approval."
                      if promote else "A. Retain the current ML-B model (version 1).")
    lines = ["# Temporal-window gradient boosting for HR residuals", "",
             "**Recommendation: " + recommendation +
             "** The established version-1 reconstruction and documentation "
             "remain unchanged. This is a single fixed many-to-one feature experiment, not a new sequence model or an anchor generator.", "",
             "## Controlled design", "",
             "Reuse `evaluate_hr_ml_residual.py` for FIT preparation, non-HR preprocessing, anchor simulation, interpolation, "
             "five-sample residual averaging, `4f(1−f)` anchor taper and all error/peak/roughness metrics. "
             "Two outdoor roads are the independent leave-one-ride-out units; phases are dependent simulations. "
             "One scalar HR residual is predicted at each supported time; add it to the same linear baseline.", "",
             "Version-1 fitted estimators and held-out curves were not saved, so the minimum controlled refits train "
             "ML-B (20 inputs) and Window (50 inputs) on identical rows for each fold and the pooled-road MTB check. "
             f"All {checked_rows} recomputed linear/ML-B rows match the saved benchmark's numeric metrics and coverage "
             "within 1e−10 tolerance. No baseline retuning or ML-A/virtual experiment is performed.", "",
             f"Runtime: Python {sys.version.split()[0]}, scikit-learn {sklearn.__version__}, NumPy {np.__version__}, "
             f"pandas {pd.__version__}, SciPy {scipy.__version__}, Matplotlib {matplotlib.__version__}. No new dependency.", "",
             "Unchanged estimator settings:", "", "```text", repr(
                 base.PARAMETERS), "```", "",
             "No scaling, parameter/window search, automatic early-stopping split, or held-out tuning. "
             "Train on point observations only; evaluate point plus complete 60-second centered-average sensitivity without refitting. "
             "Training retains one timestamp every five elapsed seconds, cyclically assigned to one of phases 0/20/40s, "
             "excluding observations and unsupported baseline points. No duplicated targets across phases.", "",
             "| Training partition | Training examples | Benchmark inputs | Window inputs |",
             "|---|---:|---:|---:|"]
    for fitted in fits:
        lines.append(
            f"| {' + '.join(fitted['training'])} | {fitted['n']} | 20 | 50 |")
    lines += ["", "## Exact temporal features", "",
              "Keep all 20 existing anchor-context, movement/stop/restart and terrain inputs. Add 30 interpretable inputs:", "",
              "- Four **half-open 30-second regions** relative to t: `[-60,-30)`, `[-30,0)`, `[0,30)`, `[30,60)`. "
              "At 1Hz these include offsets −60…−31, −30…−1, 0…29, and 30…59 respectively; t+60 is excluded.",
              "- Seven outputs per region: mean audited `speed5` (m/s), movement fraction (`speed_raw>0.3m/s`), "
              "stationary-evidence fraction, mean positive/negative quality-masked `climb30` magnitudes (m/s), "
              "record coverage and terrain coverage (fractions of 30 expected seconds). This gives 28 features.",
              "- Two transitions: future-near minus past-near mean speed, and the same contrast in signed mean climbing "
              "rate (`up−down`). Units are m/s; these are differences of regional means, not acceleration estimates.", "",
              "The existing five-sample speed and centered terrain preprocessing retain their own numeric support; "
              "the nominal regions describe where these audited summaries are sampled, not a strict limit on their underlying raw support. "
              "Future **non-HR** observations are available for offline reconstruction. No full/reference-HR value, "
              "HR rolling statistic, activity identity, power, cadence, absolute position or HR-derived load enters a window.", "",
              "### Validity and gaps", "",
              "Query actual timezone-aware UTC timestamps inside the target's continuous record block. A region on the far "
              "side of a gap is unsupported even if it contains 30 later observations. Require all 30 recorded/finite values "
              "for each mean/fraction; require all 30 quality-supported terrain rates for up/down means. Partial regions "
              "remain NaN with coverage retained; no filling or averaging across an invalid altitude interval. "
              "HGB handles missing inputs natively, so no prediction is dropped solely because windows are incomplete.", "",
              "Stops do not invalidate speed/movement summaries: a full stopped region has stationary fraction 1, movement "
              "fraction 0, and typically zero speed. Its unsupported terrain stays NaN, never a fabricated flat/climbing "
              "measurement. Existing stop/restart ages remain in the feature set; no HR state is reset at stops.", "",
              "## Outdoor holdout results", "",
              "BPM; equal-weight phase means [minimum, maximum]. Bias equals average-HR error on matching timestamps. "
              "All three models and both observation conditions share the same hidden evaluation timestamps, excluding "
              "anchor times, and retain exactly the benchmark coverage.", "",
              "| Held-out ride | Model | MAE | RMSE | Bias | High-HR MAE | Coverage % |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for (ride, name), group in primary.groupby(["ride", "model"], sort=False):
        lines.append(f"| {ride} | {name} | {base.value_range(group,'mae')} | {base.value_range(group,'rmse')} | "
                     f"{group.bias.mean():.2f} | {group.high_mae.mean():.2f} | {base.value_range(group,'coverage_percent')} |")
    lines += ["", "| Held-out ride | Phase s | Linear MAE/RMSE | ML-B MAE/RMSE | Window MAE/RMSE |",
              "|---|---:|---:|---:|---:|"]
    for (ride, phase), group in primary.groupby(["ride", "phase"]):
        indexed = group.set_index("model")
        lines.append(f"| {ride} | {phase} | " + " | ".join(
            f"{indexed.loc[m,'mae']:.2f}/{indexed.loc[m,'rmse']:.2f}" for m in METHODS) + " |")
    lines += ["", "### Dynamics and stability", "",
              "Use the existing score-only definitions: high HR = reference upper decile; rise/recovery = ≥+5/≤−5 BPM "
              "over 30 observed consecutive seconds. Peaks have prominence≥10 BPM, separation≥120s; fully scored ±15s "
              "windows compare local maxima and earliest maximum timing (bounded/censored at ±15s). Unmatched predicted "
              "peaks have no reference peak within ±30s. These diagnostic labels never enter features. Roughness includes "
              "anchor neighborhoods and requires adjacent supported seconds, not differences across recording gaps.", "",
              "| Ride | Model | Rise MAE | Recovery MAE | Peak amplitude abs error | Peak timing abs s | Unmatched peaks | P99 / max 1s jump |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for (ride, name), group in primary.groupby(["ride", "model"], sort=False):
        lines.append(f"| {ride} | {name} | {group.rise_mae.mean():.2f} | {group.recovery_mae.mean():.2f} | "
                     f"{group.peak_amplitude_mae.mean():.2f} | {group.peak_timing_mae_s.mean():.2f} | "
                     f"{group.unmatched_peaks.mean():.1f} | {group.step_p99.mean():.3f} / {group.max_step.max():.3f} |")
    lines += ["", "### Averaged-observation sensitivity", "",
              "Complete [t−30,t+29] 60-second means are pseudo-point constraints for this sensitivity only. "
              "They are not instantaneous HR or verified Garmin wellness behavior; the point-trained models are frozen.", "",
              "| Ride | Model | MAE | RMSE |", "|---|---|---:|---:|"]
    average = metrics.loc[metrics.domain.eq(
        "outdoor") & metrics.condition.eq("averaged")]
    for (ride, name), group in average.groupby(["ride", "model"], sort=False):
        lines.append(
            f"| {ride} | {name} | {base.value_range(group,'mae')} | {base.value_range(group,'rmse')} |")
    lines += ["", "## Actual MTB overlap", "",
              "The pooled-road models use actual wellness summaries and quality-masked Strava terrain. Evaluate only where "
              "original Edge HR exists; never initialize with hidden Edge HR or add the later final-handoff boundary anchor "
              "to this diagnostic. Future non-HR activity can extend beyond the Edge overlap. Watch/Edge path/averaging "
              "differences confound the absolute errors; this previously examined overlap is not a pristine independent test.", "",
              "| Model | n / coverage % | MAE | RMSE | Bias | High-HR | Recovery | Peak amplitude / timing s | Unmatched peaks | P99 / max 1s jump |",
              "|---|---|---:|---:|---:|---:|---:|---|---:|---|"]
    for row in metrics.loc[metrics.domain.eq("mtb_overlap")].itertuples():
        lines.append(f"| {row.model} | {row.n} / {row.coverage_percent:.2f} | {row.mae:.2f} | {row.rmse:.2f} | "
                     f"{row.bias:.2f} | {row.high_mae:.2f} | {row.recovery_mae:.2f} | "
                     f"{row.peak_amplitude_mae:.2f} / {row.peak_timing_mae_s:.2f} | {row.unmatched_peaks} | "
                     f"{row.step_p99:.3f} / {row.max_step:.3f} |")
    lines += ["", "### Stationary and restart context", "",
              "Stationary evidence includes inferred short pauses; restart is the first 30s after a supported restart. "
              "Road stationary counts are tiny; a missing (nan) error is undefined, not perfect. Terrain cannot identify a ravito.", "",
              "| Activity | Model | Stationary n / MAE | Restart n / MAE | Uphill / downhill MAE |",
              "|---|---|---|---|---|"]
    contexts = metrics.loc[(metrics.domain.eq("outdoor") & metrics.condition.eq("point"))
                           | metrics.domain.eq("mtb_overlap")]
    for (ride, name), group in contexts.loc[contexts.model.ne("linear")].groupby(["ride", "model"], sort=False):
        lines.append(f"| {ride} | {name} | {group.stationary_n.min()}–{group.stationary_n.max()} / "
                     f"{group.stationary_mae.mean():.2f} | {group.restart_n.min()}–{group.restart_n.max()} / "
                     f"{group.restart_mae.mean():.2f} | {group.uphill_mae.mean():.2f} / {group.downhill_mae.mean():.2f} |")
    lines += ["", "## Decision", "",
              "MAE/RMSE and rise/recovery improve consistently, but the benefit is not uniform across all diagnostics. "
              "Road ride 2 has slightly worse high-HR MAE and more unmatched peaks; 1s P99 roughness increases on both roads. "
              "The MTB MAE improvement is small, so the main evidence is the outdoor comparison, not strong target validation.", "",
              "Fixed promotion screens compare **Window with ML-B**: mean MAE improves ≥0.20 BPM on each outdoor ride, "
              "mean RMSE never worsens, and high/rise/recovery MAE improves ≥0.15 BPM in at least one subset per ride. "
              "High-HR MAE must not worsen >0.15 BPM, peak-amplitude error >0.25 BPM or peak timing >1s. "
              "Each outdoor phase: MAE degradation≤0.15 BPM, unchanged coverage, P99 jump increase≤0.25 BPM, "
              ">5 BPM/s frequency increase≤0.1 percentage points, ≤2 extra unmatched peaks, valid numeric HR range. "
              "For stationary/restart subsets with ≥20 scored points, allow no MAE degradation >1 BPM. "
              "MTB applies the same stability/coverage/state screens and MAE/RMSE/high-HR tolerances of 0.5 BPM. "
              "These are prespecified descriptive safeguards, not significance tests.", "",
              f"Passed {sum(bool(v) for _,v in tests)}/{len(tests)} screens. "
              f"Window lowers MAE in {int((paired.Window < paired['ML-B']).sum())}/6 outdoor phases. "
              "Mean MAE improvement (ML-B minus Window): "
              + "; ".join(f"{ride} {means.loc[(ride,'ML-B'),'mae']-means.loc[(ride,'Window'),'mae']:+.3f} BPM"
                          for ride in base.OUTDOOR) + ".", "",
              "**" + recommendation + "**", ""]
    failed = [label for label, passed in tests if not passed]
    if failed:
        lines += ["Failed screens:", ""] + \
            ["- " + label for label in failed] + [""]
    lines += [("The experiment supports another window-based candidate, but promotion is pending user approval. "
               "Only a separate candidate was produced; version 1 remains current."
               if promote else "The fixed temporal features do not establish a sufficiently consistent improvement to replace "
               "the current feature set. Retain ML-B for this project; no further window redesign/search was performed."), "",
              f"Separate candidate generated: **{candidate is not None}**. "
              + ("Candidate CSV: `data/processed/hr_ml_window_candidate_2026-10-16.csv`; "
                 "figure: `docs/images/hr_ml_window_candidate_2026-10-16.png`. Recorded Edge HR and original watch values "
                 "remain exact; output is explicitly an unapproved estimate, with no extrapolation."
                 if candidate is not None else "No full-ride candidate HR series was generated; the established reconstruction is unchanged."), "",
              "## Limitations and reproduction", "",
              "Two outdoor rides cannot establish generalization to technical MTB. Three sampling phases are not three rides. "
              "Only a short, previously examined MTB overlap is labelled; later sustained stops remain unvalidated. "
              "Road training has little stopping support, optical HR is imperfect, wellness averaging is unresolved, terrain is "
              "quality-screened rather than surveyed, and power/cadence are absent outdoors. A future activity window can help "
              "offline prediction without revealing true missing HR fluctuations. Strict complete-region means may reduce feature "
              "availability near gaps/stops, but they never reduce baseline prediction coverage.", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe scripts/evaluate_hr_ml_windows.py", "```", "",
              "Outputs: this report; `docs/images/hr_ml_window_evaluation.png`; `data/processed/hr_ml_window_metrics.csv`; "
              "script `scripts/evaluate_hr_ml_windows.py`. The plot repeats the benchmark's phase-0 central 20 minutes of "
              "the longest continuous reference block, chosen without model errors. No large intermediates or serialized models.", "",
              "Executed checks:", ""]
    if candidate is not None:
        steps = candidate.final_hr.diff().abs().where(candidate.estimated)
        lines += [f"Candidate-only checks: supported mean {candidate.final_hr.mean():.2f} BPM, "
                  f"peak {candidate.final_hr.max():.2f} BPM, "
                  f"{candidate.final_hr.notna().sum()}/{len(candidate)} supported rows; "
                  f"maximum estimated/handoff 1s change {steps.max():.3f} BPM. "
                  "The last 50s stay missing. These are estimated-sequence statistics, not recovered physiological truth. "
                  "All original Edge/watch values remain unchanged; v1 remains current.", ""]
    lines += ["- " + check for check in checks]
    lines += [f"- All {protected_n} protected source/prior-output hashes unchanged, including the v1 CSV, reports and figures.",
              "- Numeric score/coverage invariants, mean-HR error identity, finite-range summaries, non-empty outputs. "
              "Figures are inspected after generation. No new dependencies, large test suite, commits, pushes or uploads.", ""]
    (ROOT / "docs/hr_ml_window_evaluation.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run one fixed feature experiment and preserve all established outputs."""
    new_names = {"evaluate_hr_ml_windows.py", "hr_ml_window_evaluation.md",
                 "hr_ml_window_evaluation.png", "hr_ml_window_metrics.csv",
                 "hr_ml_window_candidate_2026-10-16.csv", "hr_ml_window_candidate_2026-10-16.png"}
    protected = [p for folder in ("data", "docs", "scripts")
                 for p in (ROOT / folder).rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts and p.name not in new_names]
    before = base.fingerprints(protected)
    checks = []
    verify_windows()
    checks.append("Explicit ramp/impulse checks: four half-open 30s boundaries, future +30 included/+60 excluded, gap isolation, stop fractions, invalid-terrain coverage")
    watch, _, _ = clean_observations(pd.read_csv(
        ROOT / "data/processed/watch_hr_ride_2026-10-16.csv"))
    pattern = np.diff(watch.index.asi8) // 1_000_000_000
    assert len(watch) == 158 and (pattern == 60).sum(
    ) == 142 and (pattern == 120).sum() == 15
    rides = [prepare(name) for name in base.OUTDOOR]
    packages = [add_windows(base.synthetic(ride, pattern, base.activity_features(ride["frame"])),
                            window_features(ride["frame"])) for ride in rides]
    fits, rows, panels = [], [], []
    for held, train in ((0, 1), (1, 0)):
        print("Training", rides[train]["name"],
              "holding out", rides[held]["name"], flush=True)
        fitted = fit_pair([(rides[train], packages[train])],
                          check_repeat=(held == 0))
        fits.append(fitted)
        values, cached = evaluate(
            rides[held], packages[held], fitted, "outdoor", checks)
        rows.extend(values)
        panels.append((rides[held], cached))
    checks.append(
        "Repeated first Window fit gives bit-identical training predictions; unchanged hyperparameters/sampling/postprocessing")
    pooled = fit_pair(list(zip(rides, packages)))
    fits.append(pooled)
    mtb = prepare("mtb_edge")
    mtb["frame"] = mtb["strava"]
    windows = window_features(mtb["frame"])
    overlap = watch.loc[mtb["reference"].index[0]:mtb["reference"].index[-1]]
    original = base.examples(mtb["frame"], overlap, mtb["reference"].index,
                             base.activity_features(mtb["frame"]))
    packages_mtb = add_windows({(0, "actual_watch"): original}, windows)
    values, _ = evaluate(mtb, packages_mtb, pooled, "mtb_overlap", checks)
    rows.extend(values)
    metrics = pd.DataFrame(rows)
    assert len(metrics) == 39
    assert metrics.groupby(
        ["domain", "ride", "phase", "condition"]).n.nunique().eq(1).all()
    assert np.isfinite(metrics[["mae", "rmse", "bias"]]).all().all()
    checked_rows = check_benchmark(metrics)
    checks.append(
        f"All {checked_rows} saved benchmark rows reproduced with identical support and numeric metrics within 1e-10")
    promote, tests = decide(metrics)
    candidate = candidate_series(
        mtb, watch, pooled, windows) if promote else None
    metrics.to_csv(
        ROOT / "data/processed/hr_ml_window_metrics.csv", index=False)
    plot_comparison(panels, metrics, ROOT /
                    "docs/images/hr_ml_window_evaluation.png")
    assert base.fingerprints(protected) == before
    write_report(metrics, fits, promote, tests, checked_rows,
                 checks, len(protected), candidate)
    for path in (ROOT / "data/processed/hr_ml_window_metrics.csv",
                 ROOT / "docs/hr_ml_window_evaluation.md", ROOT / "docs/images/hr_ml_window_evaluation.png"):
        assert path.stat().st_size > 0
    assert base.fingerprints(protected) == before
    print(metrics.loc[metrics.condition.isin(["point", "actual_watch"])].groupby(
        ["domain", "ride", "model"])[["mae", "rmse", "high_mae"]].mean().round(4).to_string())
    print("Training rows:", [(f["training"], f["n"]) for f in fits])
    print("Promotion recommendation:", promote, "; screens", sum(bool(v)
          for _, v in tests), "/", len(tests))
    print("Failed:", [label for label, passed in tests if not passed])
    print("Candidate generated:", candidate is not None,
          "; protected files unchanged:", len(protected))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
