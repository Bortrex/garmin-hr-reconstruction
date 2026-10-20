"""Evaluate fixed gradient-boosted residuals and export a cautious final ride.

Run: .venv/Scripts/python scripts/evaluate_hr_ml_residual.py
Earlier analyses and original observations are never overwritten.
"""

import hashlib

import numpy as np
import pandas as pd
import sklearn
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from audit_cycling_predictors import ROOT, ZONE, runs
from evaluate_hr_interpolation import PHASES, continuous_blocks, reconstruct, sampling_times
from evaluate_hr_physiology import prepare, make_anchors, diagnostic_masks, score
from reconstruct_hr_baseline import clean_observations
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


STEM = "hr_ml_residual_evaluation"
OUTDOOR = ("road_ride_1", "road_ride_2")
METHODS = ("linear", "pchip", "ML-A", "ML-B")
PARAMETERS = dict(learning_rate=.05, max_iter=120, max_leaf_nodes=7,
                  max_depth=3, min_samples_leaf=100, l2_regularization=10.,
                  early_stopping=False, random_state=265)
CONTEXT = ["baseline", "anchor_delta", "since_anchor_s", "until_anchor_s",
           "anchor_gap_s", "interval_fraction", "interpolation_slope"]
MOVEMENT = ["speed5", "speed30", "speed_change10", "stationary",
            "stop_age_s", "restart_age_s", "moving_fraction60", "movement_unknown"]
TERRAIN = ["climb_up", "climb_down",
           "up_mean60", "down_mean60", "terrain_valid"]
COLORS = {"linear": "#61773c", "pchip": "#8662a0",
          "ML-A": "#287d9d", "ML-B": "#c26032"}


def fingerprints(paths):
    """Record bytes of source files and prior analyses for preservation checks."""
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def activity_features(frame):
    """Use only audited non-HR channels; retain missing values explicitly."""
    result = pd.DataFrame(index=frame.index)
    group = frame.dt.ne(1).cumsum()
    result["speed5"] = frame.speed5
    result["speed30"] = frame.speed_raw.groupby(group).transform(
        lambda s: s.rolling(30, min_periods=30).mean())
    previous = frame.speed5.groupby(group).shift(10)
    result["speed_change10"] = (frame.speed5 - previous) / 10
    stationary = frame.stationary_evidence
    result["stationary"] = stationary.astype(float)
    result["movement_unknown"] = frame.uncertain
    result["stop_age_s"] = 0.
    result["restart_age_s"] = np.nan
    for indices in runs(stationary, frame.index):
        time = frame.index[indices[0]]
        age = (frame.index[indices] - time).total_seconds()
        result.loc[frame.index[indices], "stop_age_s"] = np.minimum(age, 300)
    for indices in runs(~stationary, frame.index):
        first = indices[0]
        if first > 0 and frame.dt.iloc[first] == 1 and stationary.iloc[first - 1]:
            age = (frame.index[indices] - frame.index[first]).total_seconds()
            result.loc[frame.index[indices],
                       "restart_age_s"] = np.minimum(age, 300)
    moving = frame.speed_raw.gt(.3).astype(
        float).where(frame.speed_raw.notna())
    result["moving_fraction60"] = moving.groupby(group).transform(
        lambda s: s.rolling(60, min_periods=60).mean())
    # Offline centered terrain is already audited; do not use the ODE's lag.
    result["climb_up"] = frame.climb30.clip(lower=0)
    result["climb_down"] = -frame.climb30.clip(upper=0)
    for source, target in (("climb_up", "up_mean60"), ("climb_down", "down_mean60")):
        result[target] = result[source].groupby(group).transform(
            lambda s: s.rolling(60, min_periods=60).mean())
    result["terrain_valid"] = frame.climb30.notna().astype(float)
    assert result.loc[stationary, [
        "climb_up", "climb_down"]].isna().all().all()
    assert not np.isinf(result.to_numpy()).any()
    return result[MOVEMENT + TERRAIN]


def examples(frame, anchors, availability, static):
    """Interpolate declared anchors and construct context without dense HR."""
    grid = pd.date_range(frame.index[0], frame.index[-1], freq="s")
    available = pd.Series(1., index=frame.index.intersection(availability))
    blocks = continuous_blocks(available)
    bounds = [tuple(part.index[[0, -1]]) for part in blocks]
    baseline = reconstruct(anchors, grid, bounds).reindex(frame.index)
    x = pd.DataFrame(index=frame.index, columns=CONTEXT, dtype=float)
    x["baseline"] = baseline.linear
    spans = []
    for start, end in bounds:
        sparse = anchors.loc[start:end]
        for i in range(len(sparse) - 1):
            left, right = sparse.index[i:i + 2]
            duration = (right - left).total_seconds()
            if duration > 180:
                continue
            times = frame.index[(frame.index >= left) & (frame.index <= right)]
            if len(times) != duration + 1:
                continue
            since = (times - left).total_seconds().to_numpy()
            delta = sparse.iloc[i + 1] - sparse.iloc[i]
            x.loc[times, "anchor_delta"] = delta
            x.loc[times, "since_anchor_s"] = since
            x.loc[times, "until_anchor_s"] = duration - since
            x.loc[times, "anchor_gap_s"] = duration
            x.loc[times, "interval_fraction"] = since / duration
            x.loc[times, "interpolation_slope"] = delta / duration
            spans.append((left, right))
    x = x.join(static)
    assert list(x.columns) == CONTEXT + MOVEMENT + TERRAIN
    return {"x": x, "baseline": baseline, "spans": spans, "anchors": anchors}


def synthetic(ride, pattern, static):
    """Reuse the original elapsed-time schedules and observation definitions."""
    reference = ride["reference"]
    result = {}
    for phase in PHASES:
        times = sampling_times(
            reference.index[0], reference.index[-1], pattern, phase)
        for scenario in ("point", "averaged"):
            anchors = make_anchors(reference, times, scenario)
            result[phase, scenario] = examples(
                ride["frame"], anchors, reference.index, static)
    return result


def fit(training):
    """Use one phase example per selected timestamp, with no random early split."""
    features, targets = [], []
    for ride, packages in training:
        origin = ride["reference"].index[0]
        for k, phase in enumerate(PHASES):
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
        selected = pd.concat([v for v in features[-3:]])
        assert selected.index.is_unique
    x, target = pd.concat(features), pd.concat(targets)
    models = {}
    for name, columns in (("ML-A", CONTEXT + MOVEMENT),
                          ("ML-B", CONTEXT + MOVEMENT + TERRAIN)):
        model = HistGradientBoostingRegressor(**PARAMETERS)
        model.fit(x[columns], target)
        assert not model.do_early_stopping_ and model.n_iter_ == PARAMETERS["max_iter"]
        models[name] = model
    return {"models": models, "training": tuple(r["name"] for r, _ in training),
            "n": len(target)}


def predictions(package, fitted):
    """Apply a fixed 5-sample residual mean and parabolic anchor taper."""
    x = package["x"]
    result = package["baseline"].copy()
    supported = x.baseline.notna() & x.anchor_gap_s.notna()
    for name, model in fitted["models"].items():
        raw = pd.Series(np.nan, index=x.index)
        raw.loc[supported] = model.predict(
            x.loc[supported, model.feature_names_in_])
        correction = pd.Series(np.nan, index=x.index)
        for left, right in package["spans"]:
            times = x.loc[left:right].index
            mean = raw.loc[times].rolling(5, center=True, min_periods=1).mean()
            fraction = (times - left).total_seconds() / \
                (right - left).total_seconds()
            correction.loc[times] = mean * 4 * fraction * (1 - fraction)
        result[name] = result.linear + correction
        # An isolated original observation is retained, without extrapolation.
        result.loc[package["anchors"].index, name] = package["anchors"].values
        assert np.array_equal(result.loc[package["anchors"].index, name],
                              package["anchors"].values)
        assert result.loc[result.linear.isna(), name].isna().all()
        assert result[name].notna().equals(result.linear.notna())
    return result


def extra_metrics(predicted, baseline, reference, support, peaks):
    """Measure actual trajectory roughness, including anchor neighborhoods."""
    delta = predicted.diff().where(support & support.shift(fill_value=False)
                                   & predicted.index.to_series().diff().dt.total_seconds().eq(1)).dropna()
    correction = predicted - baseline
    corr_step = correction.diff().where(
        delta.index.to_series().reindex(predicted.index).notna())
    unmatched = 0
    for block in continuous_blocks(predicted.where(support).dropna()):
        indices, _ = find_peaks(block.to_numpy(), prominence=10, distance=120)
        for time in block.index[indices]:
            if not any(abs((time - observed).total_seconds()) <= 30 for observed in peaks):
                unmatched += 1
    return {"step_p99": delta.abs().quantile(.99), "max_step": delta.abs().max(),
            "steps_gt5_percent": 100 * delta.abs().gt(5).mean(),
            "correction_abs_p95": correction[support].abs().quantile(.95),
            "correction_step_p99": corr_step.abs().quantile(.99),
            "unmatched_peaks": unmatched,
            "invalid_hr_n": int(((predicted <= 0) | (predicted >= 255))[support].sum())}


def evaluate(ride, packages, fitted, domain, checks):
    """Score all configurations on the same hidden timestamps per phase."""
    assert ride["name"] not in fitted["training"]
    masks, peaks, _ = diagnostic_masks(ride)
    rows, cached = [], {}
    phases = PHASES if domain != "mtb_overlap" else (0,)
    scenarios = ("point", "averaged") if domain != "mtb_overlap" else (
        "actual_watch",)
    for phase in phases:
        results = {scenario: predictions(packages[phase, scenario], fitted)
                   for scenario in scenarios}
        support = ride["reference"].reindex(ride["frame"].index).notna()
        for result in results.values():
            support &= result.notna().all(axis=1)
        common = support.copy()
        for scenario in scenarios:
            common.loc[packages[phase, scenario]["anchors"].index] = False
        assert common.any()
        for scenario, result in results.items():
            package = packages[phase, scenario]
            for name in METHODS:
                values = score(ride, result[name], common, masks, peaks)
                values["average_hr_error"] = values["bias"]
                error = (
                    result[name] - ride["reference"].reindex(result.index)).where(common)
                for label, mask in (
                        ("stationary", ride["frame"].stationary_evidence),
                        ("restart", package["x"].restart_age_s.le(30)),
                        ("uphill", ride["frame"].climb30.gt(.05)),
                        ("downhill", ride["frame"].climb30.lt(-.05))):
                    selected = error[mask].dropna()
                    values[label + "_n"] = len(selected)
                    values[label + "_mae"] = selected.abs().mean()
                rows.append({"domain": domain, "ride": ride["name"], "phase": phase,
                             "condition": scenario, "model": name,
                             "training": "+".join(fitted["training"]),
                             **values, **extra_metrics(result[name], result.linear,
                                                       ride["reference"], support, peaks)})
            cached[phase, scenario] = (package, result, common)
    first = next(iter(cached.values()))
    verify_features(ride, first[0], fitted, first[1], checks)
    return rows, cached


def verify_features(ride, package, fitted, expected, checks):
    """Demonstrate feature/prediction independence from hidden HR values."""
    changed = dict(ride)
    changed["reference"] = ride["reference"] * 0 + 200
    static = activity_features(ride["frame"])
    rebuilt = examples(ride["frame"], package["anchors"],
                       changed["reference"].index, static)
    pd.testing.assert_frame_equal(rebuilt["x"], package["x"])
    pd.testing.assert_frame_equal(predictions(rebuilt, fitted), expected)
    left, right = package["spans"][len(package["spans"]) // 2]
    missing = (left + (right - left) / 2).floor("s")
    broken = ride["frame"].drop(missing).copy()
    broken["dt"] = broken.index.to_series().diff().dt.total_seconds()
    with_gap = examples(broken, package["anchors"], changed["reference"].index,
                        activity_features(broken))
    interior = (with_gap["baseline"].index > left) & (
        with_gap["baseline"].index < right)
    assert with_gap["baseline"].loc[interior].isna().all().all()
    assert not any(s in package["x"].columns for s in
                   ("heart_rate", "power", "cadence", "position_lat", "position_long", "ride"))
    checks.append(
        ride["name"] + ": hidden-HR feature/prediction invariance; original anchors exact; injected missing-record gap remains unsupported")


def select_method(metrics):
    """Apply predeclared outdoor and target-transfer acceptance gates once."""
    primary = metrics.loc[metrics.domain.eq(
        "outdoor") & metrics.condition.eq("point")]
    means = primary.groupby(["ride", "model"]).mean(numeric_only=True)
    mtb = metrics.loc[metrics.domain.eq("mtb_overlap")].set_index("model")
    decisions = {}
    for model in ("ML-A", "ML-B"):
        gates = []
        for ride in OUTDOOR:
            base, candidate = means.loc[(
                ride, "linear")], means.loc[(ride, model)]
            gates += [base.mae - candidate.mae >= .25, candidate.rmse <= base.rmse,
                      max(base.high_mae - candidate.high_mae, base.rise_mae - candidate.rise_mae,
                          base.recovery_mae - candidate.recovery_mae) >= .25,
                      candidate.peak_amplitude_mae <= base.peak_amplitude_mae + .5]
        for (_, _), group in primary.groupby(["ride", "phase"]):
            group = group.set_index("model")
            base, candidate = group.loc["linear"], group.loc[model]
            gates += [candidate.mae <= base.mae + .25,
                      candidate.step_p99 <= base.step_p99 + .5,
                      candidate.steps_gt5_percent <= base.steps_gt5_percent + .1,
                      candidate.unmatched_peaks <= base.unmatched_peaks + 2,
                      candidate.invalid_hr_n == 0]
        base, candidate = mtb.loc["linear"], mtb.loc[model]
        gates += [candidate.mae <= base.mae + .5, candidate.rmse <= base.rmse + .5,
                  candidate.high_mae <= base.high_mae + 1,
                  candidate.step_p99 <= base.step_p99 + .5,
                  candidate.steps_gt5_percent <= base.steps_gt5_percent + .1,
                  candidate.unmatched_peaks <= base.unmatched_peaks + 2,
                  candidate.invalid_hr_n == 0]
        decisions[model] = {"passed": int(sum(gates)), "total": len(gates),
                            "accepted": bool(all(gates))}
    accepted = [name for name in decisions if decisions[name]["accepted"]]
    selected = min(accepted, key=lambda m: primary.loc[primary.model.eq(
        m), "mae"].mean()) if accepted else "linear"
    return selected, decisions


def final_series(ride, watch, fitted, method):
    """Preserve all Edge observations and bridge from its last real observation."""
    frame = ride["frame"]
    edge = ride["reference"].reindex(frame.index)
    cutoff = ride["reference"].index[-1]
    # This boundary point is real Edge data, used only after overlap evaluation.
    tail = pd.concat([pd.Series(edge.loc[cutoff], index=pd.DatetimeIndex([cutoff])),
                      watch.loc[watch.index > cutoff]])
    package = examples(frame, tail, frame.index, activity_features(frame))
    candidates = predictions(package, fitted)
    ml_name = method if method in ("ML-A", "ML-B") else "ML-B"
    final = candidates[method].where(edge.isna(), edge)
    table = pd.DataFrame({"edge_hr": edge, "watch_hr": watch.reindex(frame.index),
                          "interpolation_baseline": candidates.linear,
                          "ml_estimate": candidates[ml_name], "final_hr": final})
    table["edge_observed"] = edge.notna()
    table["watch_observed"] = table.watch_hr.notna()
    table["estimated"] = final.notna() & edge.isna()
    table["source_status"] = "missing_unsupported"
    table.loc[table.estimated, "source_status"] = "estimated_between_anchors"
    table.loc[table.estimated & table.watch_observed,
              "source_status"] = "estimated_at_watch_anchor"
    table.loc[edge.notna(), "source_status"] = "recorded_edge"
    table["quality_indicator"] = np.where(
        frame.climb30.notna(), "terrain_supported", "terrain_missing")
    table.loc[frame.stationary_evidence,
              "quality_indicator"] += ";stationary_evidence"
    next_watch = watch.index[watch.index > cutoff][0]
    table.loc[cutoff:next_watch,
              "quality_indicator"] += ";cross_sensor_handoff"
    table.index.name = "timestamp_utc"
    assert len(table) == len(frame) and table.index.equals(frame.index)
    assert np.array_equal(table.loc[edge.notna(), "final_hr"], edge.dropna())
    assert np.array_equal(table.loc[watch.index, "watch_hr"], watch)
    assert table.loc[table.index > watch.index[-1], "final_hr"].isna().all()
    assert table.loc[table.index < cutoff,
                     "interpolation_baseline"].isna().all()
    assert np.isclose(table.interpolation_baseline.loc[cutoff + pd.Timedelta(seconds=1)],
                      edge.loc[cutoff] + (watch.loc[next_watch] - edge.loc[cutoff]) / (next_watch - cutoff).total_seconds())
    return table, cutoff, ml_name


def comparison_figure(panels, metrics, path):
    """Reuse fixed midpoint windows, without selecting by model error."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), constrained_layout=True)
    for ax, (ride, cached) in zip(axes[:2], panels):
        package, predicted, _ = cached[0, "point"]
        block = max(continuous_blocks(ride["reference"]), key=len)
        midpoint = block.index[0] + (block.index[-1] - block.index[0]) / 2
        start = (midpoint - pd.Timedelta(minutes=10)).floor("s")
        end = start + pd.Timedelta(minutes=20)
        actual = ride["reference"].loc[start:end]
        ax.plot(actual.index, actual, color=".65", lw=1.3,
                label="Recorded HR (hidden reference)")
        for name in ("linear", "ML-A", "ML-B"):
            values = predicted.loc[start:end, name]
            ax.plot(values.index, values, color=COLORS[name], lw=1.3,
                    ls="--" if name == "linear" else "-", label=name)
        sparse = package["anchors"].loc[start:end]
        ax.scatter(sparse.index, sparse, color=".15", s=23, zorder=5,
                   label="Simulated point observations")
        peak = actual.idxmax()
        ax.axvspan(peak, min(peak + pd.Timedelta(seconds=30), end),
                   color=".4", alpha=.08, label="30s after window maximum")
        ax.set(title=f"{ride['name']} — held-out, phase 0, fixed central 20 minutes",
               ylabel="HR (BPM)", xlabel="Local time — Europe/Brussels", xlim=(start, end))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZONE))
        ax.grid(alpha=.2)
        ax.legend(ncol=3, fontsize=8, loc="best")
    primary = metrics.loc[metrics.domain.eq(
        "outdoor") & metrics.condition.eq("point")]
    for i, name in enumerate(METHODS):
        groups = [primary.loc[primary.ride.eq(
            ride) & primary.model.eq(name)] for ride in OUTDOOR]
        mean = np.array([v.mae.mean() for v in groups])
        low = np.array([v.mae.min() for v in groups])
        high = np.array([v.mae.max() for v in groups])
        axes[2].bar(np.arange(2) + (i - 1.5) * .18, mean, .16,
                    color=COLORS[name], label=name,
                    yerr=[mean - low, high - mean], capsize=3)
    axes[2].set(xticks=[0, 1], xticklabels=list(OUTDOOR), ylabel="MAE (BPM)",
                title="Outdoor point observations: mean MAE and range over phases 0/20/40s")
    axes[2].legend(ncol=4)
    axes[2].grid(axis="y", alpha=.2)
    fig.suptitle("Gradient-boosted HR residual correction — offline, held-out cycling\n"
                 "One fixed algorithm; two feature configurations; recorded anchors preserved", fontsize=14)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def final_figure(ride, table, cutoff, method, path):
    """Distinguish recorded observations, estimates, gaps and probable stops."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True,
                             gridspec_kw={"height_ratios": [2.5, 1]})
    times = table.index
    for ax in axes:
        estimated = table.final_hr.where(table.estimated)
        ax.plot(times, estimated, color="#16759a", lw=1.1,
                label=f"Selected {method} estimate (not measured 1Hz HR)")
        ax.plot(times, table.edge_hr, color=".3",
                lw=1.2, label="Original Edge HR")
        observed = table.watch_hr.dropna()
        ax.scatter(observed.index, observed, color="#c17824", s=17,
                   zorder=5, label="Original watch summaries (instantaneous meaning uncertain)")
        ax.axvline(cutoff, color="#a73e3e", ls="--",
                   lw=1.2, label="Last recorded Edge HR")
        for indices in runs(ride["frame"].probable_stop, ride["frame"].index):
            ax.axvspan(times[indices[0]], times[indices[-1]],
                       color=".4", alpha=.10)
        ax.set(ylabel="HR (BPM)",
               xlabel="October 16, 2026 — Europe/Brussels (UTC+02:00)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZONE))
        ax.grid(alpha=.2)
    axes[0].set_xlim(times[0], times[-1])
    axes[0].set_title(
        "Complete ride; gray bands = 20 probable sustained stops; final 50s unsupported")
    axes[0].legend(loc="best", fontsize=8, ncol=2)
    axes[1].set_xlim(cutoff - pd.Timedelta(minutes=2),
                     cutoff + pd.Timedelta(minutes=3))
    visible = table.loc[cutoff - pd.Timedelta(minutes=2):cutoff + pd.Timedelta(minutes=3),
                        ["edge_hr", "watch_hr", "final_hr"]].stack()
    axes[1].set_ylim(visible.min() - 5, visible.max() + 5)
    axes[1].set_title(
        "Handoff detail: last real Edge point to next watch summary; no timestamp shift")
    fig.suptitle("October 16 MTB HR: recorded data plus selected reconstruction\n"
                 "Sparse summaries do not identify the actual missing short fluctuations", fontsize=14)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def value_range(group, field):
    """Summarize phase variation descriptively, not as independent replicates."""
    values = group[field]
    return f"{values.mean():.2f} [{values.min():.2f}, {values.max():.2f}]"


def reports(metrics, fits, selected, decisions, final, cutoff, ml_name, checks, protected_n):
    """Save measured evidence, the selection rule and final output semantics."""
    primary = metrics.loc[metrics.domain.eq(
        "outdoor") & metrics.condition.eq("point")]
    lines = ["# Gradient-boosted HR residual evaluation", "",
             f"**Final selection: {selected}.** "
             "The correction algorithm/configuration and linear base policy were fixed before held-out scoring. "
             "Model selection below applies the predefined gates once; no redesign or parameter tuning follows the scores.", "",
             "## Data and method", "",
             "Two outdoor FIT rides provide leave-one-ride-out evaluation (17,735 and 15,455 HR records). "
             "The 5,115-record virtual ride is a separate transfer check, never pooled into fitting. "
             "The target has 10,409 activity timestamps, 1,525 Edge HR observations, 158 watch summaries "
             "and 20 probable sustained stops. Reuse the existing official FIT decoder, UTC timestamps, "
             "sparse observation definitions, interpolation, and terrain/movement masks. No raw data or earlier output is changed.", "",
             "Base policy: **linear interpolation**, fixed for its simple bounded segments and conservative behavior. "
             "PCHIP remains an independent comparator, not a base chosen using the new held-out results. "
             "Maximum anchor gap 180s, no extrapolation, split at every unsupported reference/recording gap.", "",
             f"Installed scikit-learn: **{sklearn.__version__}**. "
             "[The version-matched official API](https://scikit-learn.org/1.1/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) "
             "documents native NaN routing. Missing terrain stays NaN, with a support indicator; no gradient is silently filled with zero. "
             "The existing installation is used; no dependency installation was necessary.", "",
             "`estimated_HR = linear_HR + correction`; the supervised target is `recorded_HR − linear_HR`. "
             "One HistGradientBoostingRegressor per ablation configuration, with the same fixed settings:", "",
             "```text", repr(PARAMETERS), "```", "",
             "Automatic early stopping is disabled, avoiding a random validation split of neighboring seconds. "
             "No parameter search, scaler or imputation is fitted on evaluation rides. Tree bins/NaN routes are learned on training data only. "
             "The script limits thread pools to two for predictable runtime.", "",
             "### Compact feature set", "",
             "- Anchor context (7): linear HR, neighboring-anchor HR difference, seconds since/until anchors, gap duration, "
             "interval fraction and linear slope (BPM/s). Both neighboring **observed** anchors are permitted in this offline task.",
             "- Movement (8): trailing speed5/speed30 (m/s), 10s change in speed5 (m/s²), stationary evidence, "
             "time in a stationary run, time since an observed restart (ages capped at 300s), moving fraction over 60s, uncertainty flag. "
             "Restart age is missing before a supported restart; observed run age is not a claim about unobserved motion. "
             "Stationary evidence includes supported short pauses; sustained-stop support is scarce in the road training data.",
             "- Terrain (5, ML-B only): positive/negative 30-sample climbing-rate magnitudes (m/s), their trailing 60s means, "
             "and current terrain support. Reuse centered least-squares rate after the 5-sample median, movement/gap splits "
             "and >2m/1s step caution masks. Full 60 valid samples are required for histories. "
             "Offline terrain is not delayed by the ODE experiment's causal lag. Prefer Strava elevation on MTB; roads use native altitude.",
             "- ML-A uses context+movement (15 inputs); ML-B adds terrain (20). Gradient is omitted because it is redundant "
             "with speed/climbing rate and has unvalidated MTB local slopes. No absolute position, activity identity, hidden HR, "
             "rolling dense HR, power, cadence, calories or load summaries enter inputs.", "",
             "### Training, postprocessing and evaluation", "",
             "Replay the exact 142×60s/15×120s interval sequence at phases 0/20/40s in elapsed UTC time. "
             "Point observations are primary. Averaged sensitivity uses complete observed [t−30,t+29] 60-second windows, "
             "without refitting the point-trained models. Its averages are explicitly treated as pseudo-point interpolation "
             "constraints for this sensitivity assumption, not established instantaneous measurements or a verified Garmin mechanism.", "",
             "Training retains only timestamps whose elapsed seconds are divisible by five, assigning each selected timestamp "
             "to exactly one phase cyclically. Thus the same hidden HR target is never triplicated across phases. "
             "Known anchors and unsupported rows are excluded. Targets are supervised training labels only; no hidden evaluation "
             "HR affects features or normalization. Fitted sample counts: "
             + "; ".join(f"{'+'.join(f['training'])}: {f['n']}" for f in fits) + ".", "",
             "Fixed stability rule: average predicted residuals over five centered samples **within each supported anchor interval**, "
             "then multiply by `4f(1−f)` (f=interval fraction). This tapers corrections to zero at observations; no settings "
             "are chosen from held-out curves, no residual clipping or extra ML algorithm is used. Point anchors are exact. "
             "For averaged/actual wellness summaries, passing through them is an explicit pseudo-point assumption, not recovery of true instantaneous HR.", "",
             "Every ride/phase uses the same intersection of point/averaged support and all four models, with all anchor times excluded. "
             "Coverage is relative to valid recorded reference HR, not elapsed gaps. High HR = reference upper decile; "
             "rise/recovery = ≥+5/≤−5 BPM over 30 consecutive observed seconds. Peaks: prominence≥10 BPM, separation≥120s; "
             "eligible ±15s windows must be completely scored. Peak amplitude is estimated local maximum minus reference peak; "
             "timing uses the earliest local maximum and is bounded/censored by ±15s. Unmatched predicted peaks have no reference peak within ±30s. "
             "These are score-only diagnostics, not model inputs. One-second HR records and phases are correlated.", "",
             "## Outdoor holdouts", "",
             "Equal-weight phase means [minimum, maximum], BPM. Bias also equals average-HR error on the same timestamps.", "",
             "| Held-out ride | Model | MAE | RMSE | Bias | High-HR MAE | Coverage % |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for (ride, model), group in primary.groupby(["ride", "model"], sort=False):
        lines.append(f"| {ride} | {model} | {value_range(group,'mae')} | {value_range(group,'rmse')} | "
                     f"{group.bias.mean():.2f} | {group.high_mae.mean():.2f} | {value_range(group,'coverage_percent')} |")
    lines += ["", "| Ride | Phase s | Linear MAE/RMSE | PCHIP MAE/RMSE | ML-A MAE/RMSE | ML-B MAE/RMSE |",
              "|---|---:|---:|---:|---:|---:|"]
    for (ride, phase), group in primary.groupby(["ride", "phase"]):
        group = group.set_index("model")
        lines.append(f"| {ride} | {phase} | " + " | ".join(
            f"{group.loc[m,'mae']:.2f}/{group.loc[m,'rmse']:.2f}" for m in METHODS) + " |")
    lines += ["", "### Short fluctuations, recovery, peaks and stability", "",
              "Roughness includes anchor neighborhoods and requires consecutive supported seconds; it never bridges a recording gap. "
              "Large changes are inspection flags, not physiologically validated maximum rates.", "",
              "| Ride | Model | Rise MAE | Recovery MAE | Peak amplitude abs error | Peak timing abs s | P99 abs 1s step | Unmatched peaks (phase mean) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (ride, model), group in primary.groupby(["ride", "model"], sort=False):
        lines.append(f"| {ride} | {model} | {group.rise_mae.mean():.2f} | {group.recovery_mae.mean():.2f} | "
                     f"{group.peak_amplitude_mae.mean():.2f} | {group.peak_timing_mae_s.mean():.2f} | "
                     f"{group.step_p99.mean():.3f} | {group.unmatched_peaks.mean():.1f} |")
    lines += ["", "| Ride | Model | Averaged-anchor MAE | Averaged-anchor RMSE |",
              "|---|---|---:|---:|"]
    averaged = metrics.loc[metrics.domain.eq(
        "outdoor") & metrics.condition.eq("averaged")]
    for (ride, model), group in averaged.groupby(["ride", "model"], sort=False):
        lines.append(
            f"| {ride} | {model} | {value_range(group,'mae')} | {value_range(group,'rmse')} |")
    lines += ["", "### Movement and terrain contexts", "",
              "Mean point-condition errors by audited context; n is the per-phase count range. "
              "Stationary means inferred stationary evidence (including short pauses), not proven rest. "
              "Restart means the first 30s after an observed restart; uphill/downhill requires supported "
              "climbing rate >+0.05/<−0.05m/s. These subsets overlap and do not identify pedaling effort. "
              "Undefined (nan) means no supported observations, not zero error.", "",
              "| Activity | Model | Stationary n / MAE | Restart n / MAE | Uphill MAE | Downhill MAE |",
              "|---|---|---|---|---:|---:|"]
    context = metrics.loc[metrics.condition.isin(["point", "actual_watch"])
                          & metrics.model.isin(["linear", "ML-B"])
                          & ~metrics.domain.eq("virtual_transfer")]
    for (ride, model), group in context.groupby(["ride", "model"], sort=False):
        lines.append(f"| {ride} | {model} | {group.stationary_n.min()}–{group.stationary_n.max()} / "
                     f"{group.stationary_mae.mean():.2f} | {group.restart_n.min()}–{group.restart_n.max()} / "
                     f"{group.restart_mae.mean():.2f} | {group.uphill_mae.mean():.2f} | {group.downhill_mae.mean():.2f} |")
    lines += ["", "Climb/descent errors improve on both outdoor rides, but stop/restart benefit is not consistent: "
              "road ride 1 restart error increases on its very small supported subset, and MTB stationary error is slightly worse. "
              "Overall acceptance does not establish accurate reconstruction of the long unobserved MTB stops.", "",
              "## Separate transfer checks", "",
              "Both-road fits are frozen before MTB scoring. The MTB overlap uses real watch observations only, without "
              "an Edge boundary anchor or any hidden Edge input. It ends before the 20:34 altitude anomaly; watch/Edge path differences "
              "confound interpretation and this previously examined interval is not a pristine independent test. "
              "Virtual speed/terrain are simulated and the virtual ride never trains a model.", "",
              "| Domain / observation | Model | MAE | RMSE | Bias | High-HR MAE | Scored n | P99 1s step |",
              "|---|---|---:|---:|---:|---:|---|---:|"]
    for (domain, condition, model), group in metrics.loc[~metrics.domain.eq("outdoor")].groupby(
            ["domain", "condition", "model"], sort=False):
        lines.append(f"| {domain} / {condition} | {model} | {value_range(group,'mae')} | {value_range(group,'rmse')} | "
                     f"{group.bias.mean():.2f} | {group.high_mae.mean():.2f} | {group.n.min()}–{group.n.max()} | {group.step_p99.mean():.3f} |")
    lines += ["", "## Conservative selection", "",
              "Predefined gates: on **each** outdoor ride, ≥0.25 BPM mean MAE improvement over linear, no mean RMSE worsening, "
              "≥0.25 BPM improvement in high-HR, rise or recovery MAE, and peak-amplitude error no worse by >0.5 BPM. "
              "Each phase must avoid MAE degradation >0.25 BPM, P99 1s-step increase >0.5 BPM, >5 BPM/s step-frequency increase "
              ">0.1 percentage points, >2 extra unmatched peaks, or estimates outside 0–255 BPM. "
              "MTB veto: no MAE/RMSE degradation >0.5 BPM, high-HR degradation >1 BPM, and the same stability/peak/range checks. "
              "These are operational conservative screens, not significance tests. Among configurations passing every gate, "
              "choose lower mean outdoor MAE; otherwise retain the predefined linear baseline.", "",
              "| Configuration | Gates passed | Accepted |", "|---|---:|---|"]
    for name, decision in decisions.items():
        lines.append(
            f"| {name} | {decision['passed']}/{decision['total']} | {decision['accepted']} |")
    paired = primary.pivot(index=["ride", "phase"],
                           columns="model", values="mae")
    lines += ["", f"ML-B improves MAE over ML-A in {int((paired['ML-B']<paired['ML-A']).sum())}/6 outdoor phases. "
              f"ML-A improves on linear in {int((paired['ML-A']<paired.linear).sum())}/6; "
              f"ML-B in {int((paired['ML-B']<paired.linear).sum())}/6. These counts alone do not justify deployment. "
              f"**Selected final method: {selected}.** "
              + ("ML did not establish the complete predefined accuracy/dynamic/stability/transfer advantage; "
                 "its extra detail is not treated as recovered physiology. The project is completed with interpolation."
                 if selected == "linear" else
                 "The selected correction passes the predefined descriptive screens, but two roads and one short target overlap "
                 "cannot establish generalization or recover the unknown true MTB fluctuations."), "",
              "## Limitations and reproducibility", "",
              "Road rides have few short stationary examples and no sustained stops, whereas MTB has about 56 minutes of stops. "
              "Native NaN routing and state-age features do not solve that domain mismatch. Optical sensor artifacts, uncertain "
              "wellness averaging, sparse-anchor timing, imperfect terrain, absent power/cadence and only two independent outdoor "
              "rides constrain interpretation. A smaller overall error cannot establish accurate unseen short peaks. "
              "No hyperparameters or smoothing settings were changed after held-out scoring.", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe scripts/evaluate_hr_ml_residual.py", "```", "",
              "The comparison figure uses phase-0 point observations and the central 20 minutes of each ride's longest continuous "
              "HR block. Shading marks 30s after the window maximum; window selection never uses model error. "
              "All phase/subset counts and signed peak errors are in `data/processed/hr_ml_residual_metrics.csv`.", "",
              "Checks executed:", ""]
    lines += ["- " + check for check in checks]
    lines += [f"- All {protected_n} protected source/prior-analysis hashes unchanged; no original FIT/GPX/HR output overwritten.",
              "- Non-empty requested artifacts; matched timestamp metrics, bias/mean-HR identity, and finite-range checks. "
              "Visual readability is inspected after generation. No separate test suite, commits, pushes or uploads.", ""]
    (ROOT / "docs" / (STEM + ".md")).write_text("\n".join(lines), encoding="utf-8")
    summary = {"mean": final.final_hr.mean(), "peak": final.final_hr.max(),
               "n": int(final.final_hr.notna().sum()), "coverage": 100 * final.final_hr.notna().mean(),
               "estimated_n": int(final.estimated.sum()),
               "estimated_mean": final.loc[final.estimated, "final_hr"].mean()}
    estimated_steps = final.final_hr.diff().where(final.estimated).abs().dropna()
    assert np.isclose(final.final_hr.sum() / summary["n"], summary["mean"])
    assert summary["n"] == int(
        final.edge_observed.sum() + final.estimated.sum())
    final_text = ["# Final October 16, 2026 MTB HR reconstruction", "",
                  f"**Selected method: {selected}.** See [the ML evaluation](hr_ml_residual_evaluation.md) for "
                  "the outdoor holdouts, target overlap and fixed selection criteria.", "",
                  f"- Mean of the **supported combined recorded/estimated sequence**: **{summary['mean']:.2f} BPM**.",
                  f"- Supported peak: **{summary['peak']:.2f} BPM**; missing short peaks cannot be recovered or ruled out.",
                  f"- Coverage: **{summary['n']}/{len(final)} timestamps ({summary['coverage']:.2f}%)**.",
                  f"- Original Edge observations: {final.edge_observed.sum()}, preserved exactly. Estimated rows: "
                  f"{summary['estimated_n']}, mean {summary['estimated_mean']:.2f} BPM. Original watch summaries: "
                  f"{final.watch_observed.sum()}, preserved separately and exactly.", "",
                  "These are descriptive statistics of supported one-second rows, not a measured full-ride physiological "
                  "mean or guaranteed true peak. The last 50 seconds remain missing because there is no following HR anchor. "
                  "No personalized HR zones are invented.", "",
                  f"Continuity check on estimated rows and the handoff: P99 absolute 1s change "
                  f"{estimated_steps.quantile(.99):.3f} BPM, maximum {estimated_steps.max():.3f} BPM, "
                  f"{estimated_steps.gt(5).sum()} changes >5 BPM. The combined sequence's larger "
                  f"{final.final_hr.diff().abs().max():.1f} BPM maximum step belongs to original preserved "
                  "Edge data, not an injected reconstruction jump. Numerical continuity does not validate true short fluctuations.", "",
                  "## Observations, handoff and provenance", "",
                  "Sources: `data/mtb_edge.fit`, `data/VTT_Apero.gpx`, "
                  "`data/processed/watch_hr_ride_2026-10-16.csv`. Original Edge timestamps define the exact 1Hz grid; "
                  "all calculations retain floating-point precision, UTC internally and Brussels time in the plot.", "",
                  f"Last recorded Edge HR: {cutoff.isoformat()} (19:57:46 Brussels), 101 BPM. "
                  "After overlap evaluation and model selection, this **real** observation becomes the left boundary for "
                  "the first reconstructed interval, with the next real watch summary (94 BPM, 14s later). "
                  "This prevents an arbitrary sensor-switch jump without fabricating an anchor, shifting time, calibrating "
                  "the watch, or modifying recorded values. The handoff remains an explicitly flagged cross-sensor interval. "
                  "No withheld Edge observation was used during the overlap evaluation.", "",
                  "After disconnection, interpolation uses original watch summaries with a maximum 180s gap, never extrapolating. "
                  "Their storage/averaging mechanism is uncertain: treating them as pseudo-point constraints is a practical "
                  "baseline assumption, not evidence of exact instantaneous HR. Even at a watch timestamp, a final value after "
                  "disconnection is labelled an **estimate**. When Edge/watch coincide, Edge retains precedence and the watch "
                  "observation stays unchanged in its separate column. No stop sets HR to zero or resets history.", "",
                  "CSV fields: `timestamp_utc`, `edge_hr`, `watch_hr`, `interpolation_baseline`, `ml_estimate`, `final_hr`, "
                  "observation/estimate booleans, `source_status`, and `quality_indicator`. "
                  f"`ml_estimate` contains {ml_name} as a "
                  + ("rejected diagnostic alternative, **not** the selected output." if selected ==
                     "linear" else "comparison/selected correction.")
                  + " Baseline/ML columns are post-cutoff candidates; they are intentionally empty before the last Edge observation. "
                  "Quality labels mark terrain support, stationary evidence and the cross-sensor handoff; supported terrain "
                  "does not certify HR accuracy. Missing HR is never encoded as zero.", "",
                  "Status values: `recorded_edge`, `estimated_at_watch_anchor`, `estimated_between_anchors`, "
                  "`missing_unsupported`. Separate `watch_observed` preserves the distinction between an actual stored "
                  "summary and a modelled instantaneous HR value.", "",
                  "## Reproduction and checks", "", "```powershell",
                  ".\\.venv\\Scripts\\python.exe scripts/evaluate_hr_ml_residual.py", "```", "",
                  "Outputs: `data/processed/hr_reconstructed_2026-10-16.csv`, "
                  "`docs/images/hr_reconstructed_2026-10-16.png`, this report. "
                  "Verified exact Edge grid, original Edge/watch preservation, supported handoff, missing terminal tail, "
                  "mean/count identities and unchanged source/prior-analysis hashes. Figures are inspected after generation. "
                  "This work was developed with AI assistance; empirical results come from the executed script, "
                  "not inferred or fabricated physiology. No activity FIT is modified or uploaded.", ""]
    (ROOT / "docs/hr_final_reconstruction_2026-10-16.md").write_text(
        "\n".join(final_text), encoding="utf-8")
    return summary


def main():
    """Run fixed holdouts, freeze transfer fits, select once and finish the ride."""
    new_names = {"evaluate_hr_ml_residual.py", STEM + ".md", STEM + ".png",
                 "hr_ml_residual_metrics.csv", "hr_reconstructed_2026-10-16.csv",
                 "hr_reconstructed_2026-10-16.png", "hr_final_reconstruction_2026-10-16.md"}
    protected = [p for folder in ("data", "docs", "scripts")
                 for p in (ROOT / folder).rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts and p.name not in new_names]
    before = fingerprints(protected)
    watch, _, _ = clean_observations(pd.read_csv(
        ROOT / "data/processed/watch_hr_ride_2026-10-16.csv"))
    pattern = np.diff(watch.index.asi8) // 1_000_000_000
    assert len(watch) == 158 and (pattern == 60).sum(
    ) == 142 and (pattern == 120).sum() == 15
    rides = [prepare(name) for name in OUTDOOR]
    packages = [synthetic(ride, pattern, activity_features(
        ride["frame"])) for ride in rides]
    rows, fits, panels, checks = [], [], [], []
    for held, train in ((0, 1), (1, 0)):
        print("Training", rides[train]["name"],
              "holding out", rides[held]["name"], flush=True)
        fitted = fit([(rides[train], packages[train])])
        fits.append(fitted)
        values, cached = evaluate(
            rides[held], packages[held], fitted, "outdoor", checks)
        rows.extend(values)
        panels.append((rides[held], cached))
    fitted = fit(list(zip(rides, packages)))
    fits.append(fitted)
    virtual = prepare("virtual_road_ride_1")
    virtual_packages = synthetic(
        virtual, pattern, activity_features(virtual["frame"]))
    values, _ = evaluate(virtual, virtual_packages,
                         fitted, "virtual_transfer", checks)
    rows.extend(values)
    mtb = prepare("mtb_edge")
    mtb["frame"] = mtb["strava"]
    overlap = watch.loc[mtb["reference"].index[0]:mtb["reference"].index[-1]]
    package = examples(
        mtb["frame"], overlap, mtb["reference"].index, activity_features(mtb["frame"]))
    values, _ = evaluate(
        mtb, {(0, "actual_watch"): package}, fitted, "mtb_overlap", checks)
    rows.extend(values)
    metrics = pd.DataFrame(rows)
    assert len(metrics) == 76
    assert metrics.groupby(
        ["domain", "ride", "phase", "condition"]).n.nunique().eq(1).all()
    assert np.isfinite(metrics[["mae", "rmse", "bias"]]).all().all()
    assert metrics.rmse.ge(metrics.mae - 1e-9).all()
    selected, decisions = select_method(metrics)
    final, cutoff, ml_name = final_series(mtb, watch, fitted, selected)
    checks += ["Training-only binning, fixed parameters/linear base, no early-stopping split; phase examples deduplicated by timestamp",
               "All methods/conditions share hidden score times per phase; averaging windows complete; bias equals mean-HR error",
               "Exact Edge grid/HR and all 158 watch summaries preserved; real boundary handoff; no unsupported extrapolation"]
    metrics.to_csv(
        ROOT / "data/processed/hr_ml_residual_metrics.csv", index=False)
    final.to_csv(ROOT / "data/processed/hr_reconstructed_2026-10-16.csv")
    comparison_figure(panels, metrics, ROOT / "docs/images" / (STEM + ".png"))
    final_figure(mtb, final, cutoff, selected, ROOT /
                 "docs/images/hr_reconstructed_2026-10-16.png")
    assert fingerprints(protected) == before
    summary = reports(metrics, fits, selected, decisions,
                      final, cutoff, ml_name, checks, len(protected))
    for name in new_names - {"evaluate_hr_ml_residual.py"}:
        matches = list((ROOT / "data/processed").glob(name)) + \
            list((ROOT / "docs").rglob(name))
        assert len(matches) == 1 and matches[0].stat().st_size > 0
    assert fingerprints(protected) == before
    print(metrics.loc[metrics.condition.isin(["point", "actual_watch"])].groupby(
        ["domain", "ride", "model"])[["mae", "rmse", "bias"]].mean().round(3).to_string())
    print("Selection:", selected, decisions)
    print("Final:", summary)
    print("Protected files unchanged:", len(protected))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
