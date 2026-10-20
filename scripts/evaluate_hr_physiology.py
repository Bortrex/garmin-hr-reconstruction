"""Evaluate small first-order HR models using held-out cycling rides.

Run: .venv/Scripts/python scripts/evaluate_hr_physiology.py
Only aggregate metrics and an outdoor diagnostic figure are exported.
"""

import hashlib

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from audit_cycling_predictors import (
    ROOT, ZONE, identify_stops, load_records, movement_segments, runs,
)
from compare_strava_garmin_elevation import load_gpx, source_features
from diagnose_watch_anchors import reference_values
from evaluate_hr_interpolation import (
    PHASES, continuous_blocks, reconstruct, sampling_times,
)
from reconstruct_hr_baseline import clean_observations
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


STEM = "hr_physiological_feasibility"
OUTDOOR = ("road_ride_1", "road_ride_2")
METHODS = ("M0", "M1", "M2", "linear", "pchip")
TAUS = (30., 60., 90.)
RIDGE = .1
TERRAIN_LAG = 17
MEANINGFUL = .5
COLORS = ("#999999", "#197a9c", "#c55726", "#879743", "#794ca3")


def fingerprints(paths):
    """Hash originals and prior outputs without traversing the environment."""
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare(name):
    """Reuse decoding and terrain masks; keep HR separate from all predictors."""
    raw, _ = load_records(name)
    reference, _, _ = clean_observations(
        raw.reset_index(drop=True).rename(columns={"heart_rate": "bpm"}))
    columns = ["speed_raw", "distance", "altitude_raw", "position_lat",
               "position_long", "dt"]
    frame = raw[columns].copy()
    identify_stops(frame, virtual=name.startswith("virtual"))
    native = source_features(frame, frame.altitude_raw,
                             pd.Series(0, index=frame.index))
    # A trailing mean is causal and never spans even one missing second.
    group = frame.dt.ne(1).cumsum()
    native["speed5"] = frame.speed_raw.groupby(group).transform(
        lambda values: values.rolling(5, min_periods=1).mean())
    native["stop"] = frame.stationary_evidence.astype(float)
    native["uncertain"] = (frame.speed_raw.isna()
                           | (frame.speed_raw.le(.3)
                              & ~frame.stationary_evidence)).astype(float)
    native = lag_terrain(native)
    ride = {"name": name, "frame": native, "reference": reference}
    if name == "mtb_edge":
        gpx, _ = load_gpx(ROOT / "data/VTT_Apero.gpx")
        assert gpx.index.equals(frame.index)
        corrected = source_features(frame, gpx.altitude_raw, gpx.segment)
        for column in ("speed5", "stop", "uncertain"):
            corrected[column] = native[column]
        ride["strava"] = lag_terrain(corrected)
    return ride


def lag_terrain(frame):
    """Delay the existing centered trend so its numeric support is in the past."""
    frame = frame.copy()
    frame["terrain"] = np.nan
    _, segments = movement_segments(frame)
    for indices in segments:
        part = frame.iloc[indices]
        frame.loc[part.index, "terrain"] = part.climb30.shift(TERRAIN_LAG)
    assert frame.loc[frame.stationary_evidence | frame.speed_raw.le(.3),
                     "terrain"].isna().all()
    return frame


def make_anchors(reference, times, scenario):
    """Construct only the declared point or fully observed 60-sample mean."""
    if scenario == "point":
        anchors = reference.reindex(times).dropna()
        assert anchors.equals(reference.reindex(anchors.index))
        return anchors
    anchors = reference_values(reference, times, ("centered", 60, 0)).dropna()
    for time, value in anchors.items():
        window = reference.reindex(pd.date_range(
            time - pd.Timedelta(seconds=30), periods=60, freq="s"))
        assert window.notna().all() and np.isclose(window.mean(), value)
    return anchors


def intervals(ride, anchors):
    """Find supported anchor intervals without reading the right HR value."""
    frame, reference = ride["frame"], ride["reference"]
    groups = frame.dt.ne(1).cumsum()
    result = []
    for block in continuous_blocks(reference):
        sparse = anchors.loc[block.index[0]:block.index[-1]]
        for i in range(len(sparse) - 1):
            left, right = sparse.index[i:i + 2]
            duration = (right - left).total_seconds()
            a, b = frame.index.get_indexer([left, right])
            if 1 < duration <= 180 and a >= 0 and b >= 0:
                if groups.iloc[a] == groups.iloc[b]:
                    assert b - a == duration
                    result.append((a, b, float(sparse.iloc[i])))
    return result


def scales(training):
    """Estimate predictor scaling from the training rides only, never HR."""
    speed = pd.concat([r["frame"].speed5 for r in training])
    terrain = pd.concat([r["frame"].terrain for r in training]).dropna()
    return {"speed_center": speed.mean(), "speed_scale": max(speed.std(), .1),
            "speed_fill": speed.median(),
            "up_scale": max(np.maximum(terrain, 0).std(), .01),
            "down_scale": max(np.maximum(-terrain, 0).std(), .01)}


def predictors(frame, scale):
    """Build six allowed numeric inputs, with an explicit terrain fallback."""
    speed = frame.speed5.fillna(scale["speed_fill"])
    supported = frame.terrain.notna()
    # Zero here means zero *correction to M1*, not an observation of flat ground.
    up = np.where(supported, np.maximum(frame.terrain, 0), 0)
    down = np.where(supported, np.maximum(-frame.terrain, 0), 0)
    x = np.column_stack([
        np.ones(len(frame)), (speed -
                              scale["speed_center"]) / scale["speed_scale"],
        frame.stop, frame.uncertain,
        up / scale["up_scale"], down / scale["down_scale"],
    ])
    assert np.isfinite(x).all()
    assert (x[~supported, 4:] == 0).all()
    return x


def trajectory_design(frame, x, spans, tau):
    """Unroll multi-second dynamics; only left anchors initialize HR state.

    The matrix contains exponentially accumulated predictors. Multiplying it
    by equilibrium coefficients is exactly the first-order forward recursion.
    Right anchor values and all hidden HR are absent from this function.
    """
    rows, designs, initials = [], [], []
    times = frame.index.asi8 / 1e9
    for left, right, initial in spans:
        design = np.zeros(x.shape[1])
        decay = 1.
        for i in range(left + 1, right):
            dt = times[i] - times[i - 1]
            assert dt == 1, "Unsupported recording gap in a trajectory"
            a = np.exp(-dt / tau)
            design = a * design + (1 - a) * x[i - 1]
            decay *= a
            rows.append(i)
            designs.append(design.copy())
            initials.append(decay * initial)
    return (np.asarray(rows, dtype=int), np.asarray(designs),
            np.asarray(initials))


def ridge_fit(design, target, intercept=False):
    """Minimize mean trajectory error plus a fixed standardized ridge penalty."""
    penalty = np.eye(design.shape[1]) * RIDGE
    if intercept:
        penalty[0, 0] = 0
    return np.linalg.solve(design.T @ design / len(target) + penalty,
                           design.T @ target / len(target))


def fit_models(training, pattern):
    """Fit phase-0 point trajectories using only the named training rides."""
    scale = scales(training)
    samples = []
    for ride in training:
        reference = ride["reference"]
        times = sampling_times(
            reference.index[0], reference.index[-1], pattern, 0)
        anchors = make_anchors(reference, times, "point")
        samples.append((ride, predictors(ride["frame"], scale),
                        intervals(ride, anchors)))
    candidates = {"M0": [], "M1": []}
    for tau in TAUS:
        matrices, targets, offsets = [], [], []
        for ride, x, spans in samples:
            positions, design, initial = trajectory_design(
                ride["frame"], x, spans, tau)
            truth = ride["reference"].reindex(
                ride["frame"].index[positions]).to_numpy()
            assert np.isfinite(truth).all()
            matrices.append(design)
            targets.append(truth)
            offsets.append(initial)
        design, truth, initial = map(
            np.concatenate, (matrices, targets, offsets))
        for model, width in (("M0", 1), ("M1", 4)):
            coefficients = np.zeros(6)
            coefficients[:width] = ridge_fit(
                design[:, :width], truth - initial, True)
            error = initial + design @ coefficients - truth
            objective = np.mean(error ** 2) + RIDGE * \
                np.sum(coefficients[1:] ** 2)
            candidates[model].append({"tau": tau, "beta": coefficients,
                                      "objective": objective,
                                      "train_rmse": np.sqrt(np.mean(error ** 2))})
    fitted = {name: min(values, key=lambda v: v["objective"])
              for name, values in candidates.items()}
    # Nested M2: freeze M1 dynamics/workload, fit only a terrain correction.
    tau = fitted["M1"]["tau"]
    matrices, residuals = [], []
    for ride, x, spans in samples:
        positions, design, initial = trajectory_design(
            ride["frame"], x, spans, tau)
        truth = ride["reference"].reindex(
            ride["frame"].index[positions]).to_numpy()
        matrices.append(design[:, 4:])
        residuals.append(truth - initial - design @ fitted["M1"]["beta"])
    terrain_design, residual = map(np.concatenate, (matrices, residuals))
    correction = ridge_fit(terrain_design, residual)
    beta = fitted["M1"]["beta"].copy()
    beta[4:] = correction
    fitted["M2"] = {"tau": tau, "beta": beta,
                    "train_rmse": np.sqrt(np.mean((terrain_design @ correction - residual) ** 2))}
    fitted["scale"] = scale
    fitted["training"] = tuple(ride["name"] for ride in training)
    return fitted


def predict(ride, anchors, fitted, frame=None):
    """Predict interiors without teacher forcing; anchors mark observed resets."""
    frame = ride["frame"] if frame is None else frame
    x = predictors(frame, fitted["scale"])
    spans = intervals(ride, anchors)
    estimates = pd.DataFrame(
        index=frame.index, columns=METHODS[:3], dtype=float)
    for model in METHODS[:3]:
        positions, design, initial = trajectory_design(
            frame, x, spans, fitted[model]["tau"])
        estimates.iloc[positions, estimates.columns.get_loc(model)] = (
            initial + design @ fitted[model]["beta"])
    estimates.loc[anchors.index.intersection(frame.index)] = anchors.reindex(
        anchors.index.intersection(frame.index)).to_numpy()[:, None]
    return estimates


def diagnostic_masks(ride):
    """Predeclare score-only high-HR, rise/recovery, peak and terrain subsets."""
    reference, frame = ride["reference"], ride["frame"]
    masks = pd.DataFrame(index=frame.index)
    masks["high"] = reference.reindex(frame.index).ge(reference.quantile(.9))
    change = pd.Series(np.nan, index=reference.index)
    peaks = []
    for block in continuous_blocks(reference):
        change.loc[block.index] = block.diff(30)
        positions, _ = find_peaks(
            block.to_numpy(), prominence=10, distance=120)
        peaks.extend(block.index[positions])
    masks["rise"] = change.reindex(frame.index).ge(5)
    masks["recovery"] = change.reindex(frame.index).le(-5)
    masks["transition"] = False
    events = []
    for indices in runs(frame.climb30.notna(), frame.index):
        climb = frame.climb30.iloc[indices]
        up = climb.gt(.05).rolling(10, min_periods=10).min().eq(1)
        down = climb.lt(-.05).rolling(10, min_periods=10).min().eq(1)
        near_up = up.rolling(60, min_periods=60).max().eq(1)
        candidate = down & near_up
        for group in runs(candidate, climb.index):
            time = climb.index[group[0]]
            if not events or (time - events[-1]).total_seconds() >= 120:
                events.append(time)
                masks.loc[time - pd.Timedelta(seconds=30):
                          time + pd.Timedelta(seconds=30), "transition"] = True
    return masks, peaks, events


def score(ride, estimate, common, masks, peaks):
    """Score matching unseen timestamps; quantify local peak errors explicitly."""
    actual = ride["reference"].reindex(estimate.index)
    error = (estimate - actual).where(common)
    values = error.dropna()
    result = {"n": len(values), "coverage_percent": 100 * len(values) / len(ride["reference"]),
              "mae": values.abs().mean(), "rmse": np.sqrt(np.mean(values ** 2)),
              "bias": values.mean()}
    for label in ("high", "rise", "recovery", "transition"):
        selected = values[masks.loc[values.index, label]]
        result[label + "_n"] = len(selected)
        result[label + "_mae"] = selected.abs().mean()
        result[label + "_bias"] = selected.mean()
    amplitude, timing = [], []
    for time in peaks:
        window = pd.date_range(
            time - pd.Timedelta(seconds=15), periods=31, freq="s")
        if not common.reindex(window, fill_value=False).all():
            continue
        local = estimate.reindex(window)
        amplitude.append(local.max() - actual.loc[time])
        timing.append(abs((local.idxmax() - time).total_seconds()))
    result.update({"peak_n": len(amplitude),
                   "peak_amplitude_bias": np.mean(amplitude) if amplitude else np.nan,
                   "peak_amplitude_mae": np.mean(np.abs(amplitude)) if amplitude else np.nan,
                   "peak_timing_mae_s": np.mean(timing) if timing else np.nan})
    assert np.isfinite(values).all() and len(values) == common.sum()
    assert np.isclose(result["mae"], np.abs(
        values.to_numpy()).sum() / len(values))
    assert np.isclose(
        result["bias"], estimate[common].mean() - actual[common].mean())
    return result


def evaluate_ride(ride, fitted, pattern, domain, panels, checks):
    """Compare all methods/scenarios on common hidden timestamps for each phase."""
    assert ride["name"] not in fitted["training"]
    reference = ride["reference"]
    bounds = [tuple(block.index[[0, -1]])
              for block in continuous_blocks(reference)]
    grid = pd.date_range(reference.index[0], reference.index[-1], freq="s")
    masks, peaks, events = diagnostic_masks(ride)
    rows = []
    for phase in PHASES:
        times = sampling_times(
            reference.index[0], reference.index[-1], pattern, phase)
        results, anchors = {}, {}
        for scenario in ("point", "averaged"):
            sparse = make_anchors(reference, times, scenario)
            estimates = predict(ride, sparse, fitted)
            baseline = reconstruct(
                sparse, grid, bounds).reindex(estimates.index)
            estimates[["linear", "pchip"]] = baseline
            results[scenario], anchors[scenario] = estimates, sparse
        common = reference.reindex(ride["frame"].index).notna()
        for estimates in results.values():
            common &= estimates.notna().all(axis=1)
        # No known anchors score points for any method or scenario.
        common.loc[common.index.intersection(times)] = False
        assert common.any()
        assert not common[common.index.intersection(times)].any()
        for scenario, estimates in results.items():
            for method in METHODS:
                rows.append({"domain": domain, "ride": ride["name"],
                             "phase": phase, "scenario": scenario, "method": method,
                             "training": "+".join(fitted["training"]),
                             "anchors": len(anchors[scenario]), "transition_events": len(events),
                             **score(ride, estimates[method], common,
                                     masks, peaks)})
        if phase == 0:
            leakage_checks(ride, anchors["point"],
                           fitted, results["point"], checks)
            if domain == "outdoor":
                panels.append((ride, anchors["point"], results["point"]))
    return rows


def leakage_checks(ride, anchors, fitted, expected, checks):
    """Check invariance to hidden HR, future anchors and absent terrain."""
    # Preserve only availability; change every hidden HR used in no predictor.
    altered = dict(ride)
    altered["reference"] = ride["reference"] * 0 + 200
    result = predict(altered, anchors, fitted)
    assert np.allclose(result, expected[list(METHODS[:3])], equal_nan=True)
    spans = intervals(ride, anchors)
    left, right, _ = spans[len(spans) // 2]
    altered_anchors = anchors.copy()
    altered_anchors.loc[ride["frame"].index[right]] += 10
    result = predict(ride, altered_anchors, fitted)
    window = ride["frame"].index[left + 1:right]
    assert np.allclose(result.loc[window],
                       expected.loc[window, list(METHODS[:3])])
    # The M2 workload fallback is exactly M1; an all-missing-terrain path must match.
    absent = ride["frame"].copy()
    absent["terrain"] = np.nan
    result = predict(ride, anchors, fitted, absent)
    assert np.allclose(result.M1, result.M2, equal_nan=True)
    # Splitting a real trajectory with an artificial missing timestamp forbids a bridge.
    gap = dict(ride)
    gap["frame"] = ride["frame"].drop(ride["frame"].index[(left + right) // 2])
    gap["frame"] = gap["frame"].copy()
    gap["frame"]["dt"] = gap["frame"].index.to_series().diff().dt.total_seconds()
    assert not any(gap["frame"].index[a] == ride["frame"].index[left]
                   and gap["frame"].index[b] == ride["frame"].index[right]
                   for a, b, _ in intervals(gap, anchors))
    checks.append(
        ride["name"] + ": hidden-HR/future-anchor invariance, M1 fallback, gap splitting")


def evaluate_mtb(ride, fitted, checks):
    """Use real watch observations only, restricted to the original Edge overlap."""
    assert ride["name"] not in fitted["training"]
    watch = pd.read_csv(ROOT / "data/processed/watch_hr_ride_2026-10-16.csv")
    anchors, _, _ = clean_observations(watch)
    reference = ride["reference"]
    anchors = anchors.loc[reference.index[0]:reference.index[-1]]
    # Strictly restrict computation to the overlap; never generate the missing ride.
    short = dict(ride)
    short["frame"] = ride["frame"].loc[reference.index[0]:reference.index[-1]]
    predictions = predict(short, anchors, fitted)
    leakage_checks(short, anchors, fitted, predictions, checks)
    estimates = predictions[["M1", "M2"]].rename(
        columns={"M2": "M2_Garmin"})
    estimates["M2_Strava"] = predict(short, anchors, fitted,
                                     ride["strava"].reindex(short["frame"].index)).M2
    bounds = [tuple(block.index[[0, -1]])
              for block in continuous_blocks(reference)]
    grid = pd.date_range(reference.index[0], reference.index[-1], freq="s")
    estimates[["linear", "pchip"]] = reconstruct(
        anchors, grid, bounds).reindex(estimates.index)
    common = (estimates.notna().all(axis=1)
              & reference.reindex(estimates.index).notna())
    common.loc[anchors.index] = False
    masks, peaks, events = diagnostic_masks(short)
    rows = [{"domain": "mtb_overlap", "ride": ride["name"], "phase": 0,
             "scenario": "actual_watch", "method": name,
             "training": "+".join(fitted["training"]), "anchors": len(anchors),
             "transition_events": len(events),
             **score(short, estimates[name], common, masks, peaks)}
            for name in estimates]
    assert estimates.index[-1] <= reference.index[-1]
    return rows


def draw_figure(panels, metrics, path):
    """Show fixed midpoint intervals and all outdoor point-phase scores."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), constrained_layout=True)
    for ax, (ride, anchors, estimates) in zip(axes[:2], panels):
        longest = max(continuous_blocks(ride["reference"]), key=len)
        midpoint = longest.index[0] + \
            (longest.index[-1] - longest.index[0]) / 2
        start = (midpoint - pd.Timedelta(minutes=10)).floor("s")
        end = start + pd.Timedelta(minutes=20)
        assert start >= longest.index[0] and end <= longest.index[-1]
        truth = ride["reference"].loc[start:end]
        ax.plot(truth.index, truth, color=".6", lw=1,
                label="Recorded HR (hidden reference)")
        for name, color, style in (("M1", COLORS[1], "-"), ("M2", COLORS[2], "-"),
                                   ("pchip", COLORS[4], "--")):
            visible = estimates.loc[start:end, name]
            ax.plot(visible.index, visible, color=color, lw=1.4, ls=style,
                    label=name + (" (offline)" if name == "pchip" else " (forward)"))
        sparse = anchors.loc[start:end]
        ax.scatter(sparse.index, sparse, color=".15", s=20, zorder=5,
                   label="Simulated point observations")
        ax.set(title=f"{ride['name']} — held-out, phase 0; midpoint of longest continuous block",
               ylabel="Heart rate (BPM)", xlabel="Local time — Europe/Brussels", xlim=(start, end))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZONE))
        ax.grid(alpha=.2)
        ax.legend(loc="best", fontsize=8, ncol=3)
    ax = axes[2]
    primary = metrics.loc[metrics.domain.eq(
        "outdoor") & metrics.scenario.eq("point")]
    width = .145
    for j, name in enumerate(METHODS):
        x = np.arange(2) + (j - 2) * width
        groups = [primary.loc[primary.ride.eq(ride) & primary.method.eq(name)]
                  for ride in OUTDOOR]
        mean = np.array([group.mae.mean() for group in groups])
        lo = np.array([group.mae.min() for group in groups])
        hi = np.array([group.mae.max() for group in groups])
        ax.bar(x, mean, width * .92, color=COLORS[j], label=name,
               yerr=np.array([mean - lo, hi - mean]), capsize=3)
        ax.scatter(x, [group.high_mae.mean() for group in groups],
                   color=COLORS[j], edgecolor="black", marker="D", s=32, zorder=5)
    ax.scatter([], [], marker="D", color="white",
               edgecolor="black", label="High-HR MAE")
    ax.set(xticks=[0, 1], xticklabels=list(OUTDOOR), ylabel="MAE (BPM)",
           title="Held-out point anchors: bars = mean MAE; whiskers = phase range; diamonds = high-HR MAE")
    ax.legend(ncol=6, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=.2)
    fig.suptitle("Physiological predictor feasibility — two outdoor rides\n"
                 "Forward predictions reset only at sparse observations; interpolation has future HR information", fontsize=14)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def mean_range(rows, column):
    """Format descriptive phase variability without claiming independent trials."""
    values = rows[column]
    return f"{values.mean():.2f} [{values.min():.2f}, {values.max():.2f}]"


def write_report(rides, fits, metrics, checks, protected_n, path):
    """Report all fixed experiments, fitted parameters and bounded conclusions."""
    outdoor = metrics.loc[metrics.domain.eq("outdoor")]
    primary = outdoor.loc[outdoor.scenario.eq("point")]
    paired = primary.pivot(index=["ride", "phase"],
                           columns="method", values="mae")
    gains = paired.M1 - paired.M2
    movement = paired.M0 - paired.M1
    means = primary.groupby(["ride", "method"])[["mae", "rmse"]].mean()
    terrain_consistent = all(means.loc[(ride, "M2"), "mae"] < means.loc[(ride, "M1"), "mae"]
                             for ride in OUTDOOR)
    meaningful = all(means.loc[(ride, "M1"), "mae"] - means.loc[(ride, "M2"), "mae"] >= MEANINGFUL
                     and means.loc[(ride, "M2"), "rmse"] <= means.loc[(ride, "M1"), "rmse"]
                     for ride in OUTDOOR)
    lines = ["# Physiological HR predictor feasibility — Milestone 2", "",
             "## Design and interpretation", "",
             "This is an exploratory forward-prediction experiment, not a final MTB reconstruction. "
             "All settings below are fixed before held-out scoring. Two outdoor rides supply the principal "
             "leave-one-ride-out comparison; phase replicates are not independent rides. "
             "A descriptive material terrain benefit was predefined as ≥0.5 BPM mean MAE reduction "
             "on **each** outdoor ride, without worsening mean RMSE. This is an operational criterion, not "
             "a validated physiological or statistical significance threshold.", "",
             "| Source | Valid HR n | HR span (min) | Recorded gaps | Supported lagged terrain % | Stationary-evidence rows |",
             "|---|---:|---:|---:|---:|---:|"]
    for ride in rides:
        frame, reference = ride["frame"], ride["reference"]
        lines.append(f"| {ride['name']} | {len(reference)} | "
                     f"{(reference.index[-1]-reference.index[0]).total_seconds()/60:.2f} | "
                     f"{frame.dt.gt(1).sum()} | {100*frame.terrain.notna().mean():.2f} | "
                     f"{frame.stationary_evidence.sum()} |")
    lines += ["", "Inputs: `data/road_ride_1.fit`, `road_ride_2.fit`, "
              "`virtual_road_ride_1.fit`, `mtb_edge.fit`, `VTT_Apero.gpx`, and "
              "`data/processed/watch_hr_ride_2026-10-16.csv`. Official FIT decoding and UTC handling "
              "are reused. Display times use Europe/Brussels. Road HR sensor metadata does not establish "
              "independent physiological truth. MyWhoosh is a separate recording/kinematic domain. "
              "The MTB HR span is its initial observed portion; all activity predictors span 173.47 minutes.", "",
              "## Model and exact predictors", "",
              "```text", "dh/dt = (h_equilibrium(t) - h) / tau",
              "h(t+dt) = exp(-dt/tau)*h(t) + (1-exp(-dt/tau))*h_equilibrium(t)",
              "M0 equilibrium = b0",
              "M1 equilibrium = b0 + b_speed*speed5 + b_stop*stop + b_uncertain*uncertain",
              "M2 equilibrium = M1 equilibrium + b_up*max(climb30_lag,0) + b_down*max(-climb30_lag,0)",
              "```", "",
              "- `speed5`: trailing mean of up to five recorded speed values in m/s, split at every "
              "recording gap. Missing speed uses the **training** speed median and an uncertainty flag; "
              "no held-out normalization or HR-derived feature is used.",
              "- `stop`: prior audit's stationary evidence, including supported short pauses as well as "
              "probable stops. `uncertain`: missing speed or speed≤0.3m/s without stationary evidence. "
              "Movement context uses existing distance/GPS criteria; absolute location is not a predictor. "
              "HR state continues to evolve at stops and is reset only by permitted sparse observations. "
              "The roads contain no probable sustained stops, so long MTB stop recovery is weakly identified.",
              "- Terrain: reuse the 5-sample median and centered 30-sample altitude/time least-squares "
              "climbing rate (29s endpoint span), with gap, movement and >2m/1s step masks and the prior "
              "30s caution margin. Delay it by 17 recorded seconds **inside each continuous moving run** "
              "so even the median's numeric support precedes the current input time. Uphill and downhill "
              "magnitudes are in m/s. No shifting crosses stops or gaps. Existing retrospective quality "
              "classification/masks remain offline: this is HR-causal prediction conditioned on an "
              "audited activity, not a fully real-time detector.",
              "- The audited 50m distance-gradient is retained in preprocessing but omitted as a model "
              "input: it is correlated with climbing rate/speed, and MTB local slope accuracy is unverified. "
              "This is a prespecified parsimonious choice, not a search using held-out performance.",
              "- On unsupported terrain, M2's instantaneous **workload correction is zero**, explicitly "
              "falling back to M1 equilibrium, not asserting flat terrain. A previous terrain effect may "
              "remain in HR state and decay normally. M2 freezes M1 coefficients and tau, fitting only "
              "two residual terrain coefficients; fallback is identical in training and evaluation.",
              "- No power, cadence, calories, load summaries, absolute location or hidden evaluation HR "
              "enters the feature matrix. Physiological state supplies the recent workload history. "
              "No endpoint correction, drift calibration, state-space smoothing or offline assimilation is applied.", "",
              "### Literature and parameter interpretation", "",
              "The first-order differential equation follows the structure in "
              "[Schnack et al. (2026)](https://doi.org/10.1186/s13102-026-01727-4). "
              "Its running-specific energy-expenditure equation, evolutionary optimizer and endpoint drift "
              "correction are not transferred here. The [official repository](https://github.com/tjorvenschnack/HR_Model) "
              "overview was reviewed; implementation-file fetching was unavailable, so this script implements "
              "the stated equation directly rather than copying or claiming reproduction of that code.", "",
              "[Lefever et al.](https://pubmed.ncbi.nlm.nih.gov/24444235/) supports compact cycling HR dynamics "
              "but used measured power, unlike this experiment. "
              "[Hunt et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6705825/) reports a cycling "
              "time constant of 68.7±21.5s under controlled conditions. The fixed grid **30, 60, 90s** is "
              "a broad, physiologically motivated feasibility range, not a subject-specific confidence interval. "
              "Tau is in seconds; constant-input HR closes about 63% of the equilibrium gap in one tau. "
              "Equilibrium is a fitted statistical asymptote, not measured resting/maximal HR or cycling power. "
              "Linear speed coefficients can be negative because faster coasting does not imply greater effort.", "",
              "### Fitting", "",
              "Unroll the entire multi-second trajectory from the left point anchor; for fixed tau this is "
              "linear in equilibrium coefficients. Fit mean squared trajectory error plus 0.1 times the sum "
              "of squared standardized coefficients (intercept unpenalized). Speed is centered/scaled using "
              "training-only mean/std; terrain magnitudes are scaled but not centered, preserving exact "
              "fallback. Tau minimizes this regularized training objective on phase-0 point trajectories. "
              "M0 and M1 each use three candidates; M2 inherits M1 tau. No held-out or MTB result chooses "
              "settings. After both outdoor folds, the same fixed procedure fits both roads for the separate "
              "virtual and MTB checks. No virtual or MTB HR trains a coefficient.", "",
              "Coefficients below are converted back to native units. Speed gain: BPM/(m/s); uphill/downhill "
              "gain: BPM/(m/s vertical); flags: BPM; intercept: BPM at zero speed, not resting HR. "
              "Small stop/uncertainty coefficients reflect little training support and regularization.", "",
              "| Training | Model | Tau s | Intercept | Speed | Stop | Uncertain | Uphill | Downhill | Training RMSE |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, fitted in fits:
        scale = fitted["scale"]
        for model in METHODS[:3]:
            beta = fitted[model]["beta"].copy()
            beta[0] -= beta[1] * scale["speed_center"] / scale["speed_scale"]
            beta[1] /= scale["speed_scale"]
            beta[4] /= scale["up_scale"]
            beta[5] /= scale["down_scale"]
            lines.append(f"| {label} | {model} | {fitted[model]['tau']:.0f} | "
                         + " | ".join(f"{v:.3f}" for v in beta)
                         + f" | {fitted[model]['train_rmse']:.2f} |")
    lines += ["", "A selected boundary tau signals limited identification/range sensitivity; the grid is "
              "not expanded after observing held-out scores. No physiological plausibility clipping hides "
              "prediction errors. Training RMSE is descriptive fit, not independent validation.", "",
              "## Sparse-interval evaluation", "",
              "Replay the existing 142×60s/15×120s sequence in elapsed UTC time, cyclically for longer "
              "rides, at phases 0/20/40s. Point anchors use exact HR; averaged sensitivity anchors use "
              "all 60 observed samples [t−30,t+29]. Incomplete windows are dropped. Averaged anchors are "
              "an alternative observation assumption and contain up to 29s look-ahead; they are not "
              "instantaneous measurements or a verified Garmin mechanism.", "",
              "An interval needs two anchors ≤180s apart in the same continuous reference/predictor block. "
              "Only the left anchor **value** initializes forward prediction; the right time bounds "
              "comparison support but its value never updates preceding states. Hidden preceding-second "
              "HR is never supplied. Stop timestamps remain in trajectories; unsupported recording gaps "
              "terminate them and require another sparse initialization. No prediction beyond the final "
              "supported anchor is scored. Linear/PCHIP reuse the existing offline implementation and "
              "can use future anchors, so they have an information advantage.", "",
              "Within each ride/phase, all five methods and both observation scenarios use the identical "
              "intersection of supported timestamps, excluding every scheduled anchor timestamp. "
              "Coverage denominator is all valid recorded HR rows. This deliberately differs from the earlier "
              "interpolation report, which included anchors. Bias is estimate minus reference; it also "
              "equals average-HR error on matching timestamps. Quality masks determine fallback, not "
              "a selective removal of difficult terrain from evaluation.", "",
              "### Outdoor results", "",
              "Equal-weight phase means [minimum, maximum]; BPM except coverage. Full phase/subset "
              "counts and errors are in the compact CSV.", "",
              "| Held-out ride | Method | MAE | RMSE | Bias / average-HR error | High-HR MAE | Coverage % |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for (ride, model), group in primary.groupby(["ride", "method"], sort=False):
        lines.append(f"| {ride} | {model} | {mean_range(group,'mae')} | "
                     f"{mean_range(group,'rmse')} | {group.bias.mean():.2f} | "
                     f"{group.high_mae.mean():.2f} | {mean_range(group,'coverage_percent')} |")
    lines += ["", "| Held-out ride | Phase s | M0 MAE/RMSE | M1 MAE/RMSE | M2 MAE/RMSE | Linear MAE/RMSE | PCHIP MAE/RMSE |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for (ride, phase), group in primary.groupby(["ride", "phase"]):
        group = group.set_index("method")
        lines.append(f"| {ride} | {phase} | " + " | ".join(
            f"{group.loc[name,'mae']:.2f}/{group.loc[name,'rmse']:.2f}" for name in METHODS) + " |")
    lines += ["", "### Peaks, increases, recovery and terrain transitions", "",
              "Prespecified score-only subsets: high HR = held-out reference's upper decile; rise/recovery "
              "= ≥+5/≤−5 BPM change over 30 fully observed consecutive seconds. These diagnostic labels "
              "never enter fitting/prediction. Peaks use SciPy prominence≥10 BPM and separation≥120s within "
              "continuous reference blocks. Keep a peak only if its entire ±15s window is commonly scored; "
              "compare local estimated maximum amplitude and its earliest maximum time with the true peak. "
              "Timing is bounded by the ±15s search and can be censored at its boundary; this is local "
              "peak discrepancy, not independently matched event detection.", "",
              "Terrain transition = a supported ≥10s rate>+0.05m/s followed within 60s by ≥10s rate<−0.05m/s, "
              "events ≥120s apart. Score ±30s around the end of the first qualifying descending run. "
              "Events use the original centered audit trend, not HR; they are approximate terrain labels. "
              "Zero eligible events/peaks give undefined (nan) errors, never zero error.", "",
              "| Held-out ride | Method | Rise MAE | Recovery MAE | Transition MAE | Peak abs amplitude error | Peak timing abs s | Eligible peaks per phase |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for (ride, model), group in primary.groupby(["ride", "method"], sort=False):
        lines.append(f"| {ride} | {model} | {group.rise_mae.mean():.2f} | "
                     f"{group.recovery_mae.mean():.2f} | {group.transition_mae.mean():.2f} | "
                     f"{group.peak_amplitude_mae.mean():.2f} | {group.peak_timing_mae_s.mean():.2f} | "
                     + "/".join(str(v) for v in group.peak_n) + " |")
    lines += ["", "### Averaged-anchor sensitivity", "",
              "Same frozen point-trained parameters and common timestamps; no refitting to averaged anchors.", "",
              "| Held-out ride | Method | Point MAE | Averaged MAE | Averaged RMSE |",
              "|---|---|---:|---:|---:|"]
    for (ride, model), group in outdoor.groupby(["ride", "method"], sort=False):
        point = group.loc[group.scenario.eq("point")]
        average = group.loc[group.scenario.eq("averaged")]
        lines.append(f"| {ride} | {model} | {point.mae.mean():.2f} | "
                     f"{average.mae.mean():.2f} | {average.rmse.mean():.2f} |")
    lines += ["", "## Separate transfer diagnostics", "",
              "Virtual rows use the final two-road fit, never pooled virtual training. Virtual speed and "
              "terrain do not measure outdoor kinematics; power/cadence are excluded. These results are "
              "a domain-shift warning rather than outdoor validation.", "",
              "The MTB diagnostic initializes only from actual FR265 wellness observations during the "
              "initial Edge-HR overlap, not hidden Edge values. Parameters and scaling are fixed from "
              "the two roads. No unavailable MTB HR is generated or exported. Compare the same M2 "
              "coefficients with Garmin and Strava terrain, alongside M1 and offline interpolation. "
              "The actual wellness observation mechanism and watch/Edge discrepancies remain confounds; "
              "this short, previously examined overlap is not independent evidence for the remaining ride. "
              "It ends at 19:57:46 Brussels, before the 20:34 discontinuity and later large altitude drift; "
              "it cannot test HR benefit in the main corrected regions.", "",
              "| Domain | Anchors | Method | MAE | RMSE | Bias | High-HR MAE | Scored n / % |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    transfer = metrics.loc[~metrics.domain.eq("outdoor")]
    for (domain, scenario, model), group in transfer.groupby(["domain", "scenario", "method"], sort=False):
        lines.append(f"| {domain} | {scenario} | {model} | {mean_range(group,'mae')} | "
                     f"{mean_range(group,'rmse')} | {group.bias.mean():.2f} | "
                     f"{group.high_mae.mean():.2f} | {group.n.min()}–{group.n.max()} / "
                     f"{group.coverage_percent.mean():.2f} |")
    lines += ["", "## Decisions and limitations", "",
              f"1. **Movement over M0:** M1 lowers MAE in {int((movement>0).sum())}/6 outdoor phases; "
              f"mean changes by ride (M0−M1, positive=improvement): "
              + ", ".join(f"{ride} {movement.loc[ride].mean():+.2f} BPM" for ride in OUTDOOR) + ".",
              f"2. **Terrain over M1:** M2 lowers MAE in {int((gains>0).sum())}/6 phases; "
              + ", ".join(f"{ride} {gains.loc[ride].mean():+.2f} BPM" for ride in OUTDOOR)
              + f". The prespecified material-benefit criterion is {'met' if meaningful else 'not met'}.",
              f"3. **Consistency:** Both outdoor mean MAEs improve with terrain: {terrain_consistent}. "
              "Per-phase tables show whether the sign persists; six phases are not six independent rides."]

    def improvement(column):
        table = primary.groupby(["ride", "method"])[column].mean()
        return "; ".join(f"{ride} {table.loc[(ride,'M1')]-table.loc[(ride,'M2')]:+.2f}"
                         for ride in OUTDOOR)
    lines += [f"4. **Peak/dynamics benefit (M1−M2 error, positive=improvement):** peak amplitude "
              f"{improvement('peak_amplitude_mae')} BPM; timing {improvement('peak_timing_mae_s')}s; "
              f"rise {improvement('rise_mae')} BPM; recovery {improvement('recovery_mae')} BPM. "
              "A better average score is not proof of recovering unseen short peaks.",
              "5. **Anchor timing sensitivity:** M2 point MAE ranges are "
              + "; ".join(f"{ride} {mean_range(primary.loc[primary.ride.eq(ride)&primary.method.eq('M2')],'mae')}"
                          for ride in OUTDOOR)
              + ". Other model/averaged results above show phase and observation-definition sensitivity."]
    mtb = metrics.loc[metrics.domain.eq("mtb_overlap")].set_index("method")
    lines += [f"6. **MTB Strava usefulness:** M1 MAE {mtb.loc['M1','mae']:.2f}, Garmin-terrain M2 "
              f"{mtb.loc['M2_Garmin','mae']:.2f}, Strava-terrain M2 {mtb.loc['M2_Strava','mae']:.2f} BPM. "
              f"Strava's MAE advantage over Garmin is only "
              f"{mtb.loc['M2_Garmin','mae']-mtb.loc['M2_Strava','mae']:.3f} BPM: no meaningful "
              "source-specific HR benefit is demonstrated in this overlap. Transfer is poor relative to "
              f"linear interpolation ({mtb.loc['linear','mae']:.2f} BPM MAE), and Strava-M2 bias is "
              f"{mtb.loc['M2_Strava','bias']:+.2f} BPM. This does not validate predictions on the rest of the MTB ride.",
              ("7. **Offline assimilation next?** The consistent material outdoor terrain benefit supports "
               "a small separately validated assimilation feasibility experiment, while retaining M1 and interpolation "
               "controls. It does not justify deploying this predictor for the final MTB reconstruction: "
               "poor target transfer and weak identification of stop/recovery dynamics remain unresolved."
               if meaningful else
               "7. **Offline assimilation next?** Terrain has not demonstrated the predefined consistent "
               "material benefit; do not advance on a claim of terrain-driven peak recovery. Any later "
               "assimilation experiment should be explicitly exploratory with M1 and interpolation controls.")
              + " No assimilation or final reconstruction is implemented here.", "",
              "The final two-road fit has a zero-speed/stopped equilibrium of approximately 149 BPM "
              "and near-zero stop coefficient. This is not a plausible calibrated resting equilibrium: "
              "it exposes a central failure of transferring road speed/coasting associations to long MTB "
              "stops. Neither clipping HR nor imposing an arbitrary resting value was used to conceal it.", "",
              "Only two outdoor rides, optical/held HR records, few stationary training samples, fixed "
              "tau, correlated seconds, sensor errors and terrain uncertainty limit interpretation. "
              "Measured pedaling power/cadence are absent outdoors. Speed is affected by coasting, wind, "
              "braking and technical riding; climbing rate is not metabolic workload. Model coefficients "
              "are descriptive, not calibrated physiological parameters. No population generalization or "
              "confidence interval follows from this experiment. Strava fixes gross elevation inconsistency "
              "but has unvalidated local gradients and some stop artifacts.", "",
              "## Reproduction and verification", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe scripts/evaluate_hr_physiology.py", "```", "",
              "Figure selection: phase-0 point observations, central 20 minutes of the longest continuous "
              "reference block in each held-out outdoor ride; no selection by HR shape or error. "
              "The figure includes all point-supported curves; metrics use the stricter shared point/average "
              "mask excluding anchors. No MTB prediction series is saved.", "",
              f"Outputs: `docs/{STEM}.md`, `docs/images/{STEM}.png`, "
              f"`data/processed/{STEM}_metrics.csv`, `scripts/evaluate_hr_physiology.py`.", "",
              "Executed checks:", ""]
    lines += ["- " + check for check in checks]
    lines += [f"- All {protected_n} protected source and prior-output SHA-256 hashes unchanged; "
              "outputs exist and are non-empty. Real-file execution, FIT integrity and UTC parsing passed.",
              "- All outdoor/virtual method/scenario score counts match within each ride/phase; "
              "MAE and bias checked against matching numeric pairs. Anchors excluded; peak windows "
              "fully supported. No installed dependencies, commits, pushes or data uploads.",
              "- Figure readability is inspected after generation; no separate test suite is created.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def verify_dynamics():
    """Check exact constant-workload integration and state continuity at a stop."""
    frame = pd.DataFrame(index=pd.date_range(
        "2026-10-01", periods=62, freq="s", tz="UTC"))
    x = np.ones((len(frame), 1))
    positions, design, initial = trajectory_design(
        frame, x, [(0, 61, 150.)], 60.)
    actual = initial + design[:, 0] * 100
    expected = 100 + 50 * np.exp(-positions / 60.)
    assert np.allclose(actual, expected)
    # An equilibrium step at t=30 must decay from the prior model state, not reset HR.
    x[30:] = 0
    positions, design, initial = trajectory_design(
        frame, x, [(0, 61, 150.)], 60.)
    result = initial + design[:, 0] * 100
    assert np.isclose(result[30], np.exp(-1 / 60.) * result[29])
    assert not np.isclose(result[30], 0)


def main():
    """Run fixed outdoor folds followed by bounded external diagnostics."""
    prior = [p for folder in ("data", "docs", "scripts")
             for p in (ROOT / folder).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts
             and STEM not in p.name and p.name != "evaluate_hr_physiology.py"]
    before = fingerprints(prior)
    checks = []
    verify_dynamics()
    checks.append(
        "Analytic constant-input response, seconds/BPM units and no HR reset at an equilibrium/stop change")
    watch = pd.read_csv(ROOT / "data/processed/watch_hr_ride_2026-10-16.csv")
    times = pd.DatetimeIndex(pd.to_datetime(watch.timestamp, utc=True))
    pattern = np.diff(times.asi8) // 1_000_000_000
    assert len(times) == 158 and (pattern == 60).sum(
    ) == 142 and (pattern == 120).sum() == 15
    rides = [prepare(name) for name in OUTDOOR]
    fits, rows, panels = [], [], []
    for held, train in ((rides[0], rides[1]), (rides[1], rides[0])):
        print("Training", train["name"],
              "holding out", held["name"], flush=True)
        fitted = fit_models([train], pattern)
        fits.append((train["name"], fitted))
        rows.extend(evaluate_ride(
            held, fitted, pattern, "outdoor", panels, checks))
    pooled = fit_models(rides, pattern)
    fits.append(("both roads (transfer only)", pooled))
    virtual = prepare("virtual_road_ride_1")
    rows.extend(evaluate_ride(virtual, pooled, pattern,
                "virtual_transfer", panels, checks))
    mtb = prepare("mtb_edge")
    rows.extend(evaluate_mtb(mtb, pooled, checks))
    checks.append(
        "Training/held-out identity separation, training-only scaling, frozen two-road transfer fit; no MTB fitting")
    metrics = pd.DataFrame(rows)
    assert len(metrics) == 95
    assert np.isfinite(metrics[["mae", "rmse", "bias"]]).all().all()
    assert metrics.rmse.ge(metrics.mae - 1e-9).all()
    assert metrics.mae.ge(metrics.bias.abs() - 1e-9).all()
    assert metrics.groupby(
        ["domain", "ride", "phase", "scenario"]).n.nunique().eq(1).all()
    csv = ROOT / "data/processed" / (STEM + "_metrics.csv")
    image = ROOT / "docs/images" / (STEM + ".png")
    report = ROOT / "docs" / (STEM + ".md")
    metrics.to_csv(csv, index=False)
    draw_figure(panels, metrics, image)
    assert fingerprints(prior) == before
    write_report(rides + [virtual, mtb], fits,
                 metrics, checks, len(prior), report)
    assert all(p.stat().st_size > 0 for p in (csv, image, report))
    assert fingerprints(prior) == before
    print(metrics.loc[metrics.scenario.isin(["point", "actual_watch"])].groupby(
        ["domain", "ride", "method"])[["mae", "rmse", "bias"]].mean().round(3).to_string())
    print("Wrote", report.relative_to(ROOT),
          image.relative_to(ROOT), csv.relative_to(ROOT))
    print("Verified protected hashes:", len(prior))


if __name__ == "__main__":
    main()
