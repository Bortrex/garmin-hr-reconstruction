"""Train trial 10 once from scratch with MTB 1 held out, without a search.

Run: conda run --no-capture-output -n deep python scripts/validate_hr_tcn_trial10.py
Existing search weights, prior outputs and the main README remain untouched.
"""

import hashlib
import json
import time
from pathlib import Path

import search_hr_tcn_128 as search
import evaluate_hr_tcn_local as local

previous, np, pd, tf = local.previous, local.np, local.pd, local.tf
ROOT, ZONE = local.ROOT, local.ZONE
HELD = "mtb_ride_1"
TRAINING = tuple("road_ride_" + str(i) for i in range(1, 6)) + ("mtb_ride_2",)
CONFIG = dict(trial_id="trial_10", dilations=[1, 2, 4, 8], channels=16,
              learning_rate=.0003, batch_size=64, receptive_field=61,
              parameters=6753)
STEM = "hr_tcn_trial10_validation"
REPORT = ROOT / "docs" / (STEM + ".md")
FIGURE = ROOT / "docs/images" / (STEM + ".png")
METRICS = ROOT / "data/processed" / (STEM + "_metrics.csv")
MODEL = ROOT / "data/processed" / (STEM + "_model.h5")
STATE = ROOT / "data/processed" / (STEM + "_normalization.json")
LABELS = {"trial10": "Trial 10 (new fit)", "seven_ride": "Seven-ride GB (saved)",
          "TCN_256": "Previous 256-to-32 TCN (saved)"}
DIAGNOSTICS = ["high_mae", "rise_mae", "recovery_mae", "peak_amplitude_mae",
               "peak_timing_mae_s", "unmatched_peaks", "stationary_mae", "restart_mae"]


def comparisons(current, historical):
    """Import exact held-out historical scores on identical timestamps/support."""
    current = current.loc[current.model.eq("TCN")].copy()
    current["model"] = "trial10"
    frames = [current]
    keys = ["domain", "ride", "phase", "condition"]
    chosen = current.set_index(keys).sort_index()
    for source, name, origin in historical:
        old = source.loc[source.ride.eq(HELD)].copy()
        assert len(old) == 6
        old["model"], old["metric_origin"] = name, origin
        aligned = old.set_index(keys).sort_index()
        assert chosen.index.equals(aligned.index)
        for column in ["n", "evaluation_timestamp_sha256", "supported_n",
                       "high_n", "rise_n", "recovery_n", "peak_n",
                       "stationary_n", "restart_n"]:
            assert np.array_equal(chosen[column], aligned[column]), column
        assert np.allclose(chosen.supported_percent, aligned.supported_percent)
        frames.append(old)
    result = pd.concat(frames, ignore_index=True)
    assert len(result) == 18
    return result


def dynamic_screen(chosen, gb):
    """Apply the previously documented practical screen, not a new tuned score."""
    passed = all(chosen[key] <= gb[key] + .5 for key in
                 ("high_mae", "recovery_mae", "stationary_mae", "restart_mae"))
    return bool(passed and chosen.peak_amplitude_mae <= gb.peak_amplitude_mae + 1
                and chosen.peak_timing_mae_s <= gb.peak_timing_mae_s + 2
                and chosen.steps_gt5_percent == 0 and chosen.invalid_hr_n == 0)


def figure(ride, cached, metrics):
    """Show a predetermined central segment and saved-score comparisons only."""
    plt, window = previous.window.plt, previous.window
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    package, estimate, _ = cached[0, "point"]
    block = max(previous.base.continuous_blocks(ride["reference"]), key=len)
    center = block.index[0] + (block.index[-1] - block.index[0]) / 2
    start = (center - pd.Timedelta(minutes=10)).floor("s")
    end = start + pd.Timedelta(minutes=20)
    for values, label, color, style in (
            (ride["reference"], "Recorded HR reference", ".6", "-"),
            (estimate, "Trial 10 estimate (new held-out fit)", "#c35b2d", "-"),
            (package["baseline"].linear, "Linear interpolation", "#72804a", "--")):
        visible = values.loc[start:end]
        axes[0].plot(visible.index, visible, label=label,
                     color=color, ls=style, lw=1.2)
    anchors = package["anchors"].loc[start:end]
    axes[0].scatter(anchors.index, anchors, s=22, color=".1", zorder=5,
                    label="Simulated point HR observations")
    axes[0].set(title="MTB ride 1 held out — phase 0, central 20min of longest recorded HR block",
                ylabel="HR (BPM)", xlabel="Local time — Europe/Brussels", xlim=(start, end))
    axes[0].xaxis.set_major_formatter(
        window.mdates.DateFormatter("%H:%M", tz=ZONE))
    axes[0].legend(ncol=2, fontsize=9)
    axes[0].grid(alpha=.2)
    point = metrics.loc[metrics.condition.eq("point")]
    for i, metric in enumerate(("mae", "rmse")):
        groups = [point.loc[point.model.eq(name), metric] for name in LABELS]
        mean = np.array([g.mean() for g in groups])
        bounds = [mean - np.array([g.min() for g in groups]),
                  np.array([g.max() for g in groups]) - mean]
        bars = axes[1].bar(np.arange(3) + (i - .5) * .34, mean, .32,
                           yerr=bounds, capsize=4, label=metric.upper(),
                           color=("#26729a", "#d18437")[i])
        for bar, value in zip(bars, mean):
            axes[1].annotate("%.3f" % value, (bar.get_x() + bar.get_width() / 2,
                             bar.get_height() / 2), ha="center", color="white", fontsize=10)
    axes[1].set(xticks=np.arange(3), xticklabels=list(LABELS.values()),
                ylabel="Error (BPM)", title="Point-anchor phase means; whiskers show phase min–max")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("Fixed trial 10 configuration — one fresh fit, MTB ride 1 excluded\n"
                 "Historical models contribute saved numerical scores, not prediction curves", fontsize=13)
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)


def report(metrics, mtb2, history, info, counts, duration, checks, protected_n):
    """Keep phases, dynamic errors, selection limitations and both rides visible."""
    def table(frame):
        """Adapt the existing Markdown renderer to a compact dataframe."""
        return "\n".join(local.markdown_table(
            list(frame.columns), frame.itertuples(index=False, name=None)))
    means = metrics.groupby(["condition", "model"]).mean(numeric_only=True)
    primary = means.loc["point"]
    chosen, gb, old = (primary.loc[name]
                       for name in ("trial10", "seven_ride", "TCN_256"))
    beat_gb = chosen.mae < gb.mae and chosen.rmse < gb.rmse
    beat_old = chosen.mae < old.mae and chosen.rmse < old.rmse
    mtb2_means = mtb2.groupby("model").mean(numeric_only=True)
    mtb2_new, mtb2_gb = mtb2_means.loc["trial10"], mtb2_means.loc["seven_ride"]
    mtb2_beats = mtb2_new.mae < mtb2_gb.mae and mtb2_new.rmse < mtb2_gb.rmse
    dynamics = dynamic_screen(chosen, gb)
    freeze = beat_gb and beat_old and dynamics and mtb2_beats and dynamic_screen(
        mtb2_new, mtb2_gb)
    lines = ["# Trial 10: additional held-out MTB ride 1 check", "",
             "Exactly one new 30-epoch fit, freshly initialized at seed 265; no search, old-model refit, "
             "all-seven fit or October 16 reconstruction. Train: road rides 1–5 and MTB ride 2. "
             "MTB ride 1 is absent from normalization and training targets. "
             "The selected search checkpoint was never loaded; only this fit's new H5 is reloaded for verification.", "",
             "## Fixed configuration and execution", "",
             "128×28 inputs, central positions [56:72] → 16 residuals. 16 channels, "
             "kernel 3, two Conv1D layers per residual block, dilations 1/2/4/8, "
             "same padding, stride 1, ReLU/skip connections and linear projection. "
             "6,753 parameters; RF = 1+4×15 = 61s. Every central output's actual GPU "
             "gradient support is checked: aggregate input positions 26–101, entirely inside 128. "
             "Adam 0.0003, batch 64, 30 epochs, seed 265, float32. "
             "Unchanged masked Huber(delta 5) + 0.5×10s HR-change MAE; both endpoints valid/unobserved.", "",
             "Environment: Python %s; TensorFlow %s; Keras %s; executable `%s`; GPU %s."
             % (info["python"].split()[0], info["tensorflow"], info["keras"],
                info["executable"], info["gpu_details"]), "",
             "Training: %d windows, %d eligible seconds, %d temporal pairs; %.2fs. "
             "Final total/Huber/temporal losses: %.5f / %.5f / %.5f."
             % (sum(c["windows"] for c in counts.values()),
                sum(c["loss_seconds"] for c in counts.values()),
                sum(c["temporal_pairs"] for c in counts.values()), duration,
                history["loss"][-1], history["huber_component"][-1], history["temporal_component"][-1]), "",
             table(pd.DataFrame(counts).T.reset_index().rename(
                 columns={"index": "training_ride"})), "",
             "Reuse search training-window extraction at stride 16 with cyclic point phases 0/20/40; "
             "128s windows never cross absent recorded seconds. Normalization uses only the six training rides. "
             "The exact October 16 sequence (142×60s + 15×120s) is replayed. "
             "Centered averages require all 60 observations and are an alternative assumption, not verified Garmin behavior. "
             "Inference uses stride 8 and averages overlapping central-16 predictions; boundary masks stay explicit. "
             "The centered five-sample residual mean, 4f(1−f) anchor taper, ≤180s support, exact anchors "
             "and no unsupported extrapolation remain unchanged. No dense hidden HR, power or cadence enters predictors.", "",
             "## MTB 1 phase results", "",
             table(metrics[["condition", "phase", "model", "mae", "rmse", "bias", "n",
                            "coverage_percent", "supported_n", "supported_percent"]].sort_values(
                                ["condition", "model", "phase"]).round(4)), "",
             "Evaluated coverage excludes missing reference HR and both conditions' anchor times; "
             "supported coverage includes retained anchors. All three models have identical UTC timestamp "
             "hashes, support and event counts for each phase/condition; no coverage loss. "
             "Phases/seconds are correlated, not independent trials.", "",
             "Phase means:", "",
             table(means.reset_index()[
                   ["condition", "model", "mae", "rmse", "bias"]].round(4)), "",
             "Dynamic metrics by phase (BPM except peak timing in seconds and unmatched counts):", "",
             table(metrics[["condition", "phase", "model"] + DIAGNOSTICS].sort_values(
                 ["condition", "model", "phase"]).round(4)), "",
             "Dynamic phase means:", "",
             table(means.reset_index()[
                   ["condition", "model"] + DIAGNOSTICS].round(4)), "",
             "Continuity/range by phase:", "",
             table(metrics[["condition", "phase", "model", "step_p99", "max_step",
                            "steps_gt5_percent", "invalid_hr_n", "prediction_min", "prediction_max"]].sort_values(
                                ["condition", "model", "phase"]).round(4)), "",
             "Metric definitions are unchanged: high HR is the reference upper decile; rise/recovery "
             "uses ±5 BPM over 30s; peaks require 10 BPM prominence and 120s separation. "
             "Peak amplitude/timing uses supported ±15s local maxima (bounded timing error); "
             "unmatched predicted peaks lack a reference peak within ±30s. "
             "Restart MAE covers ≤30s since restart; one-second changes use consecutive supported timestamps including anchors.", "",
             "## Earlier MTB 2 selection result (saved, no new fit)", "",
             table(mtb2_means.reset_index()[
                   ["model", "mae", "rmse", "bias"] + DIAGNOSTICS].round(4)), "",
             "## Decision", "",
             "1. Beats saved gradient boosting in MTB 1 point MAE and RMSE: **%s**." % (
                 "yes" if beat_gb else "no"),
             "2. Beats the previous 256→32 TCN in both: **%s**." % (
                 "yes" if beat_old else "no"),
             "3. Peak/recovery/stationary practical screen relative to GB: **%s**. "
             "Reuse the previously documented tolerances: +0.5 BPM high/recovery/stationary/restart, "
             "+1 BPM peak amplitude, +2s peak timing, no >5 BPM/s steps or invalid HR. "
             "This is a descriptive screen, not validated scientific or clinical acceptability; "
             "rise and unmatched-peak counts remain visible separately."
             % ("passes" if dynamics else "does not pass"),
             "4. MTB 1/MTB 2 both beat GB in point MAE/RMSE: **%s**. "
             "MTB 2 gains are selected tuning results, not a second untouched test."
             % ("yes" if beat_gb and mtb2_beats else "no; the ride-level accuracy result is mixed"),
             "5. Freeze trial 10 for a later explicitly authorized all-seven fit: **%s**. "
             "No promotion or final fit is performed here."
             % ("promising enough as an experimental candidate" if freeze else
                "not supported by consistent accuracy and dynamic improvements; retain GB as principal benchmark"), "",
             "MTB ride 2 was used for hyperparameter selection; MTB ride 1 has been examined "
             "in earlier development. This is an additional ride-level check, not a completely "
             "untouched independent test. Single-seed GPU nondeterminism, sensor errors, sparse/uncertainly averaged "
             "watch observations, absent power/cadence and uncertain short peaks limit interpretation. "
             "No settings were adjusted after observing these scores.", "",
             "## Figure and checks", "",
             "![Fixed MTB 1 segment and numerical comparison](images/hr_tcn_trial10_validation.png)", "",
             "Segment rule fixed before fitting: phase 0, central 20 minutes of the longest continuous "
             "reference-HR block. Historical scores have no available held-out curves here; none are fabricated.", "",
             *["- " + check for check in checks],
             "- %d previous files, including search models/results, original data and README, "
             "retain identical SHA-256." % protected_n, "",
             "New model: `data/processed/hr_tcn_trial10_validation_model.h5` (load compile=False). "
             "Normalization/configuration/history/environment/source hashes: "
             "`data/processed/hr_tcn_trial10_validation_normalization.json`. "
             "Metrics: `data/processed/hr_tcn_trial10_validation_metrics.csv`.", "",
             "```powershell",
             "conda run --no-capture-output -n deep python scripts/validate_hr_tcn_trial10.py",
             "```", "",
             "The script refuses an existing normalization/model/output path to prevent accidental "
             "additional fits or overwrites. This command ran once for this validation; "
             "subsequent artifact checks and model reloads do not train.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Initialize and fit exactly once; keep all prior experiments read-only."""
    outputs = {REPORT, FIGURE, METRICS, MODEL, STATE}
    assert not any(path.exists()
                   for path in outputs), "Validation outputs already exist; no repeat fit"
    best = search.read_json(search.RESULTS / "best.json")
    selected = search.read_json(search.RESULTS / "trial_10/status.json")
    assert best["trial_id"] == "trial_10" and selected["status"] == "completed"
    assert {key: selected[key] for key in CONFIG} == CONFIG
    plan = search.read_json(search.RESULTS / "schedule.json")
    assert next(c for c in plan["trials"]
                if c["trial_id"] == "trial_10") == CONFIG
    assert plan["fixed"] == search.FIXED and search.geometry(
        CONFIG["dilations"]) == 61
    protected = [p for folder in ("data", "docs", "scripts", "results")
                 for p in (ROOT / folder).rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts and p not in outputs]
    protected += [ROOT /
                  name for name in ("README.md", "AGENTS.md", "requirements.txt")]
    before = previous.base.fingerprints(protected)
    info = previous.environment()
    previous.WIDTH, previous.OUTPUT, previous.LEFT = 128, 16, 56
    previous.TRAIN_STRIDE, previous.INFER_STRIDE = 16, 8
    # Rebind extraction's split in this process only.
    search.TRAINING = TRAINING
    watch, _, _ = previous.window.clean_observations(pd.read_csv(
        ROOT / "data/processed/watch_hr_ride_2026-10-16.csv"))
    pattern = np.diff(watch.index.asi8) // 1_000_000_000
    assert len(watch) == 158 and (pattern == 60).sum(
    ) == 142 and (pattern == 120).sum() == 15
    rides, packages = {}, {}
    for name in TRAINING + (HELD,):
        ride = previous.window.prepare(name)
        assert "heart_rate" not in ride["frame"]
        packs = previous.base.synthetic(
            ride, pattern, previous.base.activity_features(ride["frame"]))
        for package in packs.values():
            package["features"] = previous.raw_features(ride["frame"], package)
        rides[name], packages[name] = ride, packs
        print("Prepared", name, flush=True)
    norm = previous.normalization(TRAINING, packages)
    assert norm["training"] == list(TRAINING) and HELD not in norm["training"]
    assert len(norm["features"]) == 28
    x, y, counts = search.training_arrays(
        previous, rides, packages, norm, float("inf"))
    assert set(counts) == set(TRAINING) and HELD not in counts
    previous.seed()
    network = search.build_model(local, CONFIG)
    initial = [w.copy() for w in network.get_weights()]
    search.verify_actual_rf(local, network, CONFIG, float("inf"))
    assert all(np.array_equal(a, b)
               for a, b in zip(initial, network.get_weights()))
    payload = dict(**norm, status="training_started", config={**search.FIXED, **CONFIG},
                   environment=info, counts=counts, fresh_initialization=True,
                   selected_checkpoint_loaded=False, fits=0,
                   initial_weight_sha256=hashlib.sha256(
                       b"".join(w.tobytes() for w in initial)).hexdigest(),
                   source_sha256={str(p.relative_to(ROOT)): digest for p, digest in before.items()})
    search.write_json(STATE, payload)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_threading.private_threadpool_size = 2
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(
        len(x), seed=265, reshuffle_each_iteration=True).batch(64).with_options(options).prefetch(2)
    started = time.monotonic()
    history = network.fit(dataset, epochs=30, verbose=0,
                          callbacks=[local.Progress()])
    duration = time.monotonic() - started
    history = {key: [float(v) for v in values]
               for key, values in history.history.items()}
    assert all(len(values) == 30 and np.isfinite(values).all()
               for values in history.values())
    assert any(not np.array_equal(a, b)
               for a, b in zip(initial, network.get_weights()))
    assert all(np.isfinite(w).all() for w in network.get_weights())
    network.save(MODEL, include_optimizer=False)
    payload.update(status="trained", fits=1, history=history,
                   training_seconds=duration)
    search.write_json(STATE, payload)
    gb = pd.read_csv(search.GB_METRICS)
    gb = gb.loc[gb.model.eq("seven_ride")]
    old = pd.read_csv(search.OLD_METRICS)
    old = old.loc[old.model.eq("TCN")]
    fitted = dict(network=network, training=TRAINING, normalization=norm)
    checks = ["TensorFlow GPU matmul and all-output convolution/RF gradients pass; input/output 128×28→16×1, 6,753 parameters.",
              "Fresh seeded weights restored after RF check; the search checkpoint was never loaded. Exactly one 30-epoch fit.",
              "Training windows and normalization use only five roads plus MTB 2; gap-isolated stride-16 targets and valid endpoint masks reused."]
    rows, cached = previous.evaluate(rides[HELD], packages[HELD], fitted,
                                     gb.set_index(
                                         ["domain", "ride", "phase", "condition"]),
                                     "mtb_holdout", checks)
    metrics = comparisons(pd.DataFrame(rows), [(gb, "seven_ride", "saved_seven_ride"),
                                               (old, "TCN_256", "saved_previous_tcn")])
    metrics.to_csv(METRICS, index=False)
    reloaded = tf.keras.models.load_model(MODEL, compile=False)
    assert all(np.array_equal(a, b)
               for a, b in zip(network.get_weights(), reloaded.get_weights()))
    repeat = previous.predict(rides[HELD]["frame"], packages[HELD][0, "point"],
                              dict(network=reloaded, training=TRAINING, normalization=norm))
    pd.testing.assert_series_equal(
        repeat, cached[0, "point"][1], atol=1e-7, rtol=1e-7)
    saved = pd.read_csv(METRICS)
    assert len(saved) == 18 and set(saved.model) == set(LABELS)
    assert np.allclose(saved.select_dtypes(include="number"),
                       metrics.select_dtypes(include="number"), equal_nan=True, atol=1e-10, rtol=1e-10)
    assert saved.groupby(["condition", "phase"]
                         ).evaluation_timestamp_sha256.nunique().eq(1).all()
    checks.append(
        "New H5 reload reproduces the complete phase-0 estimate; CSV read-back matches metrics, UTC hashes/support and both saved comparators.")
    mtb2 = pd.read_csv(search.RESULTS / "trial_10/metrics.csv")
    mtb2 = mtb2.loc[mtb2.condition.eq("point")].copy()
    assert np.isclose(mtb2.mae.mean(), selected["validation_mae"])
    mtb2["model"] = "trial10"
    mtb2 = pd.concat([mtb2, gb.loc[gb.ride.eq("mtb_ride_2") & gb.condition.eq("point")],
                     old.loc[old.ride.eq("mtb_ride_2") & old.condition.eq("point")].assign(model="TCN_256")])
    assert len(mtb2) == 9 and mtb2.groupby(
        "phase").evaluation_timestamp_sha256.nunique().eq(1).all()
    figure(rides[HELD], cached, metrics)
    assert previous.base.fingerprints(protected) == before
    payload.update(status="completed", checks=checks,
                   protected_file_count=len(protected))
    search.write_json(STATE, payload)
    report(metrics, mtb2, history, info, counts,
           duration, checks, len(protected))
    assert all(path.stat().st_size > 0 for path in outputs)
    assert previous.base.fingerprints(protected) == before
    print("One fit completed:", len(x), "windows;",
          round(duration, 2), "seconds", flush=True)
    print(metrics.loc[metrics.condition.eq("point")].groupby("model")[
          ["mae", "rmse", "bias"] + DIAGNOSTICS].mean().round(6).to_string())
    print("Outputs:", *map(str, sorted(outputs)), sep="\n")


if __name__ == "__main__":
    main()
