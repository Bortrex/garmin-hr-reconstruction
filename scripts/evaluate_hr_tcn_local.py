"""Evaluate one 64-to-16 TCN with Huber plus 10-second HR-difference loss.

Run: conda run --no-capture-output -n deep python scripts/evaluate_hr_tcn_local.py
Reuse the previous experiment's helpers in this process; never edit its files.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import evaluate_hr_tcn_tensorflow as previous

tf, base, window = previous.tf, previous.base, previous.window
ROOT, ZONE, RIDES, MTB = previous.ROOT, previous.ZONE, previous.RIDES, previous.MTB
markdown_table = previous.markdown_table
STEM = "hr_tcn_local"
METRICS = ROOT / "data/processed" / (STEM + "_metrics.csv")
CANDIDATE = ROOT / "data/processed" / (STEM + "_candidate_2026-10-16.csv")
REPORT = ROOT / "docs" / (STEM + "_evaluation.md")
FIGURE = ROOT / "docs/images" / (STEM + "_evaluation.png")
TARGET_FIGURE = ROOT / "docs/images" / (STEM + "_candidate_2026-10-16.png")
MODEL = ROOT / "data/processed" / (STEM + "_model.h5")
NORMALIZATION = ROOT / "data/processed" / (STEM + "_normalization.json")
CHECKPOINTS = ROOT / "data/processed" / (STEM + "_checkpoints")
OLD_METRICS = previous.METRICS
GB_METRICS = previous.benchmark.METRICS
GB_CANDIDATE = previous.benchmark.CANDIDATE
CONFIG = dict(input_seconds=64, output_seconds=16, left=24, train_stride=16,
              inference_stride=8, channels=16, kernel=3, dilations=[1, 2, 4, 4],
              features=28, receptive_field=45, seed=265, batch_size=32,
              epochs=30, learning_rate=.001, huber_delta=5.,
              temporal_seconds=10, lambda_temporal=.5, label_channels=3)
LABELS = {"seven_ride": "Seven-ride GB", "TCN_256": "Previous 256→32 TCN",
          "TCN_local": "Local 64→16 TCN", "linear": "Linear interpolation"}
ORIGINAL_TRAIN = previous.train


def loss_values(y_true, y_pred):
    """Compute separately normalized Huber and absolute-HR 10-second errors."""
    valid = y_true[..., 1] > 0
    mask = tf.cast(valid, tf.float32)
    truth_residual = tf.where(valid, y_true[..., 0], 0.)
    predicted_residual = tf.where(valid, tf.squeeze(y_pred, -1), 0.)
    baseline = tf.where(valid, y_true[..., 2], 0.)
    error = predicted_residual - truth_residual
    absolute = tf.abs(error)
    huber = tf.where(absolute <= 5., .5 * tf.square(error),
                     5. * (absolute - 2.5))
    huber = tf.reduce_sum(huber * mask) / tf.maximum(tf.reduce_sum(mask), 1.)
    # Keep the aligned baseline explicit: the differences are on absolute HR.
    actual_hr = baseline + truth_residual
    predicted_hr = baseline + predicted_residual
    actual_change = actual_hr[:, 10:] - actual_hr[:, :-10]
    predicted_change = predicted_hr[:, 10:] - predicted_hr[:, :-10]
    pairs = mask[:, 10:] * mask[:, :-10]
    temporal = tf.reduce_sum(tf.abs(predicted_change - actual_change) * pairs)
    temporal /= tf.maximum(tf.reduce_sum(pairs), 1.)
    return huber, temporal


@tf.keras.utils.register_keras_serializable(package="hr_tcn_local")
def combined_loss(y_true, y_pred):
    """Apply the fixed Huber + 0.5 temporal objective without another penalty."""
    huber, temporal = loss_values(y_true, y_pred)
    return huber + .5 * temporal


@tf.keras.utils.register_keras_serializable(package="hr_tcn_local")
def huber_component(y_true, y_pred):
    """Report the eligible-timestamp-normalized Huber component."""
    return loss_values(y_true, y_pred)[0]


@tf.keras.utils.register_keras_serializable(package="hr_tcn_local")
def temporal_component(y_true, y_pred):
    """Report the eligible-pair-normalized temporal component in BPM."""
    return loss_values(y_true, y_pred)[1]


def local_model():
    """Build the one fixed local architecture, output positions 24 through 39."""
    inputs = tf.keras.Input((64, 28), dtype="float32", name="cycling_context")
    x = tf.keras.layers.Conv1D(
        16, 1, activation="relu", name="input_projection")(inputs)
    for dilation in CONFIG["dilations"]:
        skip = x
        x = tf.keras.layers.Conv1D(16, 3, dilation_rate=dilation, padding="same",
                                   strides=1, activation="relu")(x)
        x = tf.keras.layers.Conv1D(
            16, 3, dilation_rate=dilation, padding="same", strides=1)(x)
        x = tf.keras.layers.Activation("relu")(
            tf.keras.layers.Add()([skip, x]))
    x = tf.keras.layers.Conv1D(
        1, 1, activation=None, name="residual_projection")(x)
    outputs = tf.keras.layers.Cropping1D((24, 24), name="central_16")(x)
    network = tf.keras.Model(inputs, outputs, name="local_tcn_temporal_loss")
    network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                    loss=combined_loss, metrics=[huber_component, temporal_component])
    assert network.input_shape == (None, 64, 28)
    assert network.output_shape == (
        None, 16, 1) and network.count_params() == 6753
    return network


def training_windows(names, rides, packages, norm):
    """Extract full 64s blocks at stride 16 and add aligned baseline labels."""
    inputs, targets, counts = [], [], {}
    for name in names:
        frame, reference = rides[name]["frame"], rides[name]["reference"]
        values = {phase: previous.normalized(packages[name][phase, "point"]["features"], norm)
                  for phase in base.PHASES}
        used, n, pair_n = [], 0, 0
        for block in previous.block_indices(frame):
            last_target = None
            for start in range(0, len(block) - 64 + 1, 16):
                phase = base.PHASES[(start // 16) % len(base.PHASES)]
                package = packages[name][phase, "point"]
                output = block[start + 24:start + 40]
                times = frame.index[output]
                assert np.all(
                    np.diff(frame.index[block[start:start + 64]].asi8) == 1_000_000_000)
                assert last_target is None or times[0] == last_target + \
                    pd.Timedelta(seconds=1)
                last_target = times[-1]
                baseline = package["baseline"].linear.reindex(times)
                residual = reference.reindex(times) - baseline
                valid = residual.notna().to_numpy() & ~times.isin(
                    package["anchors"].index)
                if not valid.any():
                    continue
                # All invalid labels are finite placeholders, excluded by mask.
                target = np.column_stack(
                    [residual.fillna(0), valid, baseline.fillna(0)]).astype(np.float32)
                inputs.append(values[phase][block[start:start + 64]])
                targets.append(target)
                used.extend(times[valid])
                pair_n += int((valid[10:] & valid[:-10]).sum())
                n += 1
        assert pd.DatetimeIndex(used).is_unique
        counts[name] = {"windows": n, "loss_seconds": len(
            used), "temporal_pairs": pair_n}
    assert all(value["windows"] > 0 for value in counts.values())
    return np.stack(inputs), np.stack(targets), counts


class Progress(tf.keras.callbacks.Callback):
    """Log all fixed loss components without any held-out monitoring."""

    def on_epoch_end(self, epoch, logs=None):
        values = [float(logs[key]) for key in (
            "loss", "huber_component", "temporal_component")]
        if not np.isfinite(values).all():
            raise ValueError(
                "Nonfinite loss component: stop without another training attempt")
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}/30; total={values[0]:.5f}, Huber={values[1]:.5f}, temporal={values[2]:.5f}", flush=True)


def configure_helpers():
    """Bind existing helpers to this experiment inside this process only."""
    # The previous script and its defaults on disk remain byte-for-byte intact.
    for name, value in dict(WIDTH=64, OUTPUT=16, LEFT=24, TRAIN_STRIDE=16, INFER_STRIDE=8,
                            STEM=STEM, METRICS=METRICS, CANDIDATE=CANDIDATE, REPORT=REPORT,
                            FIGURE=FIGURE, TARGET_FIGURE=TARGET_FIGURE, MODEL=MODEL,
                            NORMALIZATION=NORMALIZATION, CHECKPOINTS=CHECKPOINTS).items():
        setattr(previous, name, value)
    previous.model, previous.training_windows, previous.Progress = local_model, training_windows, Progress
    assert previous.CHECKPOINTS.name == "hr_tcn_local_checkpoints"
    assert len(previous.FEATURES) == 28 and OLD_METRICS != METRICS


def train(names, rides, packages, fits, initial=None):
    """Reuse fitting/checkpoints while guarding the local experiment identity."""
    state = CHECKPOINTS / f"fit_{len(fits)+1}.json"
    if "--resume" in sys.argv and state.is_file():
        assert json.loads(state.read_text(encoding="utf-8")
                          ).get("experiment") == CONFIG
    fitted = ORIGINAL_TRAIN(names, rides, packages, fits, initial)
    network = fitted["network"]
    assert network.input_shape == (
        None, 64, 28) and network.output_shape == (None, 16, 1)
    if "huber_component" not in fits[-1]:
        for key in ("huber_component", "temporal_component"):
            fits[-1][key] = [float(value)
                             for value in network.history.history[key]]
    cached = json.loads(state.read_text(encoding="utf-8"))
    cached["fit"], cached["experiment"] = fits[-1], CONFIG
    state.write_text(json.dumps(cached, indent=2), encoding="utf-8")
    return fitted


def smoke():
    """Verify every output's RF, alignment, masking, losses and temporal gradients."""
    previous.seed()
    network = local_model()
    weights = network.get_weights()
    network.set_weights([np.full_like(weight, .01) for weight in weights])
    with tf.device("/GPU:0"):
        x = tf.ones((1, 64, 28), tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y = network(x, training=False)
            probes = [y[0, i, 0] for i in range(16)]
        for i, probe in enumerate(probes):
            active = np.flatnonzero(
                np.any(tape.gradient(probe, x).numpy()[0] != 0, axis=1))
            assert len(
                active) == 45 and active[0] == 24 + i - 22 and active[-1] == 24 + i + 22
            assert active[0] >= 0 and active[-1] < 64
        del tape
    assert "GPU:0" in y.device
    network.set_weights(weights)
    values = np.arange(83, dtype=np.float32)[:, None]
    centers = np.arange(0, len(values), 8)
    predictions = np.array([center + np.arange(16) for center in centers])
    assert np.array_equal(previous.overlap(
        len(values), centers, predictions), values[:, 0])
    for center in centers:
        padded = previous.pad_window(values, int(center))
        n = min(16, len(values) - center)
        assert np.array_equal(padded[24:24 + n, 0],
                              values[center:center + n, 0])
    # Synthetic gap: all training inputs and target pairs stay inside one side.
    times = pd.date_range("2026-01-01", periods=181,
                          freq="s", tz="UTC").delete(np.arange(80, 101))
    frame = pd.DataFrame(index=times)
    for block in previous.block_indices(frame):
        outputs = []
        for start in range(0, len(block) - 64 + 1, 16):
            assert np.diff(times[block[start:start + 64]
                                 ].asi8).tolist() == [1_000_000_000] * 63
            outputs.extend(times[block[start + 24:start + 40]])
        assert pd.DatetimeIndex(outputs).is_unique
        assert np.all(np.diff(pd.DatetimeIndex(outputs).asi8) == 1_000_000_000)
    target = np.zeros((1, 16, 3), np.float32)
    target[..., 1] = 1
    target[0, :, 2] = 100 + 2 * np.arange(16)
    predicted = tf.constant(
        np.arange(16, dtype=np.float32).reshape(1, 16, 1) * .5)
    h, t = loss_values(target, predicted)
    errors = np.arange(16) * .5
    expected_h = np.where(errors <= 5, .5 * errors**2,
                          5 * (errors - 2.5)).mean()
    assert np.isclose(float(h), expected_h) and float(t) == 5
    assert np.isclose(
        float(combined_loss(target, predicted)), expected_h + 2.5)
    h, t = loss_values(target, tf.ones((1, 16, 1)) * 2)
    assert float(h) == 2 and float(t) == 0
    assert np.isclose(float(h), float(
        previous.masked_huber(target, tf.ones((1, 16, 1)) * 2)))
    # Only one pair is eligible; missing/anchor endpoints exclude every other pair.
    target[..., 1] = 0
    target[0, [2, 12], 1] = 1
    variable = tf.Variable(np.zeros((1, 16, 1), np.float32))
    variable[0, 12, 0].assign(4)
    with tf.GradientTape() as tape:
        temporal = temporal_component(target, variable)
    gradient = tape.gradient(temporal, variable).numpy()[0, :, 0]
    assert float(temporal) == 4 and gradient[2] == -1 and gradient[12] == 1
    assert np.count_nonzero(gradient) == 2 and np.isfinite(gradient).all()
    # Masked NaNs, an anchor endpoint, and a batch without any eligible pair.
    target[0, 12, 1] = 0
    target[0, target[0, :, 1] == 0, 0] = np.nan
    target[0, target[0, :, 1] == 0, 2] = np.nan
    assert float(temporal_component(target, variable)) == 0
    with tf.GradientTape() as tape:
        total = combined_loss(target, variable)
    assert np.isfinite(float(total)) and np.isfinite(
        tape.gradient(total, variable).numpy()).all()
    target[..., 1] = 0
    h, t = loss_values(target, variable)
    assert float(h) == 0 and float(t) == 0
    target = np.zeros((1, 16, 3), np.float32)
    target[0, 0, 1] = 1
    assert float(huber_component(target, tf.ones((1, 16, 1)) * 10)) == 37.5
    print("Single local smoke passed: 6753 parameters, all 16 outputs have full 45s RF, aligned overlaps and finite masked loss/gradients", flush=True)
    return network


def comparisons(current):
    """Import original benchmark scores and require exact common timestamp hashes."""
    old = pd.read_csv(OLD_METRICS)
    old = old.loc[old.model.eq("TCN")].copy()
    old["model"], old["metric_origin"] = "TCN_256", "saved_previous_tcn"
    gb = pd.read_csv(GB_METRICS)
    gb = gb.loc[gb.model.eq("seven_ride")].copy()
    gb["metric_origin"] = "saved_seven_ride"
    current.loc[current.model.eq("TCN"), "model"] = "TCN_local"
    metrics = pd.concat([current, old, gb], ignore_index=True)
    assert len(metrics) == 52
    assert metrics.groupby(["domain", "ride", "phase"]).n.nunique().eq(1).all()
    assert metrics.groupby(["domain", "ride", "phase"]
                           ).evaluation_timestamp_sha256.nunique().eq(1).all()
    keys = ["domain", "ride", "phase", "condition"]
    linear = pd.read_csv(OLD_METRICS)
    linear = linear.loc[linear.model.eq("linear")].set_index(keys)
    new = current.loc[current.model.eq("linear")].set_index(keys)
    for col in ("mae", "rmse", "bias", "n", "peak_n", "stationary_n", "restart_n"):
        assert np.allclose(new[col], linear.loc[new.index, col],
                           atol=1e-9, rtol=1e-9, equal_nan=True)
    return metrics


def plots(panels, metrics, target, table):
    """Show fixed held-out segments and compare the target to saved GB estimates."""
    plt = window.plt
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    for ax, (ride, cached) in zip(axes[:2], panels):
        package, estimate, _ = cached[0, "point"]
        block = max(base.continuous_blocks(ride["reference"]), key=len)
        center = block.index[0] + (block.index[-1] - block.index[0]) / 2
        start = (center - pd.Timedelta(minutes=10)).floor("s")
        end = start + pd.Timedelta(minutes=20)
        actual = ride["reference"].loc[start:end]
        ax.plot(actual.index, actual, color=".65",
                lw=1.3, label="Recorded HR reference")
        visible = estimate.loc[start:end]
        ax.plot(visible.index, visible, color="#c35b2d",
                lw=1.3, label="Local TCN estimate")
        linear = package["baseline"].linear.loc[start:end]
        ax.plot(linear.index, linear, color="#72804a",
                lw=1, ls="--", label="Linear interpolation")
        sparse = package["anchors"].loc[start:end]
        ax.scatter(sparse.index, sparse, color=".15", s=20,
                   zorder=5, label="Sparse point observations")
        ax.set(title=ride["name"] + " — held out; phase 0, fixed central 20min", ylabel="HR (BPM)",
               xlabel="Local time — Europe/Brussels", xlim=(start, end))
        ax.xaxis.set_major_formatter(
            window.mdates.DateFormatter("%H:%M", tz=ZONE))
        ax.legend(ncol=2, fontsize=8)
        ax.grid(alpha=.2)
    primary = metrics.loc[metrics.domain.eq(
        "mtb_holdout") & metrics.condition.eq("point")]
    for i, (name, color) in enumerate((("seven_ride", "#257c9b"), ("TCN_256", "#72804a"), ("TCN_local", "#c35b2d"))):
        groups = [primary.loc[primary.ride.eq(
            ride) & primary.model.eq(name)] for ride in MTB]
        mean = np.array([g.mae.mean() for g in groups])
        low, high = np.array([g.mae.min() for g in groups]), np.array(
            [g.mae.max() for g in groups])
        bars = axes[2].bar(np.arange(2) + (i - 1) * .24, mean, .22, color=color, label=LABELS[name],
                           yerr=[mean - low, high - mean], capsize=3)
        for bar, g in zip(bars, groups):
            axes[2].annotate(f"MAE {g.mae.mean():.2f}\nRMSE {g.rmse.mean():.2f}",
                             (bar.get_x() + bar.get_width() / 2, g.mae.max() + .03), ha="center", fontsize=8)
    axes[2].set(xticks=[0, 1], xticklabels=list(MTB), ylabel="MAE (BPM)",
                title="Exact common timestamps — phase means and min–max, not confidence intervals")
    axes[2].set_ylim(0, primary.mae.max() + 1)
    axes[2].legend(loc="lower center", ncol=3, fontsize=8)
    axes[2].grid(axis="y", alpha=.2)
    fig.suptitle("Local TensorFlow TCN — 64s input → central 16s residuals\nFixed Huber + 0.5 × 10s temporal-difference loss; previous models use saved scores", fontsize=14)
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)
    gb = pd.read_csv(GB_CANDIDATE)
    gb.index = pd.to_datetime(gb.timestamp_utc, utc=True)
    assert gb.index.equals(table.index) and gb.final_hr.notna().equals(
        table.final_hr.notna())
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    ax.plot(table.index, gb.final_hr.where(gb.estimated), color="#257c9b",
            lw=1.15, ls="--", label="Seven-ride GB estimate (saved)")
    ax.plot(table.index, table.final_hr.where(table.estimated),
            color="#c35b2d", lw=1.2, label="Local TCN estimate")
    ax.plot(table.index, table.edge_hr, color=".35",
            lw=1, label="Recorded Edge HR")
    watch = table.watch_hr.dropna()
    ax.scatter(watch.index, watch, color=".15", s=12,
               zorder=5, label="Recorded wellness summaries")
    ax.axvline(target["reference"].index[-1], color=".35",
               ls=":", lw=1, label="Last recorded Edge HR")
    for i, indices in enumerate(previous.runs(target["frame"].probable_stop, table.index)):
        times = table.index[indices]
        ax.axvspan(times[0], times[-1], color=".5", alpha=.09,
                   label="Probable sustained stops" if i == 0 else None)
    ax.set(title="October 16 MTB — local TCN and seven-ride GB estimates, with recorded observations",
           ylabel="HR (BPM)", xlabel="Local time — Europe/Brussels", xlim=(table.index[0], table.index[-1]))
    ax.xaxis.set_major_formatter(window.mdates.DateFormatter("%H:%M", tz=ZONE))
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(TARGET_FIGURE, dpi=170)
    plt.close(fig)


def recommendation(metrics):
    """Require consistent practical gains; otherwise retain the principal benchmark."""
    primary = metrics.loc[metrics.domain.eq(
        "mtb_holdout") & metrics.condition.eq("point")]
    means = primary.groupby(["ride", "model"]).mean(numeric_only=True)
    passed = []
    for ride in MTB:
        new, gb, old = means.loc[(ride, "TCN_local")], means.loc[(
            ride, "seven_ride")], means.loc[(ride, "TCN_256")]
        passed += [new.mae < min(gb.mae, old.mae), new.rmse < min(gb.rmse, old.rmse),
                   new.high_mae <= gb.high_mae + .5, new.recovery_mae <= gb.recovery_mae + .5,
                   new.peak_amplitude_mae <= gb.peak_amplitude_mae +
                   1, new.peak_timing_mae_s <= gb.peak_timing_mae_s + 2,
                   new.stationary_mae <= gb.stationary_mae + .5, new.restart_mae <= gb.restart_mae + .5]
    local = metrics.loc[metrics.model.eq("TCN_local")]
    passed += [local.steps_gt5_percent.eq(0).all(),
               local.invalid_hr_n.eq(0).all()]
    return "C. Prefer the new 64→16 local TCN." if all(passed) else "A. Retain the existing gradient-boosting benchmark."


def report(metrics, fits, table, info, checks, protected_n):
    """Report the single combined-change experiment, losses and actual tradeoffs."""
    choice = recommendation(metrics)
    lines = ["# Local TCN with temporal-difference loss", "", "**" + choice + "**", "",
             "Exactly three local-TCN fits: hold out MTB 1, hold out MTB 2, then all seven real rides. Previous 256→32 TCN and seven-ride gradient boosting use saved CSV metrics only. No virtual data, search, alternate seed, early stopping, previous-model refit or automatic promotion. Geometry and loss change together; this experiment cannot attribute a difference to either change in isolation.", "",
             "## Architecture and environment", "", "```json", json.dumps(
                 info, indent=2), "```", "",
             "Existing `deep` environment, unchanged TensorFlow/Keras/CUDA and pure-Python SDK loading. Actual GPU matmul and convolutions pass. GPU nondeterminism remains possible despite seed 265, deterministic-op/cuBLAS flags and float32 with TF32/mixed precision disabled; no repeated fit is used to claim training determinism.", "",
             "Input (64,28); 1×1 projection to 16 channels; four residual blocks with two kernel-3, stride-1, same-padding convolutions at dilations **1,2,4,4**, ReLU and skip-add; linear 1×1 projection; crop [24:40], output (16,1). **6,753 parameters.** RF = 1+2×(3−1)×(1+2+4+4) = **45s**, radius 22. Gradient checks for every output prove RF endpoints [24+j−22,24+j+22], j=0…15: aggregate support [2,61] stays inside the 64s input, so central predictions do not depend on external convolution padding. This RF describes network features; existing anchor context and cycling preprocessing retain their own temporal support.", "",
             "## Reused data pipeline and fixed loss", "",
             "Reuse the previous script's 28 feature channels, raw-feature construction, training-only means/scales, sparse-anchor simulation, block discovery, padded extraction, overlap averaging, prediction postprocessing, evaluation, candidate writer and checkpoint fitting loop. Its source files stay unchanged. This script binds those helpers to local lengths and experiment-specific paths in its own process. New checkpoints also verify experiment/config identity, input/output shapes and source hashes, so the old 256→32 weights cannot be resumed accidentally.", "",
             "Train on full 64 consecutive recorded seconds, stride 16. Central targets are positions 24…39, then 40…55, then 56…71: adjacent, non-overlapping segments. Each window receives exactly one cyclic phase 0/20/40 based on start/16 modulo 3, preserving the prior phase-selection convention. Point anchors only for training; point and complete centered-60s-average anchors for sensitivity evaluation. Actual October 16 ordered 142×60s+15×120s intervals are replayed by elapsed UTC time, with ≤180s support and no extrapolation. Missing HR is never synthesized. Existing terrain masks, stop/restart features and explicit feature-validity flags remain unchanged; no power/cadence or dense HR enters inputs or normalization.", "",
             "Labels minimally extend [true_residual, valid_mask] to **[true_residual, valid_mask, aligned_linear_HR]**. Huber uses delta 5 BPM and divides its sum by valid target seconds, excluding unsupported/missing reference values and observed anchors. Invalid labels are finite placeholders; the loss additionally sanitizes masked NaNs. For the temporal component explicitly form actual_HR = baseline+true_residual and predicted_HR = baseline+predicted_residual, then compare their t→t+10 changes. Each 16s segment offers six pairs, with both endpoints valid/unobserved and inside the same recorded block and output window. Average absolute change error over eligible pairs; no valid pairs gives zero with finite gradients. Baseline cancels algebraically in the final difference error, but both absolute-HR trajectories are explicitly constructed from aligned labels as specified.", "",
             "**Total = Huber + 0.5 × temporal**, fixed before evaluation. Huber is quadratic in BPM below delta and delta-scaled linear above it; the temporal term is MAE in BPM. Their numerical scales and fixed coefficient are empirical, not a calibrated physiological likelihood. Adam 0.001, batch 32, float32, seed 265 and 30 epochs remain unchanged. Components below are Keras epoch means of per-batch normalized components; total loss uses Keras loss aggregation, so last partial batches can produce a small aggregation difference from Huber+0.5×temporal. They are training metrics, not held-out selection criteria.", ""]
    rows = [[i, "+".join(f["training"]), f["windows"], sum(c["loss_seconds"] for c in f["counts"].values()), sum(c["temporal_pairs"] for c in f["counts"].values()),
             f"{f['seconds']:.2f}", f"{f['loss'][-1]:.4f}", f"{f['huber_component'][-1]:.4f}", f"{f['temporal_component'][-1]:.4f}"] for i, f in enumerate(fits, 1)]
    lines += markdown_table(["Fit", "Training rides", "Windows", "Loss seconds",
                            "10s pairs", "Training s", "Final total", "Final Huber", "Final temporal"], rows)
    lines += markdown_table(["Fit", "Source ride", "Windows", "Loss seconds", "Pairs"], [[i, name, c["windows"],
                            c["loss_seconds"], c["temporal_pairs"]] for i, f in enumerate(fits, 1) for name, c in f["counts"].items()])
    lines += ["Inference uses central starts 0,8,16,… and averages overlapping 16s outputs at matching seconds inside each continuous block. Absent boundary context is zero-filled with record_valid=0 and explicit feature masks, never copied from another block. The existing centered five-sample residual mean and 4f(1−f) taper run independently within supported anchor spans; observations are restored exactly. Invalid terrain remains masked. Full prediction support equals linear support; the evaluated UTC hashes match both saved model CSVs exactly, with no coverage loss.", "",
              "## Held-out MTB comparison", "", "Individual seconds/phases are correlated; only two MTB activities are independent evaluation units. All methods use identical scored timestamps and exclude anchor times from both conditions. Historical predictions are never invented; previous models contribute their exact saved scores.", ""]
    held = metrics.loc[metrics.domain.eq("mtb_holdout")]
    rows = []
    for (ride, condition, phase), g in held.groupby(["ride", "condition", "phase"]):
        g = g.set_index("model")
        rows.append([ride, condition, phase] + [f"{g.loc[m,'mae']:.3f}/{g.loc[m,'rmse']:.3f}" for m in ("seven_ride", "TCN_256", "TCN_local")] + [
                    f"{g.loc['TCN_local','bias']:+.3f}", f"{int(g.loc['TCN_local','n'])} ({g.loc['TCN_local','coverage_percent']:.2f}%)"])
    lines += markdown_table(["Ride", "Anchors", "Phase", "GB MAE/RMSE",
                            "Old TCN", "Local TCN", "Local bias", "Common n (%)"], rows)
    rows = [[ride, condition, LABELS[name], f"{g.mae.mean():.3f} [{g.mae.min():.3f},{g.mae.max():.3f}]", f"{g.rmse.mean():.3f} [{g.rmse.min():.3f},{g.rmse.max():.3f}]"] for (
        ride, condition, name), g in held.groupby(["ride", "condition", "model"])]
    lines += markdown_table(["Ride", "Anchors", "Model",
                            "Mean MAE [phase range]", "Mean RMSE [phase range]"], rows)
    lines += [
        "Established metric definitions are unchanged: high HR is the reference upper decile; rise/recovery is ≥5/≤−5 BPM over 30s; peaks have ≥10 BPM prominence and ≥120s separation. Peak amplitude/timing uses supported ±15s local maxima and thus bounded timing, not unrestricted peak matching. Unmatched peaks lack a reference peak within ±30s. Stationary/restart (≤30s) errors require measured reference HR. Roughness uses consecutive supported seconds including anchors. CSV contains every phase's event counts and errors; NaN is unavailable, never zero error.", ""]
    rows = [[ride, LABELS[name], f"{g.high_mae.mean():.2f}", f"{g.rise_mae.mean():.2f}/{g.recovery_mae.mean():.2f}", f"{g.peak_amplitude_mae.mean():.2f}/{g.peak_timing_mae_s.mean():.2f}", f"{g.stationary_mae.mean():.2f}/{g.restart_mae.mean():.2f}",
             f"{g.unmatched_peaks.mean():.1f}", f"{g.step_p99.mean():.3f}/{g.max_step.max():.3f}", f"{g.steps_gt5_percent.max():.3f}"] for (ride, name), g in held.loc[held.condition.eq("point")].groupby(["ride", "model"])]
    lines += markdown_table(["Ride", "Model", "High MAE", "Rise/recovery", "Peak amp/timing",
                            "Stationary/restart", "Unmatched mean", "Step p99 mean/max", ">5 BPM/s %"], rows)
    lines += ["## October 16 secondary diagnostic and candidate", "",
              "Frozen all-seven local model uses actual FR265 wellness observations and quality-masked Strava climb30. Initial Edge overlap is withheld from predictors; only 21 actual watch anchors supply HR context there. This repeatedly examined overlap is secondary, never used to tune training.", ""]
    rows = [[LABELS[r.model], f"{r.mae:.3f}/{r.rmse:.3f}", f"{r.bias:+.3f}", f"{r.high_mae:.2f}", f"{r.recovery_mae:.2f}",
             f"{int(r.n)} ({r.coverage_percent:.2f}%)"] for r in metrics.loc[metrics.domain.eq("mtb_overlap")].itertuples()]
    lines += markdown_table(["Model", "MAE/RMSE", "Bias",
                            "High MAE", "Recovery MAE", "Scored n (%)"], rows)
    finite = table.final_hr.dropna()
    step = table.final_hr.diff().abs().where(
        table.estimated & table.estimated.shift(fill_value=False))
    lines += ["Full candidate construction reuses the established handoff: original Edge HR takes priority; its last known value anchors only subsequent estimates, followed by later watch summaries. That handoff value is never used in withheld-overlap scoring. All 158 wellness observations remain separate, exact pseudo-point summaries, not verified instantaneous HR. Final unsupported timestamps remain missing.", ""]
    lines += markdown_table(["Supported mean BPM", "Supported max BPM", "Recorded/estimated s", "Supported n (%)", "Missing s", "Estimated step p99/max"], [
                            [f"{finite.mean():.3f}", f"{finite.max():.3f}", f"{table.edge_observed.sum()}/{table.estimated.sum()}", f"{len(finite)} ({100*len(finite)/len(table):.2f}%)", int(table.final_hr.isna().sum()), f"{step.quantile(.99):.3f}/{step.max():.3f}"]])
    lines += ["## Artifacts, verification and decision", "", "![Held-out comparison](images/hr_tcn_local_evaluation.png)", "", "![Target comparison](images/hr_tcn_local_candidate_2026-10-16.png)", "",
              "Implementation `scripts/evaluate_hr_tcn_local.py`; metrics `data/processed/hr_tcn_local_metrics.csv`; candidate `data/processed/hr_tcn_local_candidate_2026-10-16.csv`; final model `data/processed/hr_tcn_local_model.h5` (load compile=False); normalization, configuration and histories `data/processed/hr_tcn_local_normalization.json`. Separate failure-safe per-fit weights and config/source-hash sidecars are in `data/processed/hr_tcn_local_checkpoints/`. These are local outputs, never committed. `--resume` only accepts this local experiment's identity; it never searches the prior TCN checkpoint directory.", "",
              "```powershell", "conda run --no-capture-output -n deep python scripts/evaluate_hr_tcn_local.py", "```", ""]
    lines += ["- " + item for item in checks]
    lines += [f"- All {protected_n} previous files retain identical SHA-256, including prior TCN models/checkpoints, GB results, original data and README principal benchmark.", "- Exactly three 30-epoch fits; one untrained smoke test. Outputs checked nonempty; figures inspected after generation. No new dependencies or separate test suite.", "",
              "Decision screen fixed before fitting: prefer local TCN only with lower mean point MAE/RMSE on both rides than both prior models; permit at most +0.5 BPM high/recovery/stationary/restart, +1 BPM peak-amplitude and +2s timing error relative to GB, with no invalid HR or >5 BPM/s steps. Otherwise retain GB as principal benchmark. Neural-candidate tradeoffs remain visible without changing this principal-model decision. This is a descriptive consistency screen, not a significance test.", "",
              "Limitations: two independent MTB references with different sensors/seasons, correlated seconds/phases, sparse and uncertainly averaged wellness summaries, reference errors, native-versus-Strava terrain differences, padding near boundaries, absent target power/cadence and uncertain short peaks. Architectural RF is measured on input features, not raw physiological signals. Lower temporal loss is not proof of recovered HR changes, and changes in geometry, target sampling and objective are inseparable in this single experiment. No further configuration is attempted.", "", choice, ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run the requested local experiment with exactly three necessary fits."""
    configure_helpers()
    info = previous.environment()
    outputs = {Path(__file__).resolve(), METRICS, CANDIDATE,
               REPORT, FIGURE, TARGET_FIGURE, MODEL, NORMALIZATION}
    protected = [p for folder in ("data", "docs", "scripts") for p in (ROOT / folder).rglob("*") if p.is_file() and
                 "__pycache__" not in p.parts and p not in outputs and CHECKPOINTS not in p.parents]
    protected += [ROOT / "README.md", ROOT
                  / "requirements.txt", ROOT / "AGENTS.md"]
    before = base.fingerprints(protected)
    initial = None if "--resume" in sys.argv else smoke()
    gb = pd.read_csv(GB_METRICS)
    saved = gb.loc[gb.model.eq("seven_ride")].set_index(
        ["domain", "ride", "phase", "condition"])
    watch, _, _ = window.clean_observations(pd.read_csv(
        ROOT / "data/processed/watch_hr_ride_2026-10-16.csv"))
    pattern = np.diff(watch.index.asi8) // 1_000_000_000
    assert len(watch) == 158 and (pattern == 60).sum(
    ) == 142 and (pattern == 120).sum() == 15
    window.verify_windows()
    rides, packages = {}, {}
    for name in RIDES:
        ride = window.prepare(name)
        assert "heart_rate" not in ride["frame"]
        packs = base.synthetic(
            ride, pattern, base.activity_features(ride["frame"]))
        for package in packs.values():
            package["features"] = previous.raw_features(ride["frame"], package)
        rides[name], packages[name] = ride, packs
        print("Prepared", name, len(
            ride["frame"]), "recorded seconds", flush=True)
    checks = ["GPU matmul/convolutions confirmed in unchanged deep environment; exact 64×28→16×1 shapes and 6,753 parameters.",
              "One smoke test verifies all 16 outputs' actual 45s RF inside [0,63], crop alignment, stride 16/8 extraction, overlap averaging and gap-isolated nonoverlapping targets.",
              "Known loss examples verify Huber delta 5, absolute-HR temporal change, fixed 0.5 coefficient, missing/anchor endpoint masks, zero-pair batches and finite nonzero gradients only at valid pair endpoints."]
    fits, rows, panels = [], [], []
    for i, held in enumerate(MTB):
        names = tuple(n for n in RIDES if n != held)
        fitted = train(names, rides, packages, fits,
                       initial if i == 0 else None)
        values, cached = previous.evaluate(
            rides[held], packages[held], fitted, saved, "mtb_holdout", checks)
        rows.extend(values)
        panels.append((rides[held], cached))
        initial = None
        del fitted
        tf.keras.backend.clear_session()
    fitted = train(RIDES, rides, packages, fits)
    target = window.prepare("mtb_edge")
    target["frame"] = target["strava"]
    sparse = watch.loc[target["reference"].index[0]
        :target["reference"].index[-1]]
    package = base.examples(
        target["frame"], sparse, target["reference"].index, base.activity_features(target["frame"]))
    package["features"] = previous.raw_features(target["frame"], package)
    values, _ = previous.evaluate(
        target, {(0, "actual_watch"): package}, fitted, saved, "mtb_overlap", checks)
    rows.extend(values)
    table = previous.candidate(target, watch, fitted)
    checks.append("Separate candidate CSV read back: all original Edge values, 158 separate watch summaries and future anchors exact; original support/grid retained and trailing unsupported seconds missing.")
    metrics = comparisons(pd.DataFrame(rows))
    assert len(fits) == 3
    metrics.to_csv(METRICS, index=False)
    fitted["network"].save(MODEL, include_optimizer=False)
    loaded = tf.keras.models.load_model(MODEL, compile=False)
    assert all(np.array_equal(a, b) for a, b in zip(
        loaded.get_weights(), fitted["network"].get_weights()))
    # Check reload on an actual normalized target window without another fit.
    sample = previous.normalized(
        package["features"], fitted["normalization"])[None, :64]
    assert np.allclose(loaded(sample, training=False).numpy(), fitted["network"](
        sample, training=False).numpy(), atol=1e-7, rtol=1e-7)
    checks.append("Final H5 reloads with compile=False, weights and actual-input predictions match; no extra fit required. All new evaluation hashes/counts also match both saved benchmark CSVs.")
    payload = {**fitted["normalization"], "environment": info,
               "fits": fits, "config": CONFIG, "parameters": 6753}
    NORMALIZATION.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plots(panels, metrics, target, table)
    assert base.fingerprints(protected) == before
    report(metrics, fits, table, info, checks, len(protected))
    assert all(path.stat().st_size > 0 for path in outputs)
    assert base.fingerprints(protected) == before
    print(metrics.loc[metrics.condition.isin(["point", "actual_watch"])].groupby(["domain", "ride", "model"])[
          ["mae", "rmse", "high_mae", "recovery_mae", "stationary_mae"]].mean().round(4).to_string())
    print("Fit windows/durations/final losses:",
          [(f["windows"], f["seconds"], f["loss"][-1], f["huber_component"][-1], f["temporal_component"][-1]) for f in fits])
    print("Candidate mean/max/coverage:", table.final_hr.mean(),
          table.final_hr.max(), 100 * table.final_hr.notna().mean())
    print("Exactly three fits; preserved files:", len(
        protected), ";", recommendation(metrics))


if __name__ == "__main__":
    main()
