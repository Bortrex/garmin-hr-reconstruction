"""Evaluate one fixed TensorFlow TCN with exactly two holdouts and a final fit.

Run: conda run --no-capture-output -n deep python scripts/evaluate_hr_tcn_tensorflow.py
Reuse the local pure-Python Garmin SDK if it is absent from deep; never change
the Conda environment or refit the saved gradient-boosting benchmark.
"""

import hashlib
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
if importlib.util.find_spec("garmin_fit_sdk") is None:
    sdk = ROOT / ".venv/Lib/site-packages/garmin_fit_sdk/__init__.py"
    if not sdk.is_file():
        raise ImportError(
            "Install only garmin-fit-sdk==21.218.0 into deep; TensorFlow must remain unchanged")
    spec = importlib.util.spec_from_file_location(
        "garmin_fit_sdk", sdk, submodule_search_locations=[str(sdk.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["garmin_fit_sdk"] = module
    spec.loader.exec_module(module)

import numpy as np
import pandas as pd
import tensorflow as tf
import evaluate_hr_seven_ride_window as benchmark
from evaluate_hr_mtb_training import markdown_table
from audit_cycling_predictors import runs

window, base = benchmark.window, benchmark.base
ZONE, RIDES, MTB = window.ZONE, benchmark.RIDES, benchmark.MTB
SEED, WIDTH, OUTPUT, LEFT = 265, 256, 32, 112
TRAIN_STRIDE, INFER_STRIDE, DELTA = 32, 16, 5.0
NUMERIC = ["baseline", "sparse_hr", "since_anchor_s", "until_anchor_s", "anchor_gap_s",
           "speed5", "speed_change10", "stop_age_s", "restart_age_s", "climb_up", "climb_down"]
VALIDITY = [name for name in NUMERIC if name != "sparse_hr"]
BINARY = ["anchor_mask", "moving", "stationary", "movement_unknown", "terrain_valid",
          "record_valid", "movement_valid"]
FEATURES = NUMERIC + [name + "_valid" for name in VALIDITY] + BINARY
STEM = "hr_tcn_tensorflow"
METRICS = ROOT / "data/processed" / (STEM + "_metrics.csv")
CANDIDATE = ROOT / "data/processed" / (STEM + "_candidate_2026-10-16.csv")
REPORT = ROOT / "docs" / (STEM + "_evaluation.md")
FIGURE = ROOT / "docs/images" / (STEM + "_evaluation.png")
TARGET_FIGURE = ROOT / "docs/images" / (STEM + "_candidate_2026-10-16.png")
MODEL = ROOT / "data/processed" / (STEM + "_model.h5")
NORMALIZATION = ROOT / "data/processed" / (STEM + "_normalization.json")
CHECKPOINTS = ROOT / "data/processed" / (STEM + "_checkpoints")


def environment():
    """Require actual float32 TensorFlow GPU work, not only device enumeration."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No TensorFlow GPU; stop before model training")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.experimental.enable_tensor_float_32_execution(False)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_soft_device_placement(False)
    with tf.device("/GPU:0"):
        value = tf.matmul(tf.ones((64, 64), tf.float32),
                          tf.ones((64, 64), tf.float32))
        total = float(tf.reduce_sum(value).numpy())
    assert "GPU:0" in value.device and total == 262144
    info = {"os": platform.platform(), "python": sys.version, "executable": sys.executable,
            "tensorflow": tf.__version__, "keras": metadata.version("keras"),
            "devices": [str(gpu) for gpu in gpus],
            "gpu_details": [tf.config.experimental.get_device_details(gpu) for gpu in gpus],
            "operation_device": value.device, "operation_sum": total,
            "sdk": sys.modules["garmin_fit_sdk"].__file__,
            "numpy": np.__version__, "pandas": pd.__version__}
    try:
        info["tf_memory_bytes"] = tf.config.experimental.get_memory_info(
            "GPU:0")
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                                 "--format=csv,noheader"], capture_output=True, text=True, check=True)
        info["device_memory_inventory"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        info["memory_note"] = str(error)
    print("GPU environment:", json.dumps(info), flush=True)
    return info


def seed():
    """Use the same declared seed for each of the three fits."""
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


@tf.keras.utils.register_keras_serializable(package="hr_tcn")
def masked_huber(y_true, y_pred):
    """Average delta-5 BPM Huber loss over eligible central target seconds."""
    error = tf.squeeze(y_pred, -1) - y_true[..., 0]
    absolute = tf.abs(error)
    loss = tf.where(absolute <= DELTA, .5 * tf.square(error),
                    DELTA * (absolute - .5 * DELTA))
    mask = y_true[..., 1]
    return tf.reduce_sum(loss * mask) / tf.maximum(tf.reduce_sum(mask), 1.)


def model():
    """Build the single fixed non-causal architecture, with a 125s receptive field."""
    inputs = tf.keras.Input((WIDTH, len(FEATURES)),
                            dtype="float32", name="cycling_context")
    x = tf.keras.layers.Conv1D(
        16, 1, activation="relu", name="input_projection")(inputs)
    for dilation in (1, 2, 4, 8, 16):
        skip = x
        x = tf.keras.layers.Conv1D(16, 3, dilation_rate=dilation, padding="same",
                                   strides=1, activation="relu")(x)
        x = tf.keras.layers.Conv1D(
            16, 3, dilation_rate=dilation, padding="same", strides=1)(x)
        x = tf.keras.layers.Activation("relu")(
            tf.keras.layers.Add()([skip, x]))
    x = tf.keras.layers.Conv1D(
        1, 1, activation=None, name="residual_projection")(x)
    outputs = tf.keras.layers.Cropping1D((LEFT, LEFT), name="central_32")(x)
    result = tf.keras.Model(inputs, outputs, name="small_noncausal_tcn")
    result.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=.001), loss=masked_huber)
    assert result.output_shape == (None, OUTPUT, 1)
    return result


def pad_window(values, center):
    """Place a central output start at position 112, padding only inside its block."""
    start = center - LEFT
    low, high = max(0, start), min(len(values), start + WIDTH)
    result = np.zeros((WIDTH, values.shape[1]), dtype=np.float32)
    result[low - start:high - start] = values[low:high]
    return result


def overlap(length, centers, predictions):
    """Average central-32 predictions at stride 16 on matching block positions."""
    positions = centers[:, None] + np.arange(OUTPUT)[None, :]
    valid = positions < length
    total, count = np.zeros(length), np.zeros(length, dtype=int)
    np.add.at(total, positions[valid], predictions[valid])
    np.add.at(count, positions[valid], 1)
    assert (count > 0).all()
    return total / count


def smoke():
    """Check one untrained model, GPU convolutions, alignment, RF and masked loss."""
    seed()
    result = model()
    saved = result.get_weights()
    result.set_weights([np.full_like(weight, .01) for weight in saved])
    with tf.device("/GPU:0"):
        probe = tf.ones((1, WIDTH, len(FEATURES)), dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(probe)
            predicted = result(probe, training=False)
            middle = predicted[0, 16, 0]
        gradient = tape.gradient(middle, probe).numpy()[0]
    active = np.flatnonzero(np.any(gradient != 0, axis=1))
    assert predicted.shape == (1, OUTPUT, 1) and "GPU:0" in predicted.device
    assert active[0] == 66 and active[-1] == 190 and len(active) == 125
    result.set_weights(saved)
    numbers = np.arange(83, dtype=np.float32)[:, None]
    centers = np.arange(0, len(numbers), INFER_STRIDE)
    predicted = np.array([center + np.arange(OUTPUT)
                         for center in centers], dtype=float)
    assert np.array_equal(
        overlap(len(numbers), centers, predicted), numbers[:, 0])
    for center in centers:
        values = pad_window(numbers, int(center))
        width = min(OUTPUT, len(numbers) - center)
        assert np.array_equal(
            values[LEFT:LEFT + width, 0], numbers[center:center + width, 0])
    target = np.zeros((1, OUTPUT, 2), np.float32)
    target[0, 7, 1] = 1
    assert float(masked_huber(target, tf.ones(
        (1, OUTPUT, 1)) * 2).numpy()) == 2
    target[..., 1] = 0
    assert float(masked_huber(target, tf.ones((1, OUTPUT, 1))).numpy()) == 0
    print("Single smoke test passed; parameters:",
          result.count_params(), "RF: 125 seconds", flush=True)
    return result


def raw_features(frame, package):
    """Construct compact per-second inputs exclusively from non-HR data and anchors."""
    x = package["x"]
    result = x[[name for name in NUMERIC if name != "sparse_hr"]].copy()
    result["sparse_hr"] = package["anchors"].reindex(frame.index)
    result["anchor_mask"] = frame.index.isin(
        package["anchors"].index).astype(float)
    result["moving"] = frame.speed_raw.gt(.3).astype(
        float).where(frame.speed_raw.notna())
    for name in ("stationary", "movement_unknown", "terrain_valid"):
        result[name] = x[name]
    result["record_valid"] = 1.
    result["movement_valid"] = frame.speed_raw.notna().astype(float)
    return result[NUMERIC + BINARY]


def normalization(names, packages):
    """Fit means/scales from declared training rides and point features only."""
    values = np.concatenate([packages[name][phase, "point"]["features"][NUMERIC].to_numpy()
                             for name in names for phase in base.PHASES])
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    mean = np.where(finite, values, 0).sum(axis=0) / np.maximum(counts, 1)
    variance = np.where(finite, (values - mean)**2,
                        0).sum(axis=0) / np.maximum(counts, 1)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1
    return {"training": list(names), "mean": mean.tolist(), "scale": scale.tolist(),
            "features": FEATURES, "numeric": NUMERIC, "validity": VALIDITY, "binary": BINARY}


def normalized(features, norm):
    """Fill missing normalized values with zero and retain explicit validity flags."""
    values = features[NUMERIC].to_numpy(dtype=np.float32)
    valid = np.isfinite(values)
    values = (values - np.asarray(norm["mean"], np.float32)
              ) / np.asarray(norm["scale"], np.float32)
    values[~valid] = 0
    flags = features[VALIDITY].notna().to_numpy(dtype=np.float32)
    binary = features[BINARY].fillna(0).to_numpy(dtype=np.float32)
    result = np.concatenate([values, flags, binary], axis=1).astype(np.float32)
    assert result.shape[1] == len(FEATURES) and np.isfinite(result).all()
    return result


def block_indices(frame):
    """Yield positional blocks separated at every absent recorded second."""
    breaks = np.r_[0, np.flatnonzero(
        np.diff(frame.index.asi8) != 1_000_000_000) + 1, len(frame)]
    return [np.arange(left, right) for left, right in zip(breaks[:-1], breaks[1:])]


def training_windows(names, rides, packages, norm):
    """Use full 256s windows, stride 32, one cyclic phase and disjoint output targets."""
    inputs, targets, counts = [], [], {}
    for name in names:
        frame, reference = rides[name]["frame"], rides[name]["reference"]
        values = {phase: normalized(
            packages[name][phase, "point"]["features"], norm) for phase in base.PHASES}
        used, n = [], 0
        for block in block_indices(frame):
            for start in range(0, len(block) - WIDTH + 1, TRAIN_STRIDE):
                phase = base.PHASES[(start // TRAIN_STRIDE) % len(base.PHASES)]
                package = packages[name][phase, "point"]
                output = block[start + LEFT:start + LEFT + OUTPUT]
                times = frame.index[output]
                residual = reference.reindex(
                    times) - package["baseline"].linear.reindex(times)
                valid = residual.notna().to_numpy() & ~times.isin(
                    package["anchors"].index)
                if not valid.any():
                    continue
                target = np.column_stack(
                    [residual.fillna(0), valid]).astype(np.float32)
                inputs.append(values[phase][block[start:start + WIDTH]])
                targets.append(target)
                used.extend(times[valid])
                n += 1
        assert pd.DatetimeIndex(used).is_unique
        counts[name] = {"windows": n, "loss_seconds": len(used)}
    assert all(value["windows"] > 0 for value in counts.values())
    return np.stack(inputs), np.stack(targets), counts


class Progress(tf.keras.callbacks.Callback):
    """Emit bounded progress without any held-out monitoring or early stopping."""

    def on_epoch_end(self, epoch, logs=None):
        loss = float(logs["loss"])
        if not np.isfinite(loss):
            raise ValueError(
                "Nonfinite training loss; stop without a restart/search")
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/30; masked Huber {loss:.5f}", flush=True)


def train(names, rides, packages, fits, initial=None):
    """Run exactly one fixed 30-epoch fit with training-only normalization."""
    number = len(fits) + 1
    checkpoint = CHECKPOINTS / (f"fit_{number}.h5")
    state = CHECKPOINTS / (f"fit_{number}.json")
    source_hashes = {name: hashlib.sha256((ROOT / "data" / (name + ".fit")).read_bytes()).hexdigest()
                     for name in names}
    if "--resume" in sys.argv and checkpoint.is_file() and state.is_file():
        cached = json.loads(state.read_text(encoding="utf-8"))
        assert cached["normalization"]["training"] == list(names)
        assert cached["normalization"]["features"] == FEATURES
        assert cached["source_sha256"] == source_hashes
        fits.append(cached["fit"])
        print("Reusing completed fit", number,
              "without additional training", flush=True)
        return {"network": tf.keras.models.load_model(checkpoint, compile=False),
                "normalization": cached["normalization"], "training": names}
    norm = normalization(names, packages)
    x, y, counts = training_windows(names, rides, packages, norm)
    seed()
    network = initial if initial is not None else model()
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_threading.private_threadpool_size = 2
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=SEED,
                                                                 reshuffle_each_iteration=True).batch(32).with_options(options).prefetch(2)
    print("Training", "+".join(names), "windows", len(x), "loss seconds",
          sum(c["loss_seconds"] for c in counts.values()), flush=True)
    started = time.perf_counter()
    history = network.fit(dataset, epochs=30, verbose=0,
                          callbacks=[Progress()])
    duration = time.perf_counter() - started
    assert len(history.history["loss"]) == 30
    assert all(np.isfinite(w).all() for w in network.get_weights())
    fits.append({"training": list(names), "counts": counts, "windows": len(x),
                 "seconds": duration, "loss": [float(v) for v in history.history["loss"]]})
    CHECKPOINTS.mkdir(exist_ok=True)
    network.save(checkpoint, include_optimizer=False)
    state.write_text(json.dumps({"normalization": norm, "fit": fits[-1],
                                 "source_sha256": source_hashes}, indent=2), encoding="utf-8")
    print("Fit duration seconds:", round(duration, 2), flush=True)
    return {"network": network, "normalization": norm, "training": names}


def predict(frame, package, fitted):
    """Average overlapping residuals, then apply the unchanged per-anchor taper."""
    features = normalized(package["features"], fitted["normalization"])
    raw = pd.Series(np.nan, index=frame.index)
    for block in block_indices(frame):
        values = features[block]
        centers = np.arange(0, len(block), INFER_STRIDE)
        predictions = []
        for offset in range(0, len(centers), 64):
            batch = np.stack([pad_window(values, int(center))
                             for center in centers[offset:offset + 64]])
            predictions.append(fitted["network"](
                batch, training=False).numpy()[..., 0])
        predicted = np.concatenate(predictions)
        assert np.isfinite(predicted).all()
        raw.iloc[block] = overlap(len(block), centers, predicted)
    correction = pd.Series(np.nan, index=frame.index)
    for left, right in package["spans"]:
        times = frame.loc[left:right].index
        assert np.all(np.diff(times.asi8) == 1_000_000_000)
        mean = raw.loc[times].rolling(5, center=True, min_periods=1).mean()
        fraction = (times - left).total_seconds() / \
            (right - left).total_seconds()
        correction.loc[times] = mean * 4 * fraction * (1 - fraction)
    result = package["baseline"].linear + correction
    result.loc[package["anchors"].index] = package["anchors"].values
    assert result.notna().equals(package["baseline"].linear.notna())
    assert np.array_equal(
        result.loc[package["anchors"].index], package["anchors"].values)
    assert np.isfinite(result.dropna()).all()
    return result


def evaluate(ride, packages, fitted, saved, domain, checks):
    """Score the exact saved benchmark timestamps and retain phase/event metrics."""
    assert ride["name"] not in fitted["training"]
    assert ride["name"] not in fitted["normalization"]["training"]
    masks, peaks, _ = window.diagnostic_masks(ride)
    phases = base.PHASES if domain == "mtb_holdout" else (0,)
    conditions = ("point", "averaged") if domain == "mtb_holdout" else (
        "actual_watch",)
    rows, cached = [], {}
    for phase in phases:
        values = {condition: predict(
            ride["frame"], packages[phase, condition], fitted) for condition in conditions}
        support = ride["reference"].reindex(ride["frame"].index).notna()
        for condition in conditions:
            support &= values[condition].notna(
            ) & packages[phase, condition]["baseline"].linear.notna()
        common = support.copy()
        for condition in conditions:
            common.loc[packages[phase, condition]["anchors"].index] = False
        digest = hashlib.sha256(
            common.index.asi8[common].tobytes()).hexdigest()
        for condition, estimate in values.items():
            package = packages[phase, condition]
            old = saved.loc[(domain, ride["name"], phase, condition)]
            assert digest == old.evaluation_timestamp_sha256 and common.sum() == old.n
            metrics = window.score(ride, estimate, common, masks, peaks)
            linear_metrics = window.score(
                ride, package["baseline"].linear, common, masks, peaks)
            for key in ("high_n", "rise_n", "recovery_n", "transition_n", "peak_n"):
                assert metrics[key] == old[key]
            error = (
                estimate - ride["reference"].reindex(estimate.index)).where(common)
            for label, mask in (("stationary", ride["frame"].stationary_evidence),
                                ("restart", package["x"].restart_age_s.le(30))):
                selected = error[mask].dropna()
                metrics[label + "_n"], metrics[label +
                                               "_mae"] = len(selected), selected.abs().mean()
            metrics.update(base.extra_metrics(
                estimate, package["baseline"].linear, ride["reference"], support, peaks))
            finite = estimate.dropna()
            if not finite.gt(0).all() or not finite.lt(255).all():
                raise ValueError(
                    "TCN produced HR outside 0–255 BPM; no clipping or retuning")
            rows.append({"domain": domain, "ride": ride["name"], "phase": phase,
                         "condition": condition, "model": "TCN", "training": "+".join(fitted["training"]),
                         **metrics, "average_hr_error": metrics["bias"],
                         "supported_n": len(finite), "supported_percent": 100 * len(finite) / len(estimate),
                         "prediction_min": finite.min(), "prediction_max": finite.max(),
                         "evaluation_timestamp_sha256": digest, "metric_origin": "executed"})
            cached[phase, condition] = (package, estimate, common)
            # Retain the recomputed interpolation comparator without training it.
            linear_row = {**rows[-1], **linear_metrics,
                          "model": "linear", "training": "none"}
            linear_row.update(base.extra_metrics(package["baseline"].linear, package["baseline"].linear,
                                                 ride["reference"], support, peaks))
            linear_row["average_hr_error"] = linear_metrics["bias"]
            linear_row["prediction_min"] = package["baseline"].linear.min()
            linear_row["prediction_max"] = package["baseline"].linear.max()
            for label, mask in (("stationary", ride["frame"].stationary_evidence), ("restart", package["x"].restart_age_s.le(30))):
                selected = (package["baseline"].linear - ride["reference"].reindex(
                    estimate.index)).where(common & mask).dropna()
                linear_row[label + "_mae"] = selected.abs().mean()
            rows.append(linear_row)
    package, _, _ = next(iter(cached.values()))
    replacement = ride["reference"] * 0 + 200
    rebuilt = base.examples(ride["frame"], package["anchors"],
                            replacement.index, base.activity_features(ride["frame"]))
    pd.testing.assert_frame_equal(raw_features(
        ride["frame"], rebuilt), package["features"])
    checks.append(ride["name"] + ": held-out exclusion, training-only normalization, hidden-HR feature substitution, exact benchmark timestamp hash/event counts, matching support and exact anchors pass.")
    return rows, cached


def candidate(ride, watch, fitted):
    """Export a separate candidate, keeping every recorded observation unchanged."""
    frame = ride["frame"]
    edge = ride["reference"].reindex(frame.index)
    cutoff = ride["reference"].index[-1]
    anchors = pd.concat([pd.Series(edge.loc[cutoff], index=pd.DatetimeIndex(
        [cutoff])), watch.loc[watch.index > cutoff]])
    package = base.examples(frame, anchors, frame.index,
                            base.activity_features(frame))
    package["features"] = raw_features(frame, package)
    estimate = predict(frame, package, fitted)
    table = pd.DataFrame({"edge_hr": edge, "watch_hr": watch.reindex(frame.index),
                          "interpolation_baseline": package["baseline"].linear,
                          "model_estimate": estimate, "final_hr": estimate.where(edge.isna(), edge)})
    table["estimated"] = table.final_hr.notna() & edge.isna()
    table["edge_observed"], table["watch_observed"] = edge.notna(
    ), table.watch_hr.notna()
    table["source_status"] = "missing_unsupported"
    table.loc[table.estimated, "source_status"] = "experimental_tcn_estimate"
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
    assert table.final_hr.dropna().between(0, 255, inclusive="neither").all()
    table.to_csv(CANDIDATE)
    reread = pd.read_csv(CANDIDATE)
    assert np.array_equal(
        reread.loc[table.edge_observed.values, "final_hr"], edge.dropna())
    assert np.array_equal(
        reread.loc[table.watch_observed.values, "watch_hr"], watch)
    return table


def plots(panels, metrics, target, table):
    """Use fixed central segments and honest saved-score-only GB comparisons."""
    plt = window.plt
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    for ax, (ride, cached) in zip(axes[:2], panels):
        package, estimate, _ = cached[0, "point"]
        block = max(base.continuous_blocks(ride["reference"]), key=len)
        center = block.index[0] + (block.index[-1] - block.index[0]) / 2
        start = (center - pd.Timedelta(minutes=10)).floor("s")
        end = start + pd.Timedelta(minutes=20)
        actual = ride["reference"].loc[start:end]
        ax.plot(actual.index, actual, color=".65", lw=1.3,
                label="Recorded hidden HR reference")
        visible = estimate.loc[start:end]
        ax.plot(visible.index, visible, color="#c35b2d",
                lw=1.4, label="TCN estimate")
        linear = package["baseline"].linear.loc[start:end]
        ax.plot(linear.index, linear, color="#72804a",
                ls="--", lw=1, label="Linear interpolation")
        sparse = package["anchors"].loc[start:end]
        ax.scatter(sparse.index, sparse, color=".15", s=20,
                   zorder=5, label="Simulated point observations")
        ax.set(title=ride["name"] + " — held out; fixed central 20min, phase 0",
               ylabel="HR (BPM)", xlabel="Local time — Europe/Brussels", xlim=(start, end))
        ax.xaxis.set_major_formatter(
            window.mdates.DateFormatter("%H:%M", tz=ZONE))
        ax.legend(ncol=2, fontsize=8)
        ax.grid(alpha=.2)
    primary = metrics.loc[metrics.domain.eq(
        "mtb_holdout") & metrics.condition.eq("point")]
    for i, (name, color, label) in enumerate((("seven_ride", "#257c9b", "Seven-ride GB (saved)"), ("TCN", "#c35b2d", "TCN"))):
        groups = [primary.loc[primary.ride.eq(
            ride) & primary.model.eq(name)] for ride in MTB]
        means = np.array([g.mae.mean() for g in groups])
        low, high = np.array([g.mae.min() for g in groups]), np.array(
            [g.mae.max() for g in groups])
        bars = axes[2].bar(np.arange(2) + (i - .5) * .3, means, .28, color=color, label=label,
                           yerr=[means - low, high - means], capsize=3)
        for bar, group in zip(bars, groups):
            axes[2].annotate(f"MAE {group.mae.mean():.2f}\nRMSE {group.rmse.mean():.2f}",
                             (bar.get_x() + bar.get_width() / 2, group.mae.max() + .03), ha="center", fontsize=9)
    axes[2].set(xticks=[0, 1], xticklabels=list(MTB), ylabel="MAE (BPM)",
                title="Identical timestamp scores: means and dependent phase ranges")
    axes[2].set_ylim(0, primary.mae.max() + 1)
    axes[2].legend(loc="lower center", ncol=2)
    axes[2].grid(axis="y", alpha=.2)
    fig.suptitle("Single TensorFlow TCN experiment — 256s inputs → central 32s residuals\n"
                 "GB held-out curves/models were not saved; only its exact saved scores are available", fontsize=13)
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    ax.plot(table.index, table.final_hr.where(table.estimated),
            color="#c35b2d", lw=1.2, label="TCN estimate (not measured)")
    ax.plot(table.index, table.edge_hr, color="#257c9b",
            lw=1, label="Recorded Edge HR")
    watch = table.watch_hr.dropna()
    ax.scatter(watch.index, watch, color=".15", s=12,
               zorder=5, label="Recorded wellness summaries")
    ax.axvline(target["reference"].index[-1], color="#257c9b",
               ls="--", lw=1, label="Last recorded Edge HR")
    for i, indices in enumerate(runs(target["frame"].probable_stop, table.index)):
        times = table.index[indices]
        ax.axvspan(times[0], times[-1], color=".5", alpha=.09,
                   label="Probable sustained stops" if i == 0 else None)
    ax.set(title="October 16 MTB — separate experimental TensorFlow TCN candidate",
           ylabel="HR (BPM)", xlabel="Local time — Europe/Brussels", xlim=(table.index[0], table.index[-1]))
    ax.xaxis.set_major_formatter(window.mdates.DateFormatter("%H:%M", tz=ZONE))
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    ax.grid(alpha=.2)
    fig.savefig(TARGET_FIGURE, dpi=170)
    plt.close(fig)


def recommendation(metrics):
    """Require consistent gains and acceptable dynamics before suggesting promotion."""
    primary = metrics.loc[metrics.domain.eq(
        "mtb_holdout") & metrics.condition.eq("point")]
    passed = []
    for ride in MTB:
        mean = primary.loc[primary.ride.eq(ride)].groupby(
            "model").mean(numeric_only=True)
        old, new = mean.loc["seven_ride"], mean.loc["TCN"]
        passed.extend([new.mae < old.mae, new.rmse < old.rmse,
                       new.high_mae <= old.high_mae + .5, new.recovery_mae <= old.recovery_mae + .5,
                       new.peak_amplitude_mae <= old.peak_amplitude_mae + 1,
                       new.peak_timing_mae_s <= old.peak_timing_mae_s + 2,
                       new.stationary_mae <= old.stationary_mae + .5,
                       new.restart_mae <= old.restart_mae + .5])
    new = metrics.loc[metrics.model.eq("TCN")]
    passed.extend([new.invalid_hr_n.eq(0).all(),
                  new.steps_gt5_percent.eq(0).all()])
    return "B. Consider promoting the TCN, subject to user approval." if all(passed) else "A. Retain the gradient-boosting seven-ride benchmark."


def report(metrics, fits, table, info, checks, parameters, protected_n):
    """Report the one fixed experiment, all phases, durations and known constraints."""
    choice = recommendation(metrics)
    lines = ["# Small TensorFlow TCN for HR reconstruction", "", "**" + choice + "**", "",
             "Exactly three 30-epoch fits, seed 265: hold out MTB 1, hold out MTB 2, final all seven. No tuning, restart, early stopping, alternate architecture, GB refit or virtual data. The single untrained smoke test is not a training fit.", "",
             "## Environment and implementation", "", "```json", json.dumps(
                 info, indent=2), "```", "",
             "The Conda launcher requires execution outside the Codex sandbox on this host (inside it reports an illegal system DLL relocation); the experiment runs via `conda run --no-capture-output -n deep python`. No environment was created or modified. Deep lacks garmin-fit-sdk: reuse only the repository's existing pure-Python SDK package by file location, without adding the whole .venv to Python paths or loading its binary dependencies. If that local package is unavailable, the minimum standalone fix is `conda run -n deep python -m pip install garmin-fit-sdk==21.218.0`; it was not executed. TensorFlow/CUDA/cuDNN/Python were never reinstalled. [TensorFlow installation reference](https://www.tensorflow.org/install/pip).", "",
             f"One non-causal float32 network: input (256,{len(FEATURES)}); 1×1 projection to 16 channels; five residual blocks with two kernel-3 same/stride-1 convolutions at dilations 1,2,4,8,16, ReLU and skip-add; final linear 1×1 residual projection and crop [112:144]. Output (32,1). Parameters: {parameters}. Effective receptive field = 1 + 2×(3−1)×(1+2+4+8+16) = **125s**, radius 62. The central outputs have sufficient margin in an unpadded 256s window. GPU gradient support independently confirms exact RF endpoints. TF32 and mixed precision are disabled.", "",
             "Adam 0.001, masked Huber delta 5 BPM, batch 32, 30 epochs. Normalize loss by eligible central seconds; unsupported/missing-reference/declared-anchor targets have zero weight. No loss or reported output outside the central 32s. TensorFlow/Python/NumPy seeds are 265; deterministic-op and cuBLAS flags are set. GPU/library versions can still prevent bit-identical results; no repeated fit was run to claim deterministic training.", "",
             "## Inputs, sampling and normalization", "",
             "Numeric inputs: `" + ", ".join(NUMERIC) + "`. Finite-value flags for every numeric channel except sparse HR, whose anchor mask is explicit; binary channels: `" + ", ".join(BINARY) + "`. Total **" + str(len(
                 FEATURES)) + " channels**, not flattened GB temporal summaries. Non-HR quantities reuse existing definitions. Sparse HR uses only declared point/averaged/actual wellness anchors. No power, cadence, ride ID or dense HR predictor. Means/scales fit exclusively on training rides' point-feature streams over the three phases, never held-out data. Missing normalized values are zero with validity=0; padded records have all channels zero and record_valid=0, so missing terrain is never presented as observed flat terrain.", "",
             "Use exactly the actual ordered 142×60s + 15×120s watch interval sequence, cycling/truncating by elapsed UTC time; phases 0/20/40. Point anchors equal recorded HR; centered averages require all 60 recorded seconds [t−30,t+29], an unverified Garmin assumption used only as sensitivity. Train only point-anchor windows. Full 256s training windows lie inside continuous recorded blocks, stride **32s**, cyclically assigned to one phase (start/32 modulo 3); central output target spans do not overlap or duplicate seconds. This sequence-loss sampling is explicitly different from GB's five-second single-target selection. Boundary/short-block targets outside full training windows are not fabricated.", ""]
    rows = []
    for i, fitted in enumerate(fits, 1):
        rows.append([i, "+".join(fitted["training"]), fitted["windows"], sum(c["loss_seconds"]
                    for c in fitted["counts"].values()), f"{fitted['seconds']:.2f}", f"{fitted['loss'][0]:.4f}/{fitted['loss'][-1]:.4f}"])
    lines += markdown_table(["Fit", "Training rides", "Windows",
                            "Eligible central seconds", "Training s", "First/last train loss"], rows)
    rows = [[i, name, value["windows"], value["loss_seconds"]]
            for i, fitted in enumerate(fits, 1) for name, value in fitted["counts"].items()]
    lines += markdown_table(["Fit", "Source ride",
                            "Windows", "Loss seconds"], rows)
    lines += ["## Inference and comparable support", "",
              "For each continuous recorded block, central starts are 0,16,32,… (fixed inference stride 16). Extract 256s around each central start, padding absent left/right context with masked zeros; clip each 32s output to this same block. Average overlaps per actual second. No input or averaging crosses a recording gap. Then, independently in every supported ≤180s anchor interval, apply the existing centered five-sample residual mean (min_periods=1) and multiply by 4f(1−f). Add to linear HR and restore anchors exactly, including isolated ones. Never extrapolate beyond baseline support. Altitude invalidity remains explicit even when neighboring non-HR context is usable.", "",
              "TCN reconstruction support equals linear support exactly, including padded boundaries; common evaluation intersects point/averaged support with recorded reference HR and excludes both sets of anchor times. Every resulting UTC timestamp hash equals the saved seven-ride benchmark hash, with matching scored counts and reference-event counts. Hence no coverage loss or unavailable GB rescoring is hidden. One-second/phase observations remain correlated; only two MTB rides are independent holdouts.", "",
              "**Plot limitation:** the existing repository saved GB scores and the target candidate, but no held-out prediction curves or fitted GB models. Drawing those curves would require an additional prohibited fit. The evaluation panels honestly show reference, sparse observations, TCN and interpolation; their bottom panel compares exact saved GB versus TCN scores. No GB curve is fabricated. The same predefined central 20min of the longest continuous HR block is used, phase 0, without inspecting errors.", "",
              "## MTB validation", ""]
    held = metrics.loc[metrics.domain.eq("mtb_holdout")]
    rows = []
    for (ride, condition, phase), group in held.groupby(["ride", "condition", "phase"]):
        g = group.set_index("model")
        t, old = g.loc["TCN"], g.loc["seven_ride"]
        rows.append([ride, condition, phase, f"{old.mae:.3f}/{old.rmse:.3f}",
                    f"{t.mae:.3f}/{t.rmse:.3f}", f"{t.bias:+.3f}", f"{int(t.n)} ({t.coverage_percent:.2f}%)"])
    lines += markdown_table(["Held out", "Anchors", "Phase", "GB MAE/RMSE",
                            "TCN MAE/RMSE", "TCN bias", "Common n (%)"], rows)
    rows = []
    for (ride, condition, name), group in held.groupby(["ride", "condition", "model"]):
        rows.append([ride, condition, name, f"{group.mae.mean():.3f} [{group.mae.min():.3f},{group.mae.max():.3f}]",
                    f"{group.rmse.mean():.3f} [{group.rmse.min():.3f},{group.rmse.max():.3f}]"])
    lines += markdown_table(["Ride", "Anchors", "Model",
                            "Mean MAE [phase range]", "Mean RMSE [phase range]"], rows)
    lines += ["Established score-only masks and metrics are reused unchanged: upper-decile high HR; ≥5/≤−5 BPM over 30s for rise/recovery; ≥10 BPM-prominence reference peaks ≥120s apart. Peak errors use fully scored ±15s local maxima, so timing is bounded by that window. Unmatched predicted peaks lack reference peaks within ±30s. Step roughness uses consecutive supported seconds including anchors. Stationary and ≤30s restart errors require recorded reference HR. Full per-phase/scenario metrics and eligible event counts are in the CSV; NaN means unavailable, never zero error.", ""]
    rows = []
    for (ride, name), group in held.loc[held.condition.eq("point")].groupby(["ride", "model"]):
        rows.append([ride, name, f"{group.high_mae.mean():.2f}", f"{group.rise_mae.mean():.2f}/{group.recovery_mae.mean():.2f}", f"{group.peak_amplitude_mae.mean():.2f}/{group.peak_timing_mae_s.mean():.2f}",
                    f"{group.stationary_mae.mean():.2f}/{group.restart_mae.mean():.2f}", f"{group.unmatched_peaks.mean():.1f}", f"{group.step_p99.mean():.3f}/{group.max_step.max():.3f}", f"{group.steps_gt5_percent.max():.3f}"])
    lines += markdown_table(["Ride", "Model", "High MAE", "Rise/recovery", "Peak amp/timing",
                            "Stationary/restart", "Unmatched mean", "Step p99 mean/max", ">5 BPM/s % max"], rows)
    lines += ["## October 16 secondary overlap and candidate", "",
              "Frozen all-seven model, real watch anchors, existing movement/stopping variables and quality-masked Strava climb30. Overlap scoring uses only the 21 actual watch anchors inside the initial Edge recording; no withheld Edge value enters inputs. This repeatedly examined overlap is secondary and does not tune training. Real wellness summaries are pseudo-point constraints, not verified instantaneous HR.", ""]
    rows = []
    for _, r in metrics.loc[metrics.domain.eq("mtb_overlap")].iterrows():
        rows.append([r.model, f"{r.mae:.3f}/{r.rmse:.3f}", f"{r.bias:+.3f}", f"{r.high_mae:.2f}",
                    f"{r.recovery_mae:.2f}", f"{int(r.n)} ({r.coverage_percent:.2f}%)"])
    lines += markdown_table(["Model", "MAE/RMSE", "Bias",
                            "High MAE", "Recovery MAE", "Common n (%)"], rows)
    finite = table.final_hr.dropna()
    steps = table.final_hr.diff().abs().where(
        table.estimated & table.estimated.shift(fill_value=False))
    lines += ["The separate full candidate preserves every Edge value and all watch summaries in a separate column. After the recorded section, its last observed Edge value is only a forward handoff anchor, followed by subsequent watch anchors; that value never enters withheld-overlap prediction. Source/quality flags identify estimates, measurements and missing support.", ""]
    lines += markdown_table(["Mean BPM", "Max BPM", "Recorded/estimated seconds", "Support n (%)", "Missing seconds", "Estimated step p99/max"],
                            [[f"{finite.mean():.3f}", f"{finite.max():.3f}", f"{table.edge_observed.sum()}/{table.estimated.sum()}", f"{len(finite)} ({100*len(finite)/len(table):.2f}%)", int(table.final_hr.isna().sum()), f"{steps.quantile(.99):.3f}/{steps.max():.3f}"]])
    lines += ["## Artifacts, checks and decision", "",
              "![Held-out evaluation](images/hr_tcn_tensorflow_evaluation.png)", "",
              "![Separate target candidate](images/hr_tcn_tensorflow_candidate_2026-10-16.png)", "",
              "Script: `scripts/evaluate_hr_tcn_tensorflow.py`; metrics: `data/processed/hr_tcn_tensorflow_metrics.csv`; candidate: `data/processed/hr_tcn_tensorflow_candidate_2026-10-16.csv`. Final compact Keras model: `data/processed/hr_tcn_tensorflow_model.h5` (load with `compile=False`); normalization/features/environment/training histories: `data/processed/hr_tcn_tensorflow_normalization.json`. Prediction also requires this script's feature construction and postprocessing. No model file is committed.", "",
              "Failure-safe per-fit models and matching normalization/data hashes are retained in `data/processed/hr_tcn_tensorflow_checkpoints/` (fit_1/2/3.h5 and JSON sidecars; each model is under 100KB). `--resume` reuses completed fits after checking source hashes and training membership, without another training call. The reported experiment completed normally with exactly three fits; checkpoints were used only for read-only reproduction checks.", "",
              "```powershell", "conda run --no-capture-output -n deep python scripts/evaluate_hr_tcn_tensorflow.py", "```", ""]
    lines += ["- " + item for item in checks]
    lines += [f"- {protected_n} prior source/report/data files retain identical SHA-256; the principal README figure and benchmark remain unchanged.",
              "- Exactly three fits and one untrained input/model smoke test; actual outputs checked nonempty and figures inspected after generation. No separate suite or new dependency.", "",
              "Recommendation screen fixed before fitting: require lower phase-mean point MAE and RMSE on both rides, no >0.5 BPM high/recovery/stationary/restart regression, no >1 BPM peak-amplitude or >2s timing regression, and no invalid HR or >5 BPM/s jumps. These are descriptive screens, not significance tests or population proof. Averaged sensitivity, event tradeoffs and the secondary overlap remain visible regardless of the decision. No settings are selected from these results.", "",
              "Only two MTB references (different sensor paths/seasons), correlated phases/seconds, sparse and uncertainly averaged wellness anchors, optical/reference errors and uncertain short peaks limit inference. No target power/cadence; speed and terrain do not uniquely measure effort. Padding addresses support but cannot create observed boundary context. More detailed/smoother curves are not evidence of true missing physiology. GPU-level nondeterminism remains possible despite fixed seeds/flags. No additional architecture is attempted.", "", choice, ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Verify GPU, smoke once, fit exactly three times and preserve prior artifacts."""
    info = environment()
    outputs = {Path(__file__).resolve(), METRICS, CANDIDATE,
               REPORT, FIGURE, TARGET_FIGURE, MODEL, NORMALIZATION}
    protected = [p for folder in ("data", "docs", "scripts") for p in (ROOT / folder).rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and p not in outputs
                 and CHECKPOINTS not in p.parents]
    protected += [ROOT / "README.md", ROOT /
                  "requirements.txt", ROOT / "AGENTS.md"]
    before = base.fingerprints(protected)
    initial = None if "--resume" in sys.argv else smoke()
    parameters = 16 * len(FEATURES) + 16 + 10 * (3 * 16 * 16 + 16) + 17
    original_metrics = pd.read_csv(benchmark.METRICS)
    saved_rows = original_metrics.loc[original_metrics.model.eq(
        "seven_ride")].copy()
    saved = saved_rows.set_index(["domain", "ride", "phase", "condition"])
    assert len(saved) == 13
    watch, _, _ = window.clean_observations(pd.read_csv(
        ROOT / "data/processed/watch_hr_ride_2026-10-16.csv"))
    pattern = np.diff(watch.index.asi8) // 1_000_000_000
    assert len(watch) == 158 and (pattern == 60).sum(
    ) == 142 and (pattern == 120).sum() == 15
    window.verify_windows()
    rides, packages = {}, {}
    for name in RIDES:
        ride = window.prepare(name)
        frame = ride["frame"]
        assert "heart_rate" not in frame
        packs = base.synthetic(ride, pattern, base.activity_features(frame))
        for package in packs.values():
            package["features"] = raw_features(frame, package)
        rides[name], packages[name] = ride, packs
        print("Prepared", name, len(frame), "recorded seconds", flush=True)
    checks = ["Physical GPU and actual float32 GPU matrix multiplication and TCN convolutions verified; no CPU fallback.",
              "One smoke test: input/output shapes, exact 125s gradient RF, central-32 alignment, padded-block extraction, overlap averaging and masked Huber values pass.",
              "Original point/complete-average anchors and deterministic UTC schedules reused; exact gap/window/stop/terrain checks pass; raw features exclude hidden HR and unsupported target labels are masked."]
    fits, rows, panels = [], [], []
    for i, held in enumerate(MTB):
        names = tuple(name for name in RIDES if name != held)
        fitted = train(names, rides, packages, fits,
                       initial=initial if i == 0 else None)
        values, cached = evaluate(
            rides[held], packages[held], fitted, saved, "mtb_holdout", checks)
        rows.extend(values)
        panels.append((rides[held], cached))
        initial = None
        del fitted
        tf.keras.backend.clear_session()
    fitted = train(RIDES, rides, packages, fits)
    target = window.prepare("mtb_edge")
    target["frame"] = target["strava"]
    overlap_watch = watch.loc[target["reference"].index[0]:target["reference"].index[-1]]
    package = base.examples(target["frame"], overlap_watch, target["reference"].index,
                            base.activity_features(target["frame"]))
    package["features"] = raw_features(target["frame"], package)
    values, _ = evaluate(
        target, {(0, "actual_watch"): package}, fitted, saved, "mtb_overlap", checks)
    rows.extend(values)
    table = candidate(target, watch, fitted)
    checks.append("Candidate read back: every original Edge HR and all separate watch observations exact; future watch anchors exact, original support retained and unsupported trailing seconds missing.")
    metrics = pd.DataFrame(rows)
    # Verify recomputed linear scores across the environment before comparing GB.
    old_linear = original_metrics.loc[original_metrics.model.eq(
        "linear")].set_index(["domain", "ride", "phase", "condition"])
    current = metrics.loc[metrics.model.eq("linear")].set_index(
        ["domain", "ride", "phase", "condition"])
    for column in ("n", "mae", "rmse", "bias", "high_mae", "rise_mae", "recovery_mae", "peak_amplitude_mae", "peak_timing_mae_s"):
        assert np.allclose(current[column], old_linear.loc[current.index,
                           column], atol=1e-9, rtol=1e-9, equal_nan=True)
    saved_rows["metric_origin"] = "saved_seven_ride"
    metrics = pd.concat([metrics, saved_rows], ignore_index=True)
    assert len(fits) == 3 and len(metrics) == 39
    assert metrics.groupby(["domain", "ride", "phase"]).n.nunique().eq(1).all()
    metrics.to_csv(METRICS, index=False)
    fitted["network"].save(MODEL, include_optimizer=False)
    payload = {**fitted["normalization"], "environment": info, "fits": fits,
               "config": {"seed": SEED, "train_stride": TRAIN_STRIDE, "inference_stride": INFER_STRIDE,
                          "input_seconds": WIDTH, "output_seconds": OUTPUT, "huber_delta": DELTA,
                          "epochs": 30, "batch_size": 32, "learning_rate": .001, "parameters": parameters}}
    NORMALIZATION.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plots(panels, metrics, target, table)
    assert base.fingerprints(protected) == before
    checks.append("All 13 evaluated timestamp hashes and saved linear metrics reproduce benchmark support and scoring; no coverage loss, benchmark refit or invented GB trajectory.")
    report(metrics, fits, table, info, checks, parameters, len(protected))
    assert all(p.stat().st_size > 0 for p in outputs)
    assert base.fingerprints(protected) == before
    print(metrics.loc[metrics.condition.isin(["point", "actual_watch"])].groupby(["domain", "ride", "model"])[
          ["mae", "rmse", "bias", "high_mae", "recovery_mae"]].mean().round(4).to_string())
    print("Final target mean/max/coverage:", table.final_hr.mean(),
          table.final_hr.max(), 100 * table.final_hr.notna().mean())
    print("Training durations:", [f["seconds"]
          for f in fits], "; parameters:", parameters)
    print("Preserved files:", len(protected),
          "; exactly three fits;", recommendation(metrics))


if __name__ == "__main__":
    main()
