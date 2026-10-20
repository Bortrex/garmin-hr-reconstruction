"""Prepare or explicitly run a fixed, cumulative-budget 128-to-16 TCN search.

The default and --dry-run use only the standard library and never load a model.
Training is possible only with --run; --run --resume keeps the original budget.
"""

import time

# Start the budget before remaining imports, especially the lazy ML imports.
ENTRY = time.monotonic()

import argparse
import csv
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/tcn_128_search"
MAX_TRIALS = 16
MAX_TOTAL_MINUTES = 70
SCHEDULING_SECONDS = 65 * 60
WATCHDOG_SECONDS = 69 * 60
SEED = 265
TRAINING = tuple("road_ride_" + str(i) for i in range(1, 6)) + ("mtb_ride_1",)
VALIDATION = "mtb_ride_2"
DILATIONS = ((1, 2, 4, 8), (1, 2, 4, 8, 8), (1, 2, 4, 8, 12))
FIXED = dict(input_seconds=128, output_seconds=16, left=56, features=28,
             train_stride=16, inference_stride=8, kernel=3, convolutions=2,
             epochs=30, seed=265, huber_delta=5., temporal_seconds=10,
             lambda_temporal=.5, optimizer="Adam", dtype="float32")
OBJECTIVE = "mean point-anchor postprocessed validation MAE over phases 0/20/40"
GB_METRICS = ROOT / "data/processed/hr_seven_ride_window_metrics.csv"
OLD_METRICS = ROOT / "data/processed/hr_tcn_tensorflow_metrics.csv"


class BudgetExpired(RuntimeError):
    """Indicate that no further work may start in the active worker."""


def write_json(path, value):
    """Replace a small JSON record atomically on the same filesystem."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False),
                         encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    """Read an existing UTF-8 JSON record."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def geometry(dilations):
    """Check exact structural dependencies of every central output, without ML."""
    radius = 2 * sum(dilations)
    rf = 1 + 2 * radius
    for output in range(56, 72):
        active = {output}
        for dilation in reversed(dilations):
            for _ in range(2):
                active = {i + d for i in active
                          for d in (-dilation, 0, dilation)}
                assert min(active) >= 0 and max(active) < 128
        assert active == set(range(output - radius, output + radius + 1))
        assert len(active) == rf
    return rf


def parameter_count(channels, blocks):
    """Count the input, residual and output convolution weights and biases."""
    return 29 * channels + 2 * blocks * (3 * channels**2 + channels) + channels + 1


def schedule():
    """Freeze one mandatory configuration and 15 seeded draws without replacement."""
    choices = list(itertools.product(DILATIONS, (12, 16, 24),
                                     (.001, .0003), (32, 64)))
    mandatory = ((1, 2, 4, 8, 8), 16, .001, 32)
    choices.remove(mandatory)
    selected = [mandatory] + \
        random.Random(SEED).sample(choices, MAX_TRIALS - 1)
    trials = []
    for i, (dilations, channels, rate, batch) in enumerate(selected, 1):
        trials.append(dict(trial_id="trial_%02d" % i,
                           dilations=list(dilations), channels=channels,
                           learning_rate=rate, batch_size=batch,
                           receptive_field=geometry(dilations),
                           parameters=parameter_count(channels, len(dilations))))
    return dict(version=1, fixed=FIXED, training=list(TRAINING),
                validation=VALIDATION, objective=OBJECTIVE, max_trials=MAX_TRIALS,
                max_total_seconds=MAX_TOTAL_MINUTES * 60,
                scheduling_seconds=SCHEDULING_SECONDS,
                watchdog_seconds=WATCHDOG_SECONDS, trials=trials)


def describe():
    """Display the complete deterministic plan without TensorFlow or data loading."""
    plan = schedule()
    print(json.dumps(plan, indent=2))
    print("Results:", RESULTS)
    print("Schedule is written before worker startup on explicit --run.")
    print("No training or TensorFlow import; default is this dry run.")


def lock_results():
    """Hold an OS-released lock so simultaneous invocations cannot spend twice."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    handle = (RESULTS / "run.lock").open("a+b")
    handle.write(b"0")
    handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("Another search process holds the results lock")
    return handle


def mark_interrupted(reason):
    """Make abandoned active trials terminal; resume never retries a started fit."""
    for path in RESULTS.glob("trial_*/status.json"):
        record = read_json(path)
        if record["status"] == "running":
            record.update(status="incomplete", reason=reason)
            history = path.parent / "history.json"
            if history.exists():
                record["completed_epochs"] = len(
                    read_json(history).get("loss", []))
            write_json(path, record)


def supervise(resume):
    """Charge startup/import/preparation/training/scoring/writes to one budget."""
    lock = lock_results()
    process = None
    ledger_path = RESULTS / "runtime.json"
    prior, active = 0., False
    try:
        frozen = RESULTS / "schedule.json"
        if resume:
            if not frozen.exists() or not ledger_path.exists():
                raise RuntimeError(
                    "No initialized search to resume; use --run")
            assert read_json(frozen) == schedule(
            ), "Frozen schedule/code changed"
            ledger = read_json(ledger_path)
            if ledger.get("active"):
                # A machine/process crash has unknown unaccounted time. Fail closed.
                ledger.update(active=False, cumulative_seconds=max(4200., ledger["cumulative_seconds"]),
                              stop_reason="Unclean supervisor exit: budget exhausted conservatively")
                write_json(ledger_path, ledger)
                mark_interrupted(ledger["stop_reason"])
                raise RuntimeError(ledger["stop_reason"])
            prior = ledger["cumulative_seconds"]
            mark_interrupted("Interrupted trial retained; no automatic retry")
        else:
            existing = [p for p in RESULTS.iterdir() if p.name != "run.lock"]
            if existing:
                raise RuntimeError("Results already exist; use --run --resume")
            write_json(frozen, schedule())
        ledger = dict(active=True, cumulative_seconds=prior,
                      limit_seconds=4200, scheduling_seconds=3900,
                      watchdog_seconds=4140, stop_reason="running")
        active = True

        def persist(reason, running=True):
            ledger.update(active=running, cumulative_seconds=prior + time.monotonic() - ENTRY,
                          stop_reason=reason)
            write_json(ledger_path, ledger)

        persist("starting")
        publish_metrics()
        if all((RESULTS / t["trial_id"] / "status.json").exists()
               for t in schedule()["trials"]):
            persist("Every scheduled trial already has a terminal status", False)
            active = False
            return
        if prior + time.monotonic() - ENTRY >= SCHEDULING_SECONDS:
            persist("Scheduling budget exhausted; no worker started", False)
            active = False
            return
        deadline = ENTRY + SCHEDULING_SECONDS - prior
        kill_at = ENTRY + WATCHDOG_SECONDS - prior
        environment = os.environ.copy()
        environment["HR_TCN_SEARCH_WORKER"] = "1"
        # Windows uses one shared system monotonic clock across these processes.
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--deadline", str(deadline)]
        if resume:
            command.append("--resume")
        process = subprocess.Popen(command, cwd=ROOT, env=environment)
        reason = "worker completed"
        while process.poll() is None:
            persist("running")
            if time.monotonic() >= kill_at:
                reason = "69-minute cumulative watchdog stopped worker; reserve remains for shutdown"
                process.kill()
                process.wait(timeout=5)
                break
            try:
                process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                pass
        if process.returncode != 0 and reason == "worker completed":
            reason = "Worker exited with code %s; inspect worker_status.json" % process.returncode
        mark_interrupted(reason)
        publish_metrics()
        persist(reason, False)
        active = False
        print(reason, "; cumulative seconds:",
              round(ledger["cumulative_seconds"], 2))
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            # Ctrl+C also reaches the console worker; allow its small status write.
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        mark_interrupted(
            "Manual interruption; started trial will not be retried")
        publish_metrics()
        if active:
            persist("manual interruption", False)
            active = False
        print("Interrupted safely; completed trials retained. Resume with --run --resume.")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if active:
            mark_interrupted("Supervisor exception; no retry of active trial")
            publish_metrics()
            persist("supervisor exception", False)
        lock.close()


def check_time(deadline):
    """Refuse work after the conservative cumulative scheduling deadline."""
    if time.monotonic() >= deadline:
        raise BudgetExpired("65-minute cumulative scheduling deadline reached")


def source_hashes(deadline):
    """Identify all source data, helper code and historical metrics read by the run."""
    paths = list((ROOT / "scripts").glob("*.py"))
    paths += [ROOT / "AGENTS.md", ROOT / "requirements.txt", GB_METRICS, OLD_METRICS,
              ROOT / "data/processed/watch_hr_ride_2026-10-16.csv"]
    # prepare() reads each activity FIT; no target/virtual FIT is used here.
    paths += [ROOT / "data" / (name + ".fit")
              for name in TRAINING + (VALIDATION,)]
    hashes = {}
    for path in sorted(set(paths)):
        check_time(deadline)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                check_time(deadline)
                digest.update(chunk)
        hashes[path.relative_to(ROOT).as_posix()] = digest.hexdigest()
    return hashes


def training_arrays(previous, rides, packages, norm, deadline):
    """Reuse phase/mask definitions with full 128s inputs and disjoint 16s targets."""
    np, pd = previous.np, previous.pd
    inputs, targets, counts = [], [], {}
    for name in TRAINING:
        check_time(deadline)
        frame, reference = rides[name]["frame"], rides[name]["reference"]
        values = {p: previous.normalized(packages[name][p, "point"]["features"], norm)
                  for p in previous.base.PHASES}
        used, pair_n, n = [], 0, 0
        for block in previous.block_indices(frame):
            last = None
            for start in range(0, len(block) - 128 + 1, 16):
                check_time(deadline)
                phase = previous.base.PHASES[(start // 16) % 3]
                package = packages[name][phase, "point"]
                positions = block[start:start + 128]
                times = frame.index[block[start + 56:start + 72]]
                assert (
                    np.diff(frame.index[positions].asi8) == 1_000_000_000).all()
                assert last is None or times[0] == last + \
                    pd.Timedelta(seconds=1)
                last = times[-1]
                baseline = package["baseline"].linear.reindex(times)
                residual = reference.reindex(times) - baseline
                valid = residual.notna().to_numpy() & ~times.isin(
                    package["anchors"].index)
                if not valid.any():
                    continue
                inputs.append(values[phase][positions])
                targets.append(np.column_stack([residual.fillna(0), valid,
                                                baseline.fillna(0)]).astype(np.float32))
                used.extend(times[valid])
                pair_n += int((valid[10:] & valid[:-10]).sum())
                n += 1
        assert pd.DatetimeIndex(used).is_unique and n > 0
        counts[name] = dict(windows=n, loss_seconds=len(
            used), temporal_pairs=pair_n)
    x, y = np.stack(inputs), np.stack(targets)
    assert x.shape[1:] == (128, 28) and y.shape[1:] == (16, 3)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return x, y, counts


def build_model(local, config):
    """Build exactly one declared architecture with the unchanged combined loss."""
    tf = local.tf
    c = config["channels"]
    inputs = tf.keras.Input((128, 28), dtype="float32")
    x = tf.keras.layers.Conv1D(c, 1, activation="relu")(inputs)
    for dilation in config["dilations"]:
        skip = x
        x = tf.keras.layers.Conv1D(c, 3, padding="same", strides=1,
                                   dilation_rate=dilation, activation="relu")(x)
        x = tf.keras.layers.Conv1D(c, 3, padding="same", strides=1,
                                   dilation_rate=dilation)(x)
        x = tf.keras.layers.Activation("relu")(
            tf.keras.layers.Add()([skip, x]))
    x = tf.keras.layers.Conv1D(1, 1, activation=None)(x)
    network = tf.keras.Model(inputs, tf.keras.layers.Cropping1D((56, 56))(x))
    network.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
                    loss=local.combined_loss,
                    metrics=[local.huber_component, local.temporal_component])
    assert network.input_shape == (None, 128, 28)
    assert network.output_shape == (None, 16, 1)
    assert network.count_params() == config["parameters"]
    return network


def verify_actual_rf(local, network, config, deadline):
    """Before fitting, prove actual GPU gradient support for all central outputs."""
    tf, np = local.tf, local.np
    weights = network.get_weights()
    network.set_weights([np.full_like(w, .01) for w in weights])
    try:
        with tf.device("/GPU:0"):
            x = tf.ones((1, 128, 28), tf.float32)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                output = network(x, training=False)
                probes = [output[0, i, 0] for i in range(16)]
            radius = (config["receptive_field"] - 1) // 2
            for i, probe in enumerate(probes):
                check_time(deadline)
                active = np.flatnonzero(
                    np.any(tape.gradient(probe, x).numpy()[0] != 0, axis=1))
                assert np.array_equal(active, np.arange(
                    56 + i - radius, 57 + i + radius))
            del tape
        assert "GPU:0" in output.device
    finally:
        network.set_weights(weights)


def completed_records():
    """Read only committed completed trial records."""
    return [r for p in sorted(RESULTS.glob("trial_*/status.json"))
            for r in [read_json(p)] if r["status"] == "completed"]


def publish_metrics():
    """Rebuild the compact phase CSV from completed, individually committed trials."""
    records = completed_records()
    if not records:
        return
    temporary = RESULTS / "trial_metrics.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = None
        for record in records:
            with (RESULTS / record["trial_id"] / "metrics.csv").open(
                    encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if writer is None:
                    writer = csv.DictWriter(
                        output, fieldnames=reader.fieldnames)
                    writer.writeheader()
                assert writer.fieldnames == reader.fieldnames
                writer.writerows(reader)
    os.replace(temporary, RESULTS / "trial_metrics.csv")
    best = min(records, key=lambda r: (r["validation_mae"], r["trial_id"]))
    assert best.get(
        "checkpoint"), "A new leading checkpoint must exist before completion"
    assert (RESULTS / best["checkpoint"]).stat().st_size > 0
    write_json(RESULTS / "best.json", dict(trial_id=best["trial_id"],
               validation_mae=best["validation_mae"], checkpoint=best["checkpoint"],
               normalization="normalization.json", objective=OBJECTIVE,
               caveat="Selected on one tuning activity; no promotion or final fit"))


def worker(deadline, resume):
    """Execute only the frozen schedule in the supervised explicit-run process."""
    if os.environ.get("HR_TCN_SEARCH_WORKER") != "1":
        raise RuntimeError(
            "Internal worker must be launched by --run supervisor")
    check_time(deadline)
    import evaluate_hr_tcn_local as local
    previous = local.previous
    tf, np, pd = local.tf, local.np, local.pd
    previous.WIDTH, previous.OUTPUT, previous.LEFT = 128, 16, 56
    previous.TRAIN_STRIDE, previous.INFER_STRIDE = 16, 8
    plan = read_json(RESULTS / "schedule.json")
    assert plan == schedule() and len(previous.FEATURES) == 28
    hashes = source_hashes(deadline)
    metadata_path = RESULTS / "metadata.json"
    if resume and metadata_path.exists():
        assert read_json(metadata_path)[
            "source_sha256"] == hashes, "Input/code hashes changed"
    check_time(deadline)
    info = previous.environment()
    check_time(deadline)
    if resume and metadata_path.exists():
        old = read_json(metadata_path)["environment"]
        for key in ("python", "tensorflow", "keras", "numpy", "pandas", "gpu_details"):
            assert old[key] == info[key], "Resume environment changed: " + key
    write_json(metadata_path, dict(source_sha256=hashes, environment=info,
               fixed=FIXED, objective=OBJECTIVE, preprocessing_cache="in memory only"))
    watch, _, _ = previous.window.clean_observations(pd.read_csv(
        ROOT / "data/processed/watch_hr_ride_2026-10-16.csv"))
    pattern = np.diff(watch.index.asi8) // 1_000_000_000
    assert len(watch) == 158 and (pattern == 60).sum(
    ) == 142 and (pattern == 120).sum() == 15
    rides, packages = {}, {}
    for name in TRAINING + (VALIDATION,):
        check_time(deadline)
        ride = previous.window.prepare(name)
        assert "heart_rate" not in ride["frame"]
        packs = previous.base.synthetic(
            ride, pattern, previous.base.activity_features(ride["frame"]))
        for package in packs.values():
            package["features"] = previous.raw_features(ride["frame"], package)
        # Hidden reference substitution cannot alter declared-anchor predictors.
        package = packs[0, "point"]
        rebuilt = previous.base.examples(ride["frame"], package["anchors"],
                                         ride["reference"].index,
                                         previous.base.activity_features(ride["frame"]))
        pd.testing.assert_frame_equal(previous.raw_features(
            ride["frame"], rebuilt), package["features"])
        rides[name], packages[name] = ride, packs
        print("Prepared", name, flush=True)
    norm = previous.normalization(TRAINING, packages)
    assert VALIDATION not in norm["training"] and norm["training"] == list(
        TRAINING)
    normalization_path = RESULTS / "normalization.json"
    if resume and normalization_path.exists():
        assert read_json(
            normalization_path) == norm, "Preprocessing normalization changed"
    write_json(normalization_path, norm)
    x, y, counts = training_arrays(previous, rides, packages, norm, deadline)
    write_json(RESULTS / "preprocessing.json", dict(counts=counts, windows=len(x),
               input_shape=list(x.shape), label_shape=list(y.shape),
               pattern_seconds=pattern.tolist(), feature_order=previous.FEATURES))
    saved = pd.read_csv(GB_METRICS)
    saved = saved[saved.model.eq("seven_ride")].set_index(
        ["domain", "ride", "phase", "condition"])
    publish_metrics()

    class DeadlineCallback(tf.keras.callbacks.Callback):
        """Stop at batch boundaries and persist finished epochs without validation."""

        def __init__(self, folder):
            super().__init__()
            self.folder, self.values, self.expired = folder, {}, False

        def on_train_batch_begin(self, batch, logs=None):
            check_time(deadline)

        def on_train_batch_end(self, batch, logs=None):
            if time.monotonic() >= deadline:
                self.expired = True
                self.model.stop_training = True

        def on_epoch_end(self, epoch, logs=None):
            if self.expired:
                return  # A partially processed epoch is not a completed epoch.
            for key in ("loss", "huber_component", "temporal_component"):
                value = float(logs[key])
                if not np.isfinite(value):
                    raise ValueError("Nonfinite training " + key)
                self.values.setdefault(key, []).append(value)
            write_json(self.folder / "history.json", self.values)
            print(self.folder.name, "epoch", epoch
                  + 1, "loss", logs["loss"], flush=True)

    class BudgetPredictor:
        """Check the deadline before every reused inference batch."""

        def __init__(self, network):
            self.network = network

        def __call__(self, values, training=False):
            check_time(deadline)
            return self.network(values, training=training)

    for config in plan["trials"]:
        check_time(deadline)
        folder = RESULTS / config["trial_id"]
        if (folder / "status.json").exists():
            # Completed, failed and incomplete trials are all terminal.
            continue
        completed = completed_records()
        if completed:
            # Measured end-to-end costs, including validation/writes, not epoch guesses.
            estimate = 1.5 * max(r["runtime_seconds"] for r in completed)
            if time.monotonic() + estimate >= deadline:
                print(
                    "Insufficient scheduling budget for measured next-trial cost", flush=True)
                break
        folder.mkdir(exist_ok=True)
        record = dict(**config, status="running",
                      reason="", completed_epochs=0)
        write_json(folder / "status.json", record)
        started = time.monotonic()
        callback = DeadlineCallback(folder)
        network, fit_started, fit_ended = None, None, None
        try:
            tf.keras.backend.clear_session()
            previous.seed()
            network = build_model(local, config)
            verify_actual_rf(local, network, config, deadline)
            check_time(deadline)
            options = tf.data.Options()
            options.experimental_deterministic = True
            options.experimental_threading.private_threadpool_size = 2
            dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(
                len(x), seed=SEED, reshuffle_each_iteration=True).batch(
                    config["batch_size"]).with_options(options).prefetch(2)
            fit_started = time.monotonic()
            network.fit(dataset, epochs=30, verbose=0, callbacks=[callback])
            fit_ended = time.monotonic()
            if callback.expired or len(callback.values.get("loss", [])) != 30:
                raise BudgetExpired(
                    "Training interrupted by time budget; not rankable")
            check_time(deadline)
            assert all(np.isfinite(w).all() for w in network.get_weights())
            fitted = dict(network=BudgetPredictor(network),
                          normalization=norm, training=TRAINING)
            checks = []
            rows, cached = previous.evaluate(rides[VALIDATION], packages[VALIDATION],
                                             fitted, saved, "mtb_holdout", checks)
            del cached, fitted
            metrics = pd.DataFrame(rows)
            metrics = metrics[metrics.model.eq("TCN")].copy()
            assert len(metrics) == 6
            metrics["trial_id"] = config["trial_id"]
            metrics["model"] = "TCN_128_search"
            point = metrics[metrics.condition.eq("point")]
            averaged = metrics[metrics.condition.eq("averaged")]
            assert len(point) == len(averaged) == 3
            means = point.select_dtypes(include="number").mean()
            for key in means.index:
                value = float(means[key])
                record["validation_"
                       + key] = value if np.isfinite(value) else None
            record["averaged_mae"] = float(averaged.mae.mean())
            record["averaged_rmse"] = float(averaged.rmse.mean())
            assert np.isfinite(record["validation_mae"])
            check_time(deadline)
            previous_best = min((r["validation_mae"]
                                for r in completed), default=float("inf"))
            if record["validation_mae"] < previous_best:
                network.save(folder / "model.h5", include_optimizer=False)
                record["checkpoint"] = config["trial_id"] + "/model.h5"
            metrics.to_csv(folder / "metrics.csv", index=False)
            record.update(status="completed", checks=checks,
                          completed_epochs=30, training_seconds=fit_ended - fit_started,
                          final_huber=callback.values["huber_component"][-1],
                          final_temporal=callback.values["temporal_component"][-1],
                          final_total=callback.values["loss"][-1])
        except (BudgetExpired, KeyboardInterrupt) as error:
            record.update(status="incomplete", reason=str(
                error) or "Manual interruption")
        except Exception as error:
            record.update(status="failed", reason=type(
                error).__name__ + ": " + str(error))
            (folder / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            record["completed_epochs"] = len(callback.values.get("loss", []))
            if fit_started is not None:
                record["training_seconds"] = (
                    fit_ended or time.monotonic()) - fit_started
            record["runtime_seconds"] = time.monotonic() - started
            write_json(folder / "status.json", record)
            publish_metrics()
            # Charge checkpoint/CSV/status writing and session cleanup to trial cost.
            network = None
            fitted = None
            callback = None
            tf.keras.backend.clear_session()
            gc.collect()
            record["runtime_seconds"] = time.monotonic() - started
            write_json(folder / "status.json", record)
        if record["status"] == "incomplete":
            break
    write_json(RESULTS / "worker_status.json", dict(status="stopped", completed=len(completed_records()),
               reason="Schedule finished, no measured-cost room, or time/interruption stop"))


def main():
    """Keep all default/help/dry-run paths free of TensorFlow and training."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Show frozen plan; no ML imports or writes")
    mode.add_argument("--run", action="store_true",
                      help="Explicitly launch budgeted GPU training")
    parser.add_argument("--resume", action="store_true",
                        help="With --run, reuse schedule/budget and skip all started trials")
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.deadline, args.resume)
        except BaseException as error:
            write_json(RESULTS / "worker_status.json", dict(status="stopped",
                       reason=type(error).__name__ + ": " + str(error)))
            raise
    elif args.run:
        supervise(args.resume)
    elif args.resume:
        parser.error("--resume requires the explicit --run flag")
    else:
        describe()


if __name__ == "__main__":
    main()
