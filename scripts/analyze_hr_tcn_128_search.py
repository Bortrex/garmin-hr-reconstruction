"""Analyze saved 128-to-16 TCN search results without TensorFlow or any fitting."""

import argparse
import csv
import hashlib

from search_hr_tcn_128 import (
    GB_METRICS, OLD_METRICS, RESULTS, ROOT, VALIDATION, read_json, schedule,
)

REPORT = ROOT / "docs/hr_tcn_128_search_results.md"
FIGURE = ROOT / "docs/images/hr_tcn_128_search.png"
LABELS = {"gb": "Historical seven-ride GB", "tcn": "Historical 256-to-32 TCN"}
DIAGNOSTICS = ("mae", "rmse", "bias", "high_mae", "rise_mae", "recovery_mae",
               "peak_amplitude_mae", "peak_timing_mae_s", "unmatched_peaks",
               "stationary_mae", "restart_mae", "step_p99", "max_step",
               "steps_gt5_percent", "supported_percent")


def historical():
    """Read actual saved MTB-2 phase scores; never invent historical predictions."""
    values = {}
    metadata = RESULTS / "metadata.json"
    hashes = read_json(metadata)["source_sha256"] if metadata.exists() else {}
    for key, path, model in (("gb", GB_METRICS, "seven_ride"),
                             ("tcn", OLD_METRICS, "TCN")):
        if hashes:
            assert hashlib.sha256(path.read_bytes()).hexdigest(
            ) == hashes[path.relative_to(ROOT).as_posix()]
        with path.open(encoding="utf-8", newline="") as stream:
            rows = [r for r in csv.DictReader(stream)
                    if r["ride"] == VALIDATION and r["domain"] == "mtb_holdout" and
                    r["model"] == model and r["condition"] == "point"]
        assert len(rows) == 3 and {int(r["phase"])
                                   for r in rows} == {0, 20, 40}
        values[key] = {metric: sum(float(r[metric]) for r in rows) / 3
                       for metric in DIAGNOSTICS}
        values[key]["hashes"] = {
            int(r["phase"]): r["evaluation_timestamp_sha256"] for r in rows}
    assert values["gb"]["hashes"] == values["tcn"]["hashes"]
    return values


def inspect_only():
    """Describe analysis inputs and destinations without creating search results."""
    print("Inputs:", RESULTS, "(exists:", RESULTS.exists(), ")")
    print("Completed trial records:", sum(read_json(p)["status"] == "completed"
                                          for p in RESULTS.glob("trial_*/status.json")))
    for key, values in historical().items():
        print(LABELS[key], "MTB 2 point phase-mean MAE/RMSE:",
              "%.6f / %.6f" % (values["mae"], values["rmse"]))
    print("Future outputs:", RESULTS / "leaderboard.csv", RESULTS / "trial_status.csv",
          FIGURE, REPORT, sep="\n")
    print("No TensorFlow import, fitting, result generation or fabricated trial scores.")


def collect(pd, plan, context):
    """Rank only completed trials with all six phase/condition scores committed."""
    rows, statuses = [], []
    for config in plan["trials"]:
        path = RESULTS / config["trial_id"] / "status.json"
        record = read_json(path) if path.exists() else dict(
            **config, status="not_started", reason="No fit started")
        statuses.append({k: record.get(k) for k in (
            "trial_id", "status", "reason", "completed_epochs", "training_seconds", "runtime_seconds")})
        if record["status"] != "completed":
            continue
        assert record["completed_epochs"] == 30
        metrics = pd.read_csv(path.parent / "metrics.csv")
        assert len(metrics) == 6 and metrics.trial_id.eq(
            config["trial_id"]).all()
        assert set(zip(metrics.phase, metrics.condition)) == {
            (p, c) for p in (0, 20, 40) for c in ("point", "averaged")}
        point = metrics[metrics.condition.eq("point")].sort_values("phase")
        averaged = metrics[metrics.condition.eq("averaged")]
        assert point.n.gt(0).all() and point.mae.notna(
        ).all() and point.rmse.notna().all()
        assert dict(zip(point.phase, point.evaluation_timestamp_sha256)
                    ) == context["gb"]["hashes"]
        means = point[list(DIAGNOSTICS)].mean()
        assert abs(means.mae - record["validation_mae"]) < 1e-10
        rows.append({**config, "dilations": "/".join(map(str, config["dilations"])),
                     **{k: record[k] for k in ("status", "completed_epochs", "training_seconds",
                                               "runtime_seconds", "final_huber", "final_temporal", "final_total")},
                     **means.to_dict(), "mae_min": point.mae.min(), "mae_max": point.mae.max(),
                     "averaged_mae": averaged.mae.mean(), "averaged_rmse": averaged.rmse.mean()})
    leaderboard = pd.DataFrame(rows)
    if len(leaderboard):
        leaderboard = leaderboard.sort_values(
            ["mae", "trial_id"]).reset_index(drop=True)
        leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    else:
        leaderboard = pd.DataFrame(
            columns=["rank", "trial_id", "status", "mae", "rmse"])
    return leaderboard, pd.DataFrame(statuses)


def plot(board, context):
    """Show saved point-anchor scores and uncombined complexity/peak tradeoffs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ids = board.trial_id.str.replace("trial_", "", regex=False)
    for ax, metric, label in ((axes[0, 0], "mae", "MAE (BPM)"),
                              (axes[0, 1], "rmse", "RMSE (BPM)")):
        ax.bar(ids, board[metric], color="#d18437")
        for key, color in (("gb", "#26729a"), ("tcn", "#65874d")):
            ax.axhline(context[key][metric], color=color,
                       linestyle="--", label=LABELS[key])
        ax.set(xlabel="Trial ID, ranked by point MAE", ylabel=label)
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend(fontsize=8)
    for channels, group in board.groupby("channels"):
        axes[1, 0].scatter(group.receptive_field, group.mae,
                           s=group.parameters / 70, alpha=.7, label="%d channels" % channels)
    for _, row in board.iterrows():
        short = row.trial_id.replace("trial_", "")
        axes[1, 0].annotate(short, (row.receptive_field, row.mae), xytext=(3, 3),
                            textcoords="offset points", fontsize=8)
        axes[1, 1].annotate(short, (row.recovery_mae, row.unmatched_peaks), xytext=(3, 3),
                            textcoords="offset points", fontsize=8)
    axes[1, 0].set(
        xlabel="Receptive field (s); marker area tracks parameters", ylabel="Point MAE (BPM)")
    axes[1, 0].legend(fontsize=8)
    points = axes[1, 1].scatter(board.recovery_mae, board.unmatched_peaks,
                                c=board.mae, cmap="viridis", s=65)
    fig.colorbar(points, ax=axes[1, 1], label="Point MAE (BPM)")
    axes[1, 1].set(xlabel="Recovery MAE (BPM)",
                   ylabel="Mean unmatched predicted peaks")
    for ax in axes.flat:
        ax.grid(alpha=.2, axis="y")
    fig.suptitle("128-to-16 TCN search: one MTB tuning activity, three correlated phases\n"
                 "Historical lines are saved scores; no independent generalization test", fontsize=12)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)


def markdown_table(frame):
    """Render a small table without introducing the optional tabulate dependency."""
    def cell(value):
        return str(value).replace("|", "/").replace("\n", " ")
    rows = ["| " + " | ".join(map(cell, frame.columns)) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |"]
    rows += ["| " + " | ".join(map(cell, values)) + " |"
             for values in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def analyze():
    """Generate an honest leaderboard/report from completed results, without ML."""
    import pandas as pd
    if not (RESULTS / "schedule.json").exists():
        raise SystemExit(
            "No search outputs yet. Run the explicit search first; --dry-run only inspects inputs.")
    if read_json(RESULTS / "runtime.json").get("active"):
        raise SystemExit(
            "Search is active. Analyze after it stops to avoid inconsistent snapshots.")
    plan = read_json(RESULTS / "schedule.json")
    assert plan == schedule(), "Search schedule differs from this implementation"
    context = historical()
    board, statuses = collect(pd, plan, context)
    board.to_csv(RESULTS / "leaderboard.csv", index=False)
    statuses.to_csv(RESULTS / "trial_status.csv", index=False)
    ledger = read_json(RESULTS / "runtime.json")
    lines = ["# Budget-limited 128-to-16 TCN search", "",
             "Selection uses only MTB ride 2: mean postprocessed point-anchor MAE across phases 0/20/40. "
             "Five roads plus MTB 1 train each configuration; phases and seconds are correlated. "
             "This is tuning performance, not independent generalization evidence. No winner is promoted or retrained.", "",
             "Completed %d/%d scheduled trials. Charged runtime %.2f/70 minutes. Stop reason: %s."
             % (len(board), len(plan["trials"]), ledger["cumulative_seconds"] / 60, ledger["stop_reason"]), "",
             "Historical MTB-2 point phase means (saved results, unchanged evaluation hashes):", "",
             markdown_table(pd.DataFrame([dict(model=LABELS[k], **{m: round(v[m], 4)
                                                                   for m in DIAGNOSTICS}) for k, v in context.items()])), ""]
    if len(board):
        best = board.iloc[0]
        lines += ["Leading completed candidate: **%s**, dilations %s, %d channels, LR %g, batch %d, "
                  "RF %ds and %d parameters; MAE %.4f, RMSE %.4f BPM."
                  % (best.trial_id, best.dilations, best.channels, best.learning_rate, best.batch_size,
                     best.receptive_field, best.parameters, best.mae, best.rmse), "",
                  markdown_table(board[["rank", "trial_id", "dilations", "channels", "learning_rate",
                                        "batch_size", "parameters", "mae", "rmse", "recovery_mae", "peak_amplitude_mae",
                                        "peak_timing_mae_s", "unmatched_peaks", "stationary_mae"]].round(4)), ""]
        if len(board) > 1:
            gap = board.iloc[1].mae - best.mae
            lines += ["The first-to-second MAE gap is %.4f BPM. %s No significance or generalization claim follows."
                      % (gap, "Below 0.1 BPM is descriptively close and inconclusive here."
                         if gap < .1 else "A numerical ranking on this one tuning ride is not a population comparison."), ""]
            peak_difference = best.unmatched_peaks - \
                board.iloc[1].unmatched_peaks
            lines += ["The lower-MAE leader has %+.2f mean unmatched peaks versus the runner-up. "
                      "This %s a false-peak tradeoff between these two candidates."
                      % (peak_difference, "shows" if peak_difference > 0 else "does not show"), ""]
        for column, label in (("receptive_field", "Receptive field"), ("channels", "Channel count")):
            grouped = board.groupby(column).agg(trials=("mae", "size"),
                                                mean_mae=("mae", "mean"), best_mae=("mae", "min"))
            lines += [label + " descriptive groups:", "",
                      markdown_table(grouped.reset_index().round(4)), ""]
        for key, values in context.items():
            lines += ["Leading trial versus %s: MAE difference %+.4f BPM; unmatched peaks difference %+.2f, "
                      "recovery MAE difference %+.4f BPM; peak amplitude/timing differences %+.4f BPM / %+.2fs."
                      % (LABELS[key], best.mae - values["mae"], best.unmatched_peaks - values["unmatched_peaks"],
                         best.recovery_mae
                         - values["recovery_mae"], best.peak_amplitude_mae
                         - values["peak_amplitude_mae"],
                         best.peak_timing_mae_s - values["peak_timing_mae_s"]), ""]
        lines += ["RF/channel groups mix learning rates and batch sizes and contain unequal trial counts. "
                  "Their means and the complexity scatter describe only completed sampled configurations; "
                  "they cannot establish that larger RFs or models cause improvement. "
                  "Inspect the unmatched-peak column and recovery scatter alongside the MAE ranking; "
                  "lower MAE with more unmatched peaks is a visible tradeoff, not a new combined score.", "",
                  "![Search comparison](images/hr_tcn_128_search.png)", ""]
        plot(board, context)
    else:
        lines += ["No completed trials: no ranking, leading candidate or performance figure can be produced.", ""]
    excluded = statuses[statuses.status.ne("completed")]
    lines += ["Unranked trials (never assigned a validation rank):", "",
              markdown_table(excluded) if len(excluded) else "None.", "",
              "Per-phase point and averaged-anchor sensitivity metrics remain in `results/tcn_128_search/trial_metrics.csv` "
              "and each completed trial's `metrics.csv`; aggregate CSV is recoverable from those committed trial files. "
              "Leaderboard includes phase MAE min/max and averaged-anchor means. "
              "Phase spread is not a confidence interval. Histories record training losses, never the selection objective.", "",
              "Only the fixed six training rides enter normalization. No MyWhoosh, October 16 fitting, "
              "final all-seven fit or additional model experiment is performed. "
              "GPU nondeterminism, sensor/reference error, uncertain wellness averaging, sparse anchors and "
              "selection optimism remain limitations. Historical comparisons provide context only. "
              "Independent validation on new activities is needed before promotion.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", RESULTS / "leaderboard.csv",
          RESULTS / "trial_status.csv", REPORT)
    if len(board):
        print("Figure:", FIGURE)


def main():
    """Expose inspection and analysis only; no training code is imported."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect inputs/paths; write nothing")
    args = parser.parse_args()
    if args.dry_run:
        inspect_only()
    else:
        analyze()


if __name__ == "__main__":
    main()
