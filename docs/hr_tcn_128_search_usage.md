# Prepared 128-to-16 TCN search

**Implementation only: no search or training was executed while preparing these
scripts.** Use the existing Conda `deep` environment. No dependency installation,
environment change, winner promotion or final seven-ride fit is included.

From the repository root:

```powershell
conda run --no-capture-output -n deep python scripts/search_hr_tcn_128.py --help
conda run --no-capture-output -n deep python scripts/search_hr_tcn_128.py --dry-run
```

Default behavior is also a dry run. Neither path imports TensorFlow, reads FIT
records, writes experiment outputs or fits a model. The ordered schedule is
displayed; the explicit run freezes it to JSON before starting its worker.

Start training yourself:

```powershell
conda run --no-capture-output -n deep python scripts/search_hr_tcn_128.py --run
```

Resume after a normal interruption:

```powershell
conda run --no-capture-output -n deep python scripts/search_hr_tcn_128.py --run --resume
```

Analyze committed results after the search stops:

```powershell
conda run --no-capture-output -n deep python scripts/analyze_hr_tcn_128_search.py
```

The analysis script also supports `--help` and a write-free `--dry-run`.
It never imports TensorFlow or fits a model.

## Fixed experiment

Train on road rides 1–5 and MTB ride 1. **MTB ride 2 is the sole tuning activity**
and never enters training or normalization. No MyWhoosh or October 16 activity is
used for fitting. Only the already recorded October 16 watch interval sequence
is reused to define synthetic observations.

The search space has 36 combinations:

| Setting | Choices |
|---|---|
| Dilations / receptive field | 1,2,4,8 / 61s; 1,2,4,8,8 / 93s; 1,2,4,8,12 / 109s |
| Channels | 12, 16, 24 |
| Adam learning rate | 0.001, 0.0003 |
| Batch size | 32, 64 |

Exactly 16 unique configurations are scheduled. The first is dilations
1,2,4,8,8, 16 channels, LR 0.001, batch 32. The other 15 are sampled without
replacement using an isolated Python random generator seeded 265. There is no
adaptive expansion, repeat seed, early stopping based on MTB 2, or extra fit.

All configurations use float32, 128×28 inputs, central output positions 56–71,
16 residual predictions, training stride 16, inference stride 8, kernel 3,
two convolutions per residual block, ReLU, skip connections, a linear projection,
30 epochs and seed 265. RF = 1 + 4×sum(dilations). The largest RF covers input
positions 2–125 across all central outputs. Structural dependency sets are checked
in the dry run. Actual GPU gradient support for every output is checked before
each future fit, using temporary positive weights that are restored afterward.

Reuse the existing 28 predictors, training-only normalization, continuous blocks,
point-anchor cyclic phases 0/20/40, exact 142×60s + 15×120s sampling pattern,
60-second complete-window averaged-anchor sensitivity, ≤180-second interpolation
support, boundary masks and no extrapolation. No hidden HR, power or cadence is
added to predictors. Inputs/labels are prepared once in memory per invocation;
no redundant dense datasets are saved. Resume recomputes deterministic preprocessing.

The existing local TCN loss functions are imported unchanged:
masked Huber(delta 5) + 0.5×MAE of 10-second absolute-HR changes, with both endpoints
valid and unobserved inside a single 16-second output. Inference retains overlap
averaging, the centered five-sample residual mean, 4f(1−f) anchor taper and exact
declared observations. Historical timestamp hashes and event counts must match.

## Cumulative runtime and interruption

The maximum is **70 cumulative minutes**, including Python startup within the
script, imports, preprocessing, RF checks, training, validation and output writes.
Conda's launch time before Python starts and the separately requested analysis
are outside the script's clock. A parent process uses a monotonic clock while
the worker imports TensorFlow and performs the experiment.

- At cumulative 65 minutes, batch callbacks stop training and inference guards
  refuse new batches. An interrupted trial is incomplete and unranked.
- Subsequent trials start only when the 65-minute scheduling deadline has room
  for 1.5 times the largest **observed complete trial runtime**. The first trial
  has no invented per-epoch estimate. Observed costs include model/RF setup,
  compilation, fitting, scoring and writes.
- An independent parent watchdog kills a nonresponsive worker at cumulative
  69 minutes, leaving the last minute for small status/runtime writes. This also
  bounds expensive preprocessing, compilation or scoring that cannot cooperate
  with a Keras callback. No user-space deadline can guarantee timing if the OS
  itself hangs or cannot schedule/terminate processes.
- `runtime.json` is atomically updated about twice per second. Resume adds to
  the prior charged runtime, including fresh imports and preprocessing. Offline
  time between invocations is not charged. A process lock prevents concurrent runs.
- Ctrl+C retains completed trials and attempts a brief clean shutdown. Every
  started trial is terminal: completed, failed or incomplete trials are skipped
  on resume. Interrupted fits are never restarted or partially resumed.
- If the supervisor itself is forcibly killed or the machine crashes, unobserved
  elapsed time cannot be recovered from a monotonic clock. Resume conservatively
  exhausts the remaining budget instead of risking a reset. Do not edit the
  ledger or schedule to obtain additional trials under this experiment.

Run outside system sleep/hibernation: Python 3.9 monotonic clock behavior during
suspend is platform dependent. Cumulative accounting and the watchdog are
implemented but have not been validated by an actual timed training run here.
Checkpoint/status files use separate per-trial directories; only committed
completed trials can be selected. A trial interrupted during scoring or writing
is also incomplete, even if its 30 training epochs finished.

## Outputs and interpretation

All search artifacts are new under `results/tcn_128_search/`:

- `schedule.json`, `metadata.json` (source/code SHA-256, environment, configuration).
- `normalization.json`, `preprocessing.json` (feature order, counts, interval sequence).
- `runtime.json`, `worker_status.json`, `run.lock`.
- `trial_XX/status.json`, `history.json`, and completed `metrics.csv`; failures also
  have `failure.txt`. Histories are persisted after each full epoch.
- Compact `trial_XX/model.h5` checkpoints only when a trial establishes a new
  completed MAE lead. Older leading checkpoints remain for recovery. `best.json`
  identifies the best completed checkpoint and common normalization; load a model
  with `compile=False` and reuse these scripts' preprocessing/postprocessing.
- `trial_metrics.csv` contains all six scored phase/condition rows per completed
  trial. It can be rebuilt from committed per-trial files on resume.

Analysis writes `leaderboard.csv` (completed trials only), `trial_status.csv`
(all scheduled trials), `docs/hr_tcn_128_search_results.md`, and
`docs/images/hr_tcn_128_search.png` when at least one trial completed.
The report shows excluded trials and reasons. No completed trials means no
performance figure or invented leading model.

The sole ranking objective is mean **postprocessed point-anchor validation MAE**
over the three phases. RMSE, bias, high-HR/rise/recovery, peaks, stationary/restart,
coverage, numerical range/steps, averaged anchors and training components remain
separate diagnostics. Historical MTB-2 TCN and gradient-boosting lines use saved
CSV scores, never fabricated prediction curves. Inspect false peaks and dynamics
before considering a leading candidate; no combined secondary score is used.

RF/channel group means are descriptive because the sampled configurations also
vary learning rate and batch size. Phase ranges are not confidence intervals.
Small score differences on one repeatedly tuned ride can be inconclusive; even
the best score is selected validation performance, not independent generalization.
GPU nondeterminism remains possible. Neither promotion nor final retraining occurs.

Preparation checks cover syntax, CLI safety, deterministic scheduling, unique
configurations, parameter arithmetic and structural RF alignment only. Actual
GPU RF checks, training, live deadlines/resume behavior, end-to-end validation
and generated-result plot readability remain unverified until execution.
