# VTT Apéro

## Seven-Ride Window-Enhanced HR Reconstruction

The principal experimental benchmark combines five outdoor road rides and two MTB rides. Window v2 uses 50 temporal/context features and gradient boosting to predict corrections to linear HR interpolation. Its fixed model is evaluated with two MTB holdouts before one final fit on all seven rides; previous reconstructions remain available below.

MAE and RMSE do not both improve on both held-out MTB rides. MTB 1 MAE/RMSE: 4.147/6.344 BPM (four-ride 4.146/6.324). MTB 2 MAE/RMSE: 4.405/6.018 BPM (four-ride 4.441/6.080). High-HR, peak-amplitude and recovery MAE improve on 1/2, 0/2 and 2/2 rides respectively. The October 16 estimate has supported mean HR 121.60 BPM, peak 172.00 BPM and 99.52% coverage. Garmin Edge/watch observations remain recorded data; reconstructed seconds are estimates. Sparse watch observations, uncertain short HR peaks, sensor errors and limited independent MTB validation remain limitations. This is the new experimental benchmark to beat, not proof of true second-by-second HR.

![Seven-ride window-enhanced heart-rate reconstruction](docs/images/hr_seven_ride_window_benchmark.png)

See the [technical report](docs/hr_seven_ride_window_benchmark.md) and [candidate CSV](data/processed/hr_seven_ride_candidate_2026-10-16.csv).

```powershell
.venv\Scripts\python.exe scripts/evaluate_hr_seven_ride_window.py
```

*Historical sections below retain their original datasets and model status.*

## Initial state: Heart-rate data before reconstruction

During the MTB ride on October 16, 2026, the Garmin Edge 840 lost the
heart-rate broadcast from a Forerunner 265. The Edge activity consequently
contains only a short initial period of heart-rate measurements. The watch's
wellness FIT file provides additional sparse HR observations for the remainder
of the ride. These recorded observations serve as inputs for estimating the
missing heart-rate sequence.

![Recorded watch and Edge heart-rate observations and watch sampling intervals before reconstruction](docs/images/watch_wellness_hr_2026-10-16.png)

*Available watch HR observations before reconstruction, alongside the initial
Edge HR measurements. The lower panel shows intervals between stored watch
observations. The points are recorded data, not a reconstructed continuous
signal.*

## Baseline interpolation

The first baseline compares linear and PCHIP estimates built only from the
recorded watch anchors, using the initial Edge HR measurements for evaluation.
No extrapolation is performed, and gaps longer than 180 seconds remain missing.

Using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/reconstruct_hr_baseline.py
```

See the [baseline report](docs/hr_baseline_interpolation_2026-10-16.md) for
validation results and limitations, and the
[comparison figure](docs/images/hr_baseline_interpolation_2026-10-16.png).
The one-second output contains estimates, not additional measured HR samples.

## Wellness anchor diagnostic

The [October 12 diagnostic](docs/watch_anchor_diagnostic_2026-10-12.md)
compares FR265 wellness anchors with MyWhoosh's Bluetooth recording from the
same optical HR sensor. It tests averaging windows and modest time offsets,
with chronological candidate selection and held-out evaluation. This is a
diagnostic of recording paths, not a new reconstruction.

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_watch_anchors.py
```

![Wellness anchors, MyWhoosh HR and averaging-window diagnostics](docs/images/watch_anchor_diagnostic_2026-10-12.png)

## Predictor quality and stops

Run the four-activity speed, terrain and stationary-period diagnostic:

```powershell
.\.venv\Scripts\python.exe scripts/audit_cycling_predictors.py
```

See the [predictor-quality report](docs/cycling_predictor_quality.md) for
candidate MTB stops, feature coverage and limitations. This audit does not
estimate heart rate.

## Physiological predictor feasibility

Run the first-order HR model comparison with outdoor ride holdouts, sparse
observation simulations, and separate virtual/initial-MTB diagnostics:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_hr_physiology.py
```

See the [feasibility report](docs/hr_physiological_feasibility.md) and
[comparison figure](docs/images/hr_physiological_feasibility.png). This
experiment does not produce a final reconstructed MTB heart-rate series.

## Gradient-boosted residual reconstruction

The final experiment adds a HistGradientBoostingRegressor correction to linear
interpolation. ML-A uses sparse-anchor context and movement; ML-B adds audited
climbing-rate features. Both use fixed complexity, gap handling, residual
smoothing and an anchor taper. Hidden reference HR is excluded from inputs.

In leave-one-outdoor-ride-out evaluation, mean MAE/RMSE across three sampling
phases was 3.77/5.52 BPM and 3.64/5.37 BPM for ML-B, compared with linear's
4.13/6.20 and 3.92/5.99 BPM. ML-A showed no meaningful consistent improvement.
ML-B passed the predefined accuracy, dynamics, stability and target-overlap
checks and was selected. Its MTB overlap MAE/RMSE was 12.57/16.68 BPM, versus
12.98/17.39 BPM for linear interpolation.

See the [evaluation report](docs/hr_ml_residual_evaluation.md),
[comparison figure](docs/images/hr_ml_residual_evaluation.png), and
[phase metrics](data/processed/hr_ml_residual_metrics.csv).

The [final reconstruction report](docs/hr_final_reconstruction_2026-10-16.md),
[complete-ride figure](docs/images/hr_reconstructed_2026-10-16.png), and
[one-second CSV](data/processed/hr_reconstructed_2026-10-16.csv) preserve all
1,525 Edge HR observations and all 158 watch summaries separately. The supported
recorded/estimated sequence has mean 120.73 BPM, peak 172 BPM and 99.52% coverage;
the final 50 seconds remain unsupported. These are estimated sequence statistics,
not recovered physiological measurements or a guaranteed true peak.

Reproduce using the project environment (scikit-learn 1.1.3 was already installed
and is now declared in `requirements.txt`):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/evaluate_hr_ml_residual.py
```

Only two outdoor reference rides are available, with little stationary training
data. Stop/restart errors did not improve consistently. Wellness averaging,
sensor-path differences and terrain accuracy remain
uncertain; sparse summaries are treated as pseudo-point constraints explicitly.
The 1 Hz output cannot establish the true missing short HR fluctuations.
Development was AI-assisted; reported results come from executing the scripts
on the local files. Original FIT files and previous analyses remain unchanged.

## Temporal-window experiment (unapproved candidate)

The [window evaluation](docs/hr_ml_window_evaluation.md) adds four 30-second
regions around each target time, using only observed cycling measurements.
With unchanged model settings and training rows, road MAE/RMSE improves to
3.50/5.14 and 3.34/4.97 BPM. The MTB overlap gain is small, and some peak/high-HR
diagnostics worsen. Promotion to version 2 is recommended subject to approval;
**ML-B version 1 remains current and its final reconstruction is unchanged.**

See the [comparison figure](docs/images/hr_ml_window_evaluation.png),
[phase metrics](data/processed/hr_ml_window_metrics.csv), and separate
[candidate CSV](data/processed/hr_ml_window_candidate_2026-10-16.csv) and
[candidate figure](docs/images/hr_ml_window_candidate_2026-10-16.png).

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_hr_ml_windows.py
```
