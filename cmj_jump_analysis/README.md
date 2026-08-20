# Countermovement jump analysis: force plate + IMU

This project detects countermovement jumps independently from a ForceDecks force plate and a wearable IMU, synchronizes the two recordings, calculates force-plate ground truth, extracts IMU features, predicts jump impulse, evaluates candidate models, and creates an analytics report.

## Important result from the sample

The supplied recording contains **5 matched CMJs**. All five are detected in both sensors, with a median absolute synchronization error of about **18 ms** after event-based clock refinement.

Five jumps from one athlete/session are enough to test the code, but **not enough to validate machine learning**. The generated model scores are marked `DEMONSTRATION ONLY`. For a defensible model, collect at least 30–50 jumps and preferably many more, across multiple athletes and independent sessions. Split evaluation by athlete (preferred) or by session—not randomly by individual jump windows.

## Improved project design

| Stage | Implementation | Why it matters |
|---|---|---|
| 1. Profile | Parse metadata, shape, nulls, timestamps and sampling rates | The exports have different preambles and rates |
| 2. Preprocess | Sort/deduplicate IMU packets; 4th-order, zero-phase 20 Hz low-pass filter | Repairs duplicate/out-of-order packets without shifting events |
| 3. Categorize | Store protocol label in a session manifest | Do not infer CMJ/SJ/DJ from the same features used for outcome prediction |
| 4. Force detection | Detect unloaded flight phase with loaded pre/post checks | Rejects the initial empty plate and final walk-off |
| 5. IMU detection | Detect sustained near-zero acceleration magnitude | Orientation-independent first detector for a body-mounted IMU |
| 6. Match | Robust event pairing plus affine clock mapping | Raw file clocks differ; row-to-row matching would be wrong |
| 7. Biomechanics | Net/positive impulse, takeoff velocity, jump height and flight-time cross-check | Produces force-plate ground truth and quality checks |
| 8. Model | Compare full-axis and magnitude-only feature families across three regressors | Handles inconsistent IMU mounting without silently rotating signals |
| 9. Evaluate | Out-of-fold MAE, RMSE and R²; grouped CV when data permits | Prevents leakage between the same athlete's train/test jumps |
| 10. Report | PNG chart plus self-contained HTML summary | Makes actual vs predicted and residuals reviewable |

The default ML target is **net impulse normalized by body mass (N·s/kg)**. Normalization usually transfers better across athletes than raw N·s. Numerically, net impulse divided by mass equals takeoff velocity when the integration starts from rest, so reports must not describe these as independent outcomes. Raw net impulse and jump height remain available as alternative targets.

## IMU orientation policy

The current Phase 2 data contains two mounting orientations: 46 recordings are `Y_positive` and 12 are `Y_negative`. All 11 recordings from 30 July plus `20260804_Test6` use the negative orientation. The pipeline records this as a quality label but does **not** flip or rotate axes automatically. A simple gravity alignment reduced validation accuracy and cannot resolve rotation about gravity.

Training evaluates two documented feature families under identical grouped cross-validation:

- `all_axes`: all leakage-safe IMU statistics, including individual X/Y/Z axes;
- `magnitude_only`: IMU flight time, acceleration/gyroscope magnitude statistics and magnitude areas.

The lowest out-of-fold RMSE determines the saved `best_impulse_model.joblib`. `imu_orientation_group`, `test_number` and `recording_date` are explicitly excluded from predictors. The detector retains the previously validated scalar-magnitude filter for now; its occasional negative ringing is logged in quality output and must be retuned against force events before the formula is changed.

## Open in VS Code

1. Open this folder in VS Code.
2. Open the integrated terminal (`Terminal` → `New Terminal`).
3. Create and select a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code, press `Cmd+Shift+P`, choose **Python: Select Interpreter**, and select `.venv/bin/python`.

## Run the complete sample

From this project folder:

```bash
python run_pipeline.py \
  --force "/Users/anselbiju/Downloads/KivoMotion-Team17/Phase 2 Data jumps/20260730_CMJ_Test1.csv" \
  --imu "/Users/anselbiju/Downloads/KivoMotion-Team17/Phase 2 Data jumps/20260730_IMU_Test1.csv"
```

Stage 1 can also be run independently:

```bash
python scripts/01_profile_data.py --force "/path/to/force.csv" --imu "/path/to/imu.csv"
```

## Code map

The ten functions in `run_pipeline.py` are the readable, step-by-step workflow. Reusable algorithms live in:

- `src/cmj_analysis/io.py`: ForceDecks/IMU readers and sampling rates
- `src/cmj_analysis/signals.py`: filtering and contiguous event regions
- `src/cmj_analysis/events.py`: force and IMU jump detection, impulse and velocity
- `src/cmj_analysis/sync.py`: event-based clock synchronization and matching
- `src/cmj_analysis/features.py`: per-jump IMU features
- `src/cmj_analysis/modeling.py`: leakage-aware cross-validation and model comparison
- `src/cmj_analysis/reporting.py`: final analytics chart and HTML report

Generated files are written to `data/processed`, `models`, and `reports`.

## Training with the full study

Run the detection/feature pipeline for every synchronized force+IMU recording, preserving a unique `session_id` and the force export's `athlete_id`. Then concatenate/train from all generated feature tables:

```bash
python scripts/08_train_many_sessions.py \
  session_01/model_features.csv \
  session_02/model_features.csv \
  session_03/model_features.csv
```

The trainer automatically chooses grouped cross-validation when at least three athletes or sessions are present, compares both feature families and saves the best combination. A strong study design would include:

- multiple athletes, days and IMU reattachments;
- consistent IMU body location and axis/quaternion convention;
- CMJ/SJ/DJ protocol labels entered during collection;
- force and IMU clock timestamps plus a synchronization action;
- manual review labels for false detections and unusual trials;
- a final untouched athlete-level test set after model selection.

## Definitions used

- **Net impulse:** integral of `(vertical force − bodyweight)` from movement onset to takeoff.
- **Positive impulse:** integral of only the positive portion of `(force − bodyweight)` over the same interval.
- **Takeoff velocity:** `net impulse / body mass`.
- **Flight-time velocity cross-check:** `g × flight time / 2`.
- **Jump height:** `takeoff velocity² / (2g)`.

Movement onset is found from the last stable bodyweight window before takeoff. Force flight uses a threshold of `max(20 N, 5% bodyweight)`. Current-format IMU detection defaults to acceleration magnitude below `0.65 g` for at least `0.15 s`. Batch training uses this same fixed detector so training and deployment features are consistent, then accepts only event pairs whose refined force/IMU timing error is at most 100 ms. `--imu-detector-mode adaptive` retains the force-assisted `0.35–0.80 g` grid for diagnostic comparison only. Thresholds must be locked and revalidated when hardware placement or protocol changes.

## Process a directory of recordings

The batch runner pairs files by date and test number, processes each session separately, keeps a quality-control log, creates one combined ML table, performs session-grouped validation, and generates an updated report:

```bash
python scripts/run_batch.py \
  --data-dir "/Users/anselbiju/Downloads/Phase 2 Data jumps"
```

Outputs are stored in `batch_results/`:

- `batch_session_status.csv`: accepted/excluded sessions and thresholds
- `all_matched_events.csv`: combined force/IMU event table
- `all_model_features.csv`: one ML-ready row per accepted jump
- `sessions/`: per-session detections, matches, features and quality metadata
- `models/best_impulse_model.joblib`: fitted best pipeline
- `models/best_all_axes_model.joblib`: best full-axis candidate
- `models/best_magnitude_only_model.joblib`: best magnitude-only candidate
- `reports/`: grouped-CV metrics, predictions, HTML report and chart

With the inference-compatible fixed detector, 56 of 58 sessions passed automated QC, producing 287 matched jumps. `20260731_Test7` was excluded because only two fixed-detector events matched well; `20260806_Test3` had no force-plate flight event. Magnitude-only Ridge regression led the session-grouped comparison with MAE 0.1700 N·s/kg, RMSE 0.2264 N·s/kg and R² 0.6755.

All exports currently contain the same ForceDecks `AthleteId`, despite recorded masses ranging from 48.47 to 111.13 kg. This indicates that the metadata ID is not a reliable participant identifier. The current validation is therefore grouped by recording session and must not be interpreted as performance on unseen athletes. Add a correct participant-to-session manifest before athlete-level validation.

Each session quality JSON and the combined status CSV now record orientation group, orientation review state, negative filtered-magnitude samples, large timestamp gaps and accelerometer saturation. These fields are quality controls, not model inputs.

## Predict a new IMU-only recording

Run the batch trainer first so `batch_results/models/best_impulse_model.joblib` exists. Then a current-format IMU recording can be processed without a force file:

```bash
python scripts/predict_imu.py \
  --imu "/path/to/new_IMU_Test.csv" \
  --expected-jumps 5 \
  --mass-kg 83.08
```

The command preprocesses the IMU, records orientation/signal quality, detects jumps, extracts target-free features, loads the saved feature order, predicts normalized impulse/takeoff velocity and creates CSV, PNG and HTML outputs under `imu_predictions/<input filename>/`. Body mass is optional; when supplied it also calculates predicted raw net impulse in N·s.

IMU-only inference defaults to `0.65 g` for at least `0.15 s`. Against the 298 accepted Phase 2 matched events, this fixed setting achieved 96.4% event precision, 97.7% recall and 97.0% F1 with median takeoff difference of 10 ms. It is configurable because a fixed detector cannot use force-assisted threshold selection. Always review warnings and the detected-event table, especially when the detected count differs from the known protocol.

Normalized impulse in N·s/kg and takeoff velocity in m/s have the same numeric value under the training formulation. Raw impulse requires body mass. The current model remains a session-grouped prototype until verified athlete IDs are available.

## Test

```bash
pytest -q
```
