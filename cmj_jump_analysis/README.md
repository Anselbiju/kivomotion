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
| 8. Model | Ridge, random forest and histogram gradient boosting | Compares a stable linear baseline with nonlinear models |
| 9. Evaluate | Out-of-fold MAE, RMSE and R²; grouped CV when data permits | Prevents leakage between the same athlete's train/test jumps |
| 10. Report | PNG chart plus self-contained HTML summary | Makes actual vs predicted and residuals reviewable |

The default ML target is **net impulse normalized by body mass (N·s/kg)**. Normalization usually transfers better across athletes than raw N·s. Raw net impulse and takeoff velocity remain available as alternative targets.

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

The trainer automatically chooses grouped cross-validation when at least three athletes or sessions are present. A strong study design would include:

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

Movement onset is found from the last stable bodyweight window before takeoff. Force flight uses a threshold of `max(20 N, 5% bodyweight)`. Single-file IMU detection defaults to acceleration magnitude below `0.50 g` for at least `0.15 s`. Batch processing evaluates a documented threshold grid from `0.35–0.80 g`, then accepts only event pairs whose refined force/IMU timing error is at most 100 ms. Thresholds must be locked and revalidated before deployment when hardware placement or protocol changes.

## Process a directory of recordings

The batch runner pairs files by date, test number, and optional repeat-athlete suffix; validates the force/IMU athlete IDs; processes each session separately; keeps a quality-control log; creates one combined ML table; performs athlete-grouped validation; and generates an updated report:

```bash
python scripts/run_batch.py \
  --data-dir "../Phase 2 Data jumps"
```

Outputs are stored in `batch_results/`:

- `batch_session_status.csv`: accepted/excluded sessions and thresholds
- `all_matched_events.csv`: combined force/IMU event table
- `all_model_features.csv`: one ML-ready row per accepted jump
- `sessions/`: per-session detections, matches, features and quality metadata
- `models/best_impulse_model.joblib`: fitted best pipeline
- `reports/`: grouped-CV metrics, predictions, HTML report and chart

For the current directory, all 58 sessions passed automated QC, producing 308 matched jumps. The force detector bridges sub-50 ms threshold gaps caused by landing-transition filter ringing; this recovered valid events in three affected sessions, including `20260806_Test3`. Ridge regression led the athlete-grouped comparison with MAE 0.1778 N·s/kg, RMSE 0.2427 N·s/kg and R² 0.6477.

The updated dataset contains 51 valid anonymous athlete UUIDs across 58 sessions. Every force/IMU pair has the same athlete ID in both metadata blocks, and repeat sessions for the four identified longitudinal athletes reuse the appropriate UUID. The batch runner validates this relationship before processing and uses five-fold grouped cross-validation by athlete, so no athlete appears in both training and validation within a fold.

## Test

```bash
pytest -q
```
