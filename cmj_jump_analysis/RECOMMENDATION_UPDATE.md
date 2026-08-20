# Orientation-safe model update

## Implemented

- Preserved the currently validated IMU free-fall detector and scalar-magnitude filter.
- Added per-session IMU orientation and signal-quality fields.
- Explicitly excluded `test_number`, `recording_date`, IDs and orientation labels from predictors.
- Compared `all_axes` and `magnitude_only` feature families under the same grouped CV.
- Saved each feature-family candidate and the overall lowest-RMSE winner.
- Updated reports to identify the selected feature set and summarize orientation groups.
- Added regression tests for administrative leakage and orientation labels.
- Updated the VS Code notebook to use the same comparison logic.
- Added `scripts/predict_imu.py` for target-free prediction from a new current-format IMU file.

## Full Phase 2 validation

- Complete force/IMU pairs: 58
- Accepted sessions: 56
- Excluded sessions: 2 (`20260731_Test7`, fewer than three fixed-detector matches; `20260806_Test3`, no force flight event)
- Matched jumps: 287
- Accepted orientation groups: 44 `Y_positive`, 12 `Y_negative`
- Selected feature set/model: `magnitude_only` / Ridge
- MAE: 0.1700 N·s/kg
- RMSE: 0.2264 N·s/kg
- R²: 0.6755
- Validation: five-fold grouped CV by session
- Unseen-athlete research readiness: false until correct athlete IDs are supplied

## Run

```bash
python scripts/run_batch.py \
  --data-dir "/Users/anselbiju/Downloads/KivoMotion-Team17/Phase 2 Data jumps"
```

The generated `batch_session_status.csv` contains orientation, timestamp-gap,
negative filtered-magnitude and saturation quality fields. These are never used
as model predictors.

After training, predict a new IMU file with:

```bash
python scripts/predict_imu.py --imu "/path/to/new_IMU.csv" --expected-jumps 5
```
