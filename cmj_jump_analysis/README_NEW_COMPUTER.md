# CMJ Force Plate and IMU Analysis — New Computer Setup

This guide explains how another team member can install and run the CMJ analysis project on a new computer using VS Code and Python. No raw jump data or trained models are stored in Git; obtain access to the team data folder separately.

## What the project does

The pipeline:

1. reads current-format ForceDecks force-plate and IMU CSV files;
2. preprocesses and quality-checks both signals;
3. detects jumps independently in force and IMU data;
4. synchronizes matching force/IMU events;
5. calculates ground-truth impulse and takeoff velocity;
6. extracts leakage-safe IMU features;
7. compares machine-learning models with session-grouped validation;
8. saves the best model and analytics reports; and
9. predicts normalized impulse/takeoff velocity from a new IMU-only recording.

The current model is a session-validated prototype. It has not yet been validated on verified unseen athletes because reliable athlete IDs are unavailable.

## Requirements

- Git
- Python 3.10 or newer
- VS Code
- VS Code Python extension
- Access to the Azure DevOps repository
- Access to the current Phase 2 force/IMU data folder

## 1. Clone the Azure DevOps repository

In a terminal:

```bash
git clone "https://kivo-motion-team17@dev.azure.com/kivo-motion-team17/KivoMotion-Team17/_git/KivoMotion-Team17"
cd KivoMotion-Team17/cmj_jump_analysis
```

Use an Azure DevOps Personal Access Token or your configured credential manager if authentication is requested.

If the CMJ pull request has not been merged into `main`, fetch and check out the feature branch:

```bash
git fetch origin
git switch feature/cmj-jump-analysis-onedrive
```

## 2. Create the Python environment

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate `.venv` again.

## 3. Open the project in VS Code

From `cmj_jump_analysis`:

```bash
code .
```

In VS Code:

1. press `Cmd+Shift+P` on macOS or `Ctrl+Shift+P` on Windows/Linux;
2. select **Python: Select Interpreter**; and
3. choose the interpreter inside `.venv`.

## 4. Obtain the current data

The training folder must contain matched force and IMU CSV files in one directory. Standard names are:

```text
20260730_CMJ_Test1.csv
20260730_IMU_Test1.csv
```

Paired suffixes are supported when both files use the same suffix:

```text
20260730_CMJ_Test1_A.csv
20260730_IMU_Test1_A.csv
```

Do not mix the older May/December recordings with the current Phase 2 dataset.

### Using OneDrive

1. Add the client-shared folder as a shortcut in OneDrive.
2. Allow the OneDrive desktop application to synchronize it.
3. Select **Always keep on this device** or **Always available offline**.
4. Wait for the green synchronization checkmark before running Python.

Typical local paths are:

```text
macOS:  /Users/<username>/Library/CloudStorage/OneDrive-Personal/Shortcuts/T17 Shared Folder/Phase 2 Data
Windows: C:\Users\<username>\OneDrive\Shortcuts\T17 Shared Folder\Phase 2 Data
```

The web sharing URL is not passed to Python. Always pass the synchronized local folder path.

## 5. Verify the installation

Run the regression tests from `cmj_jump_analysis`:

```bash
python -m pytest -q
```

Expected result: all tests pass.

## 6. Train and evaluate from all paired sessions

Replace `<DATA_DIRECTORY>` with the synchronized local folder:

```bash
python scripts/run_batch.py \
  --data-dir "<DATA_DIRECTORY>" \
  --imu-detector-mode fixed
```

Windows PowerShell equivalent:

```powershell
python scripts/run_batch.py `
  --data-dir "C:\Users\<username>\OneDrive\Shortcuts\T17 Shared Folder\Phase 2 Data" `
  --imu-detector-mode fixed
```

Use the fixed detector for training and deployment consistency. Adaptive mode uses force ground truth to select an IMU threshold and is intended only for diagnostic comparisons.

The current validated data should discover 58 complete pairs. Dataset updates may change the accepted-session and jump totals.

## 7. Find the batch outputs

The batch command writes local results to:

```text
batch_results/
├── batch_session_status.csv
├── all_matched_events.csv
├── all_model_features.csv
├── models/
│   └── best_impulse_model.joblib
├── reports/
│   ├── model_leaderboard.csv
│   ├── out_of_fold_predictions.csv
│   ├── prediction_report.html
│   └── prediction_report.png
└── sessions/
```

Open `batch_results/reports/prediction_report.html` in a browser for the analytics report.

These generated outputs are ignored by Git. Do not force-add them unless the team explicitly changes the repository data policy.

## 8. Predict a new IMU-only recording

Train the batch model first, then run:

```bash
python scripts/predict_imu.py \
  --imu "<PATH_TO_NEW_IMU_CSV>" \
  --expected-jumps 5 \
  --mass-kg 80.0
```

`--mass-kg` is optional. When supplied, the output includes predicted raw impulse in N·s. Without mass, the model still predicts normalized impulse in N·s/kg and takeoff velocity in m/s.

Outputs are written to:

```text
imu_predictions/<input-file-name>/
```

Review `imu_quality.json` and `imu_events.csv`, especially when the detected jump count differs from the expected protocol.

## 9. Run one force/IMU session step by step

```bash
python run_pipeline.py \
  --force "<PATH_TO_CMJ_CSV>" \
  --imu "<PATH_TO_MATCHING_IMU_CSV>"
```

This writes intermediate files to `data/processed/`, models to `models/`, and reports to `reports/`.

The end-to-end notebook is available at:

```text
notebooks/CMJ_End_to_End_Analysis.ipynb
```

Select the `.venv` kernel before running notebook cells.

## Important interpretation notes

- The force plate supplies the supervised-learning ground truth.
- The deployment model uses IMU-derived predictors only.
- Net impulse normalized by mass in N·s/kg is numerically equal to calculated takeoff velocity in m/s under the current formulation.
- Current validation is grouped by recording session, not verified athlete.
- Do not claim unseen-athlete accuracy until a reliable athlete/session manifest is available.
- Do not automatically flip IMU axes. Orientation is recorded as a QC label, while the preferred model uses magnitude-based features.

## Troubleshooting

### `Discovered 0 complete force/IMU pairs`

- Confirm that `--data-dir` points to the directory containing the CSV files, not its parent.
- Check that every session has both `CMJ` and `IMU` files.
- Check the filename pattern and matching suffix.
- Ensure OneDrive files are fully downloaded and available offline.

### `No sessions passed quality control`

First confirm the correct current-format dataset is being used. Then inspect `batch_results/batch_session_status.csv` for the exclusion reason. Do not lower QC thresholds globally just to accept a problematic recording.

### `Model not found`

Run `scripts/run_batch.py` first. The default inference model should be:

```text
batch_results/models/best_impulse_model.joblib
```

### `python` or a package is not found

Activate `.venv`, confirm VS Code selected the same interpreter, and rerun:

```bash
python -m pip install -r requirements.txt
```

### Matplotlib builds a font cache

This is normal on the first run and may take a minute.

### Paths contain spaces

Wrap the complete path in quotation marks, as shown in the examples.

## Contributing safely

Start from the latest `main`:

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/<short-description>
```

Before committing:

```bash
python -m pytest -q
git status
```

Commit source code, tests and documentation only. Do not commit:

- raw athlete or sensor data;
- OneDrive shortcuts or local absolute paths;
- `batch_results/`;
- `imu_predictions/`;
- trained `.joblib` models;
- `.venv`, caches or operating-system files.

Push the branch and create a pull request into `main`:

```bash
git add .
git commit -m "Describe the change"
git push -u origin feature/<short-description>
```

## Key project files

```text
run_pipeline.py                 Readable ten-stage single-session workflow
scripts/run_batch.py            Multi-session training and evaluation
scripts/predict_imu.py          IMU-only inference
src/cmj_analysis/io.py          File parsing and timestamps
src/cmj_analysis/signals.py     Preprocessing and signal QC
src/cmj_analysis/events.py      Force and IMU jump detection
src/cmj_analysis/sync.py        Event synchronization and matching
src/cmj_analysis/features.py    IMU feature extraction
src/cmj_analysis/modeling.py    Grouped validation and model selection
src/cmj_analysis/reporting.py   CSV, PNG and HTML reports
tests/                          Regression tests
```
