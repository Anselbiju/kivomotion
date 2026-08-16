#!/usr/bin/env python3
"""Train Stage 8/9 from feature CSVs generated for multiple sessions."""
from pathlib import Path
import argparse, os, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd
from cmj_analysis.modeling import train_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument("features", type=Path, nargs="+", help="model_features.csv files")
parser.add_argument("--output-dir", type=Path, default=ROOT)
args = parser.parse_args()

table = pd.concat([pd.read_csv(path) for path in args.features], ignore_index=True)
result = train_and_evaluate(table, model_path=args.output_dir / "models" / "best_impulse_model.joblib")
(args.output_dir / "reports").mkdir(parents=True, exist_ok=True)
result.leaderboard.to_csv(args.output_dir / "reports" / "model_leaderboard.csv", index=False)
result.predictions.to_csv(args.output_dir / "reports" / "out_of_fold_predictions.csv", index=False)
print(result.leaderboard.to_string(index=False))
print(f"Validation: {result.validation_method}")
print(f"Research-ready: {result.research_ready}")

