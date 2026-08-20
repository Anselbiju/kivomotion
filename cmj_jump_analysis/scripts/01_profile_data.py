#!/usr/bin/env python3
"""Standalone Stage 1: inspect CSV structure, time range and missing values."""
from pathlib import Path
import argparse, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmj_analysis.io import estimate_sampling_rate, read_force_csv, read_imu_csv

parser = argparse.ArgumentParser()
parser.add_argument("--force", type=Path, required=True)
parser.add_argument("--imu", type=Path, required=True)
args = parser.parse_args()

for label, recording, time_col in [
    ("Force", read_force_csv(args.force), "Time"),
    ("IMU", read_imu_csv(args.imu), "time_s"),
]:
    print(f"\n{label}: {recording.source}")
    print(f"rows={len(recording.data):,}, columns={len(recording.data.columns)}, fs={estimate_sampling_rate(recording.data[time_col]):.2f} Hz")
    print("nulls:\n", recording.data.isna().sum().to_string())
    print("numeric summary:\n", recording.data.describe().round(3).to_string())

