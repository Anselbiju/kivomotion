#!/usr/bin/env python3
"""Run the ten-stage CMJ force-plate + IMU workflow.

Example (from the project folder):
    python run_pipeline.py --force /path/to/CMJ.csv --imu /path/to/IMU.csv
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd

from cmj_analysis.events import detect_force_jumps, detect_imu_jumps
from cmj_analysis.features import extract_imu_features
from cmj_analysis.io import (
    estimate_sampling_rate,
    matching_athlete_id,
    nominal_clock_offset_s,
    read_force_csv,
    read_imu_csv,
)
from cmj_analysis.modeling import compare_feature_sets
from cmj_analysis.reporting import create_report
from cmj_analysis.signals import assess_imu_quality, preprocess_force, preprocess_imu
from cmj_analysis.sync import match_events


def stage_1_profile(force_path: Path, imu_path: Path):
    force_rec = read_force_csv(force_path)
    imu_rec = read_imu_csv(imu_path)
    force_fs = estimate_sampling_rate(force_rec.data["Time"])
    imu_fs = estimate_sampling_rate(imu_rec.data["time_s"])
    print("\n1. DATA PROFILE")
    print(f"Force: {len(force_rec.data):,} rows, {force_fs:.2f} Hz, {force_rec.data['Time'].iloc[-1]:.3f} s")
    print(f"IMU:   {len(imu_rec.data):,} cleaned rows, {imu_fs:.2f} Hz, {imu_rec.data['time_s'].iloc[-1]:.3f} s")
    print(f"Force null cells: {int(force_rec.data.isna().sum().sum())}; IMU null cells: {int(imu_rec.data.isna().sum().sum())}")
    return force_rec, imu_rec, force_fs, imu_fs


def stage_2_preprocess(force_rec, imu_rec, force_fs: float, imu_fs: float, processed: Path):
    force = preprocess_force(force_rec.data, force_fs)
    imu = preprocess_imu(imu_rec.data, imu_fs)
    force.to_csv(processed / "force_preprocessed.csv", index=False)
    imu.to_csv(processed / "imu_preprocessed.csv", index=False)
    print("2. PREPROCESS: timestamps cleaned; force and IMU low-pass filtered")
    return force, imu


def stage_3_categorize(force_rec, imu_rec, processed: Path):
    # In this sample the protocol/file name supplies the category. For a mixed
    # study, maintain one manifest row per recording with CMJ/SJ/DJ labels.
    jump_type = "CMJ" if "CMJ" in force_rec.source.name.upper() else "UNLABELLED"
    athlete_id = matching_athlete_id(force_rec, imu_rec)
    manifest = {
        "session_id": force_rec.source.stem,
        "athlete_id": athlete_id,
        "jump_type": jump_type,
        "force_source": str(force_rec.source),
        "imu_source": str(imu_rec.source),
        "force_metadata": force_rec.metadata,
        "imu_metadata": imu_rec.metadata,
    }
    (processed / "session_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"3. CATEGORIZE: protocol label = {jump_type}")
    return manifest


def stage_4_force_detection(force, force_rec, force_fs: float, processed: Path):
    mass = float(force_rec.metadata["Weight"]) if "Weight" in force_rec.metadata else None
    events = detect_force_jumps(force, force_fs, mass_kg=mass)
    if events.empty:
        raise RuntimeError("No force-plate jumps were detected; review thresholds and signal units")
    events.to_csv(processed / "force_events.csv", index=False)
    print(f"4. FORCE DETECTION: {len(events)} jumps")
    return events


def stage_5_imu_detection(imu, processed: Path):
    events = detect_imu_jumps(imu)
    if events.empty:
        raise RuntimeError("No IMU jumps were detected; review placement, units and free-fall threshold")
    events.to_csv(processed / "imu_events.csv", index=False)
    print(f"5. IMU DETECTION: {len(events)} jumps")
    return events


def stage_6_match(force_events, imu_events, processed: Path, expected_offset_s: float | None = None):
    matched, sync = match_events(force_events, imu_events, expected_offset_s=expected_offset_s)
    matched.to_csv(processed / "matched_events.csv", index=False)
    (processed / "sync_parameters.json").write_text(json.dumps(sync, indent=2), encoding="utf-8")
    print(f"6. MATCH: {len(matched)} pairs; median absolute timing error = {sync['median_abs_sync_error_s'] * 1000:.1f} ms")
    return matched, sync


def stage_7_velocity_and_features(imu, matched, manifest: dict, processed: Path):
    imu_fs = estimate_sampling_rate(imu["time_s"])
    imu_quality = assess_imu_quality(imu, imu_fs)
    features = extract_imu_features(imu, matched)
    features["session_id"] = manifest["session_id"]
    features["athlete_id"] = manifest["athlete_id"]
    features["imu_orientation_group"] = imu_quality["imu_orientation_group"]
    matched["session_id"] = manifest["session_id"]
    matched["imu_orientation_group"] = imu_quality["imu_orientation_group"]
    matched.to_csv(processed / "matched_events.csv", index=False)
    (processed / "imu_quality.json").write_text(
        json.dumps(imu_quality, indent=2), encoding="utf-8"
    )
    features.to_csv(processed / "model_features.csv", index=False)
    comparison = matched[["jump_id", "takeoff_velocity_m_s", "flight_velocity_m_s", "jump_height_impulse_m", "jump_height_flight_m"]]
    comparison.to_csv(processed / "takeoff_velocity_analysis.csv", index=False)
    print("7. TAKEOFF VELOCITY + FEATURES: impulse and flight-time estimates calculated")
    return features


def stage_8_model(features, models_dir: Path):
    result = compare_feature_sets(
        features,
        model_path=models_dir / "best_impulse_model.joblib",
        candidate_models_dir=models_dir,
    )
    print(
        f"8. MODEL: compared {len(result.leaderboard)} model/feature-set pipelines; "
        f"selected {result.best_feature_set}"
    )
    return result


def stage_9_evaluate(result, reports: Path):
    result.leaderboard.to_csv(reports / "model_leaderboard.csv", index=False)
    result.predictions.to_csv(reports / "out_of_fold_predictions.csv", index=False)
    best = result.leaderboard.iloc[0]
    print(f"9. EVALUATE: best={best['model']}; MAE={best['mae']:.4f}, RMSE={best['rmse']:.4f}, R²={best['r2']:.4f}")
    print(f"   Validation: {result.validation_method}")
    print(f"   Research-ready: {result.research_ready}")


def stage_10_report(matched, result, reports: Path):
    html_path, png_path = create_report(
        matched, result.leaderboard, result.predictions,
        result.validation_method, result.research_ready, reports,
    )
    print(f"10. REPORT: {html_path}")
    print(f"            {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", type=Path, required=True, help="ForceDecks raw CSV")
    parser.add_argument("--imu", type=Path, required=True, help="IMU raw CSV")
    parser.add_argument("--output-dir", type=Path, default=PROJECT, help="Project/result folder")
    args = parser.parse_args()

    processed = args.output_dir / "data" / "processed"
    reports = args.output_dir / "reports"
    models_dir = args.output_dir / "models"
    for folder in (processed, reports, models_dir, PROJECT / ".matplotlib"):
        folder.mkdir(parents=True, exist_ok=True)

    force_rec, imu_rec, force_fs, imu_fs = stage_1_profile(args.force, args.imu)
    force, imu = stage_2_preprocess(force_rec, imu_rec, force_fs, imu_fs, processed)
    manifest = stage_3_categorize(force_rec, imu_rec, processed)
    force_events = stage_4_force_detection(force, force_rec, force_fs, processed)
    imu_events = stage_5_imu_detection(imu, processed)
    expected_offset = nominal_clock_offset_s(force_rec, imu_rec)
    matched, _ = stage_6_match(force_events, imu_events, processed, expected_offset)
    features = stage_7_velocity_and_features(imu, matched, manifest, processed)
    result = stage_8_model(features, models_dir)
    stage_9_evaluate(result, reports)
    stage_10_report(matched, result, reports)


if __name__ == "__main__":
    main()
