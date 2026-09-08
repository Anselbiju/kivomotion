#!/usr/bin/env python3
"""Process every paired CMJ/IMU CSV in a directory and train one combined model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
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
from cmj_analysis.modeling import train_and_evaluate
from cmj_analysis.reporting import create_report
from cmj_analysis.signals import preprocess_force, preprocess_imu
from cmj_analysis.sync import match_events


NAME_PATTERN = re.compile(
    r"(?P<date>\d{8})_(?P<sensor>CMJ|IMU)_Test(?P<test>\d+)"
    r"(?:_(?P<suffix>[A-Za-z0-9]+))?\.csv$",
    re.I,
)
IMU_CONFIGS = [
    (0.35, 0.18),
    (0.50, 0.15),
    (0.65, 0.15),
    (0.80, 0.12),
]


def discover_pairs(data_dir: Path) -> tuple[list[dict], list[dict]]:
    sessions: dict[tuple[str, int, str], dict] = {}
    ignored = []
    for path in sorted(data_dir.glob("*.csv")):
        match = NAME_PATTERN.match(path.name)
        if not match:
            ignored.append({
                "file": str(path),
                "status": "ignored",
                "reason": "filename does not match expected pattern",
            })
            continue
        suffix = (match["suffix"] or "").upper()
        key = (match["date"], int(match["test"]), suffix)
        sessions.setdefault(key, {})[match["sensor"].upper()] = path
    pairs = []
    for (date, test, suffix), files in sorted(sessions.items()):
        session_id = f"{date}_Test{test}" + (f"_{suffix}" if suffix else "")
        if {"CMJ", "IMU"}.issubset(files):
            pairs.append({
                "session_id": session_id,
                "date": date,
                "test": test,
                "suffix": suffix,
                **files,
            })
        else:
            ignored.append({
                "session_id": session_id,
                "status": "ignored",
                "reason": "missing force or IMU file",
            })
    return pairs, ignored


def best_imu_alignment(imu, force_events, expected_offset):
    candidates = []
    for threshold, duration in IMU_CONFIGS:
        imu_events = detect_imu_jumps(
            imu,
            freefall_threshold_g=threshold,
            min_freefall_s=duration,
        )
        if imu_events.empty:
            continue
        try:
            matched, sync = match_events(
                force_events,
                imu_events,
                expected_offset_s=expected_offset,
            )
        except ValueError:
            continue
        good = int((matched["sync_error_s"].abs() <= 0.10).sum())
        score = (-good, sync["median_abs_sync_error_s"], abs(len(force_events) - len(imu_events)))
        candidates.append((score, threshold, duration, imu_events, matched, sync))
    if not candidates:
        raise ValueError("No IMU threshold configuration produced alignable events")
    return min(candidates, key=lambda item: item[0])


def process_session(pair: dict, session_dir: Path, global_start_id: int):
    force_rec = read_force_csv(pair["CMJ"])
    imu_rec = read_imu_csv(pair["IMU"])
    athlete_id = matching_athlete_id(force_rec, imu_rec)
    force_fs = estimate_sampling_rate(force_rec.data["Time"])
    imu_fs = estimate_sampling_rate(imu_rec.data["time_s"])
    force = preprocess_force(force_rec.data, force_fs)
    imu = preprocess_imu(imu_rec.data, imu_fs)
    mass = float(force_rec.metadata["Weight"]) if "Weight" in force_rec.metadata else None
    force_events = detect_force_jumps(force, force_fs, mass_kg=mass)
    if force_events.empty:
        raise ValueError("No force-plate events detected")

    expected_offset = nominal_clock_offset_s(force_rec, imu_rec)
    _, threshold, duration, imu_events, matched, sync = best_imu_alignment(
        imu, force_events, expected_offset
    )
    matched = matched[matched["sync_error_s"].abs() <= 0.10].copy().reset_index(drop=True)
    if len(matched) < 3:
        raise ValueError(f"Only {len(matched)} well-synchronized event pairs; minimum is 3")

    matched["source_jump_id"] = matched["jump_id"]
    matched["jump_id"] = range(global_start_id, global_start_id + len(matched))
    features = extract_imu_features(imu, matched)
    for table in (matched, features):
        table["session_id"] = pair["session_id"]
        table["athlete_id"] = athlete_id
        table["recording_date"] = pair["date"]
        table["test_number"] = pair["test"]

    session_dir.mkdir(parents=True, exist_ok=True)
    force_events.to_csv(session_dir / "force_events.csv", index=False)
    imu_events.to_csv(session_dir / "imu_events.csv", index=False)
    matched.to_csv(session_dir / "matched_events.csv", index=False)
    features.to_csv(session_dir / "model_features.csv", index=False)
    details = {
        **sync,
        "session_id": pair["session_id"],
        "athlete_id": athlete_id,
        "force_file": str(pair["CMJ"]),
        "imu_file": str(pair["IMU"]),
        "imu_freefall_threshold_g": threshold,
        "imu_min_freefall_s": duration,
        "accepted_events": len(matched),
    }
    (session_dir / "quality.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return matched, features, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "batch_results")
    args = parser.parse_args()
    sessions_dir = args.output_dir / "sessions"
    reports_dir = args.output_dir / "reports"
    models_dir = args.output_dir / "models"
    for folder in (sessions_dir, reports_dir, models_dir):
        folder.mkdir(parents=True, exist_ok=True)

    pairs, ignored = discover_pairs(args.data_dir)
    print(f"Discovered {len(pairs)} complete force/IMU pairs")
    status_rows = list(ignored)
    all_matched, all_features = [], []
    next_jump_id = 1
    for number, pair in enumerate(pairs, start=1):
        try:
            matched, features, details = process_session(
                pair, sessions_dir / pair["session_id"], next_jump_id
            )
            next_jump_id += len(matched)
            all_matched.append(matched)
            all_features.append(features)
            status_rows.append({
                "session_id": pair["session_id"], "status": "accepted",
                "force_events": details["force_events_detected"],
                "imu_events": details["imu_events_detected"],
                "accepted_events": details["accepted_events"],
                "median_abs_sync_error_s": details["median_abs_sync_error_s"],
                "imu_threshold_g": details["imu_freefall_threshold_g"],
                "imu_min_duration_s": details["imu_min_freefall_s"],
            })
            print(f"[{number:02d}/{len(pairs)}] {pair['session_id']}: accepted {len(matched)} jumps")
        except Exception as exc:
            status_rows.append({
                "session_id": pair["session_id"], "status": "excluded",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{number:02d}/{len(pairs)}] {pair['session_id']}: EXCLUDED — {exc}")

    status = pd.DataFrame(status_rows)
    status.to_csv(args.output_dir / "batch_session_status.csv", index=False)
    if not all_features:
        raise RuntimeError("No sessions passed quality control")
    matched_table = pd.concat(all_matched, ignore_index=True)
    feature_table = pd.concat(all_features, ignore_index=True)
    matched_table.to_csv(args.output_dir / "all_matched_events.csv", index=False)
    feature_table.to_csv(args.output_dir / "all_model_features.csv", index=False)

    result = train_and_evaluate(
        feature_table,
        model_path=models_dir / "best_impulse_model.joblib",
    )
    result.leaderboard.to_csv(reports_dir / "model_leaderboard.csv", index=False)
    result.predictions.to_csv(reports_dir / "out_of_fold_predictions.csv", index=False)
    create_report(
        matched_table, result.leaderboard, result.predictions,
        result.validation_method, result.research_ready, reports_dir,
    )
    print(f"\nAccepted {status['status'].eq('accepted').sum()} sessions and {len(feature_table)} jumps")
    print(result.leaderboard.to_string(index=False))
    print(f"Validation: {result.validation_method}")
    print(f"Research-ready for unseen athletes: {result.research_ready}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
