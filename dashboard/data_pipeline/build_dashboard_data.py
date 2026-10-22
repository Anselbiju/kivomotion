#!/usr/bin/env python3
"""Build the KivoMotion dashboard snapshot and its local SQLite database."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_FILES = [
    "cmj_jump_analysis/batch_results/all_matched_events.csv",
    "cmj_jump_analysis/batch_results/all_model_features.csv",
    "cmj_jump_analysis/batch_results/batch_session_status.csv",
    "cmj_jump_analysis/batch_results/reports/model_leaderboard.csv",
    "cmj_jump_analysis/batch_results/reports/out_of_fold_predictions.csv",
    "Phase 2 Data jumps/athlete_id_map.csv",
]
MODEL_VALIDATION_FILE = "cmj_jump_analysis/batch_results/model_validation.json"

ACCELERATION_FIELDS = {
    "ax_g_f_mean": ("xAccelerationMeanG", "X-axis mean"),
    "ax_g_f_std": ("xAccelerationStdG", "X-axis variability"),
    "ax_g_f_min": ("xAccelerationMinG", "X-axis minimum"),
    "ax_g_f_max": ("xAccelerationMaxG", "X-axis maximum"),
    "ax_g_f_range": ("xAccelerationRangeG", "X-axis range"),
    "ax_g_f_rms": ("xAccelerationRmsG", "X-axis RMS"),
    "ay_g_f_mean": ("yAccelerationMeanG", "Y-axis mean"),
    "ay_g_f_std": ("yAccelerationStdG", "Y-axis variability"),
    "ay_g_f_min": ("yAccelerationMinG", "Y-axis minimum"),
    "ay_g_f_max": ("yAccelerationMaxG", "Y-axis maximum"),
    "ay_g_f_range": ("yAccelerationRangeG", "Y-axis range"),
    "ay_g_f_rms": ("yAccelerationRmsG", "Y-axis RMS"),
    "az_g_f_mean": ("zAccelerationMeanG", "Z-axis mean"),
    "az_g_f_std": ("zAccelerationStdG", "Z-axis variability"),
    "az_g_f_min": ("zAccelerationMinG", "Z-axis minimum"),
    "az_g_f_max": ("zAccelerationMaxG", "Z-axis maximum"),
    "az_g_f_range": ("zAccelerationRangeG", "Z-axis range"),
    "az_g_f_rms": ("zAccelerationRmsG", "Z-axis RMS"),
    "acc_mag_g_f_mean": ("accelerationMagnitudeMeanG", "Magnitude mean"),
    "acc_mag_g_f_std": ("accelerationMagnitudeStdG", "Magnitude variability"),
    "acc_mag_g_f_min": ("accelerationMagnitudeMinG", "Magnitude minimum"),
    "acc_mag_g_f_max": ("accelerationMagnitudeMaxG", "Magnitude peak"),
    "acc_mag_g_f_range": ("accelerationMagnitudeRangeG", "Magnitude range"),
    "acc_mag_g_f_rms": ("accelerationMagnitudeRmsG", "Magnitude RMS"),
    "acc_mag_area_g_s": ("accelerationAreaGs", "Acceleration area"),
}


def records(frame: pd.DataFrame) -> list[dict]:
    """Return JSON-safe records with finite precision and nulls for NaN."""
    return json.loads(frame.to_json(orient="records", double_precision=10, date_format="iso"))


def short_label(athlete_id: str) -> str:
    return f"Athlete {str(athlete_id).split('-')[0].upper()}"


def prepare_athlete_aliases(athlete_map: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Validate and normalize the optional persistent athlete display names."""
    prepared = athlete_map.copy()
    if "renamed_athlete" not in prepared:
        prepared["renamed_athlete"] = ""
    prepared["renamed_athlete"] = prepared["renamed_athlete"].fillna("").astype(str).str.strip()
    if prepared["renamed_athlete"].str.len().gt(48).any():
        raise ValueError("renamed_athlete values must contain 48 characters or fewer")
    named = prepared[prepared["renamed_athlete"].ne("")]
    conflicting = named.groupby("athlete_id")["renamed_athlete"].nunique()
    if (conflicting > 1).any():
        raise ValueError("athlete_id_map.csv assigns more than one renamed_athlete to an athlete_id")
    aliases = named.drop_duplicates("athlete_id").set_index("athlete_id")["renamed_athlete"].to_dict()
    return prepared, aliases


def percent_change(first: float, last: float) -> float | None:
    if pd.isna(first) or pd.isna(last) or first == 0:
        return None
    return (last / first - 1) * 100


def athlete_map_session_id(row: pd.Series) -> str:
    """Recreate the batch session key from one athlete-map row."""
    date_value = str(row["date"]).strip().removesuffix(".0")
    test_value = str(row["test"]).strip()
    suffix_value = "" if pd.isna(row.get("suffix")) else str(row.get("suffix", "")).strip()
    return f"{date_value}_{test_value}{f'_{suffix_value}' if suffix_value else ''}"


def downsample_trace(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    start_s: float,
    end_s: float,
    *,
    origin_s: float,
    maximum_points: int = 220,
) -> list[list[float]]:
    """Return a compact min/max envelope while retaining short signal peaks."""
    times = np.asarray(time_values, dtype=float)
    values = np.asarray(signal_values, dtype=float)
    valid = np.isfinite(times) & np.isfinite(values) & (times >= start_s) & (times <= end_s)
    source_indices = np.flatnonzero(valid)
    if not len(source_indices):
        return []
    if len(source_indices) > maximum_points:
        bucket_count = max(1, maximum_points // 2)
        buckets = np.array_split(source_indices, bucket_count)
        retained: set[int] = {int(source_indices[0]), int(source_indices[-1])}
        for bucket in buckets:
            if not len(bucket):
                continue
            bucket_values = values[bucket]
            retained.add(int(bucket[int(np.argmin(bucket_values))]))
            retained.add(int(bucket[int(np.argmax(bucket_values))]))
        source_indices = np.asarray(sorted(retained), dtype=int)
    return [
        [round(float(times[index] - origin_s), 4), round(float(values[index]), 4)]
        for index in source_indices
    ]


def build_signal_traces(
    project_root: Path,
    combined: pd.DataFrame,
    athlete_map: pd.DataFrame,
    status: pd.DataFrame,
    raw_data_dir: Path,
) -> pd.DataFrame:
    """Build review-sized force and IMU traces from the synchronized raw CSVs."""
    analysis_source = project_root / "cmj_jump_analysis" / "src"
    if str(analysis_source) not in sys.path:
        sys.path.insert(0, str(analysis_source))
    from cmj_analysis.io import estimate_sampling_rate, read_force_csv, read_imu_csv
    from cmj_analysis.signals import preprocess_force, preprocess_imu

    map_by_session = {
        athlete_map_session_id(row): row
        for _, row in athlete_map.iterrows()
    }
    accepted_status = status[status["status"].eq("accepted")].dropna(subset=["session_id"])
    threshold_by_session = accepted_status.drop_duplicates("session_id").set_index("session_id")["imu_threshold_g"].to_dict()
    rows: list[dict] = []
    for session_id, session_events in combined.groupby("session_id", sort=True):
        mapping = map_by_session.get(str(session_id))
        if mapping is None:
            continue
        force_path = raw_data_dir / str(mapping["cmj_file"])
        imu_path = raw_data_dir / str(mapping["imu_file"])
        if not force_path.exists() or not imu_path.exists():
            continue
        force_recording = read_force_csv(force_path)
        imu_recording = read_imu_csv(imu_path)
        force = preprocess_force(
            force_recording.data,
            estimate_sampling_rate(force_recording.data["Time"]),
        )
        imu = preprocess_imu(
            imu_recording.data,
            estimate_sampling_rate(imu_recording.data["time_s"]),
        )
        for _, event in session_events.iterrows():
            force_takeoff = float(event["takeoff_s"])
            force_landing = float(event["landing_s"])
            movement_onset = float(event["movement_onset_s"])
            imu_takeoff = float(event["imu_takeoff_s"])
            imu_landing = float(event["imu_landing_s"])
            imu_freefall_end = float(event["imu_freefall_end_s"])
            force_points = downsample_trace(
                force["Time"].to_numpy(),
                force["force_n"].to_numpy(),
                movement_onset - 0.2,
                force_landing + 0.3,
                origin_s=force_takeoff,
            )
            imu_points = downsample_trace(
                imu["time_s"].to_numpy(),
                imu["acc_mag_g_f"].to_numpy(),
                imu_takeoff - 1.0,
                imu_landing + 0.3,
                origin_s=imu_takeoff,
            )
            rows.append({
                "jumpId": int(event["jump_id"]),
                "sessionId": str(session_id),
                "jumpNumber": int(event["jumpNumber"]),
                "forceTraceJson": json.dumps(force_points, separators=(",", ":")),
                "imuTraceJson": json.dumps(imu_points, separators=(",", ":")),
                "forceMovementOnsetS": movement_onset - force_takeoff,
                "forceTakeoffS": 0.0,
                "forceLandingS": force_landing - force_takeoff,
                "imuTakeoffS": 0.0,
                "imuFreefallEndS": imu_freefall_end - imu_takeoff,
                "imuLandingS": imu_landing - imu_takeoff,
                "bodyweightN": float(event["bodyweight_n"]),
                "imuThresholdG": float(threshold_by_session.get(session_id, 0.65)),
                "forceSourceFile": str(mapping["cmj_file"]),
                "imuSourceFile": str(mapping["imu_file"]),
            })
    return pd.DataFrame(rows).sort_values(["sessionId", "jumpId"]).reset_index(drop=True)


def source_metadata(
    table: str,
    coverage: str,
    generated_at: str,
    raw_signal_files: list[str] | None = None,
) -> dict:
    definitions = {
        "events": [
            {
                "label": "Jump height",
                "definition": "Takeoff velocity squared divided by 2g, converted from metres to centimetres.",
                "componentIds": ["athlete-height-trend", "coach-height-distribution", "scientist-event-table"],
                "sourceLineage": [{"files": [SOURCE_FILES[0]]}],
            },
            {
                "label": "Relative net impulse",
                "definition": "Force-plate net impulse divided by body mass, in N·s/kg.",
                "componentIds": ["athlete-impulse", "coach-athlete-table", "scientist-predictions"],
                "sourceLineage": [{"files": [SOURCE_FILES[1]]}],
            },
        ],
        "acceleration_events": [
            {
                "label": "Acceleration magnitude",
                "definition": "Vector magnitude of the filtered X-, Y-, and Z-axis IMU acceleration signal, expressed in g.",
                "componentIds": ["athlete-acceleration-trend", "scientist-acceleration-distribution", "scientist-acceleration-table"],
                "sourceLineage": [{"files": [SOURCE_FILES[1]]}],
            },
            {
                "label": "Peak acceleration",
                "definition": "Maximum filtered acceleration magnitude within the modeled jump window, expressed in g.",
                "componentIds": ["athlete-acceleration-average-peak", "athlete-acceleration-max-peak", "coach-acceleration-trend", "scientist-acceleration-peak"],
                "sourceLineage": [{"files": [SOURCE_FILES[1]]}],
            },
            {
                "label": "Acceleration variability",
                "definition": "Within-jump standard deviation of filtered acceleration magnitude, expressed in g.",
                "componentIds": ["athlete-acceleration-variability", "scientist-acceleration-variability"],
                "sourceLineage": [{"files": [SOURCE_FILES[1]]}],
            },
            {
                "label": "Acceleration RMS",
                "definition": "Root mean square of filtered acceleration magnitude within the modeled jump window, expressed in g.",
                "componentIds": ["athlete-acceleration-trend", "coach-acceleration-trend"],
                "sourceLineage": [{"files": [SOURCE_FILES[1]]}],
            },
            {
                "label": "Acceleration area",
                "definition": "Area under the filtered acceleration-magnitude curve across the modeled jump window, expressed in g·s.",
                "componentIds": ["scientist-acceleration-area", "scientist-acceleration-table"],
                "sourceLineage": [{"files": [SOURCE_FILES[1]]}],
            },
        ],
        "acceleration_athletes": [
            {
                "label": "Acceleration session change",
                "definition": "Percentage change in session-average peak acceleration from the athlete's first accepted session to the latest accepted session.",
                "componentIds": ["athlete-acceleration-change", "coach-acceleration-change", "coach-athlete-table"],
                "sourceLineage": [{"files": [SOURCE_FILES[1], SOURCE_FILES[5]]}],
            }
        ],
        "athlete_comparison": [
            {
                "label": "Two-athlete comparison",
                "definition": "Athlete-level summaries of matched jump height, filtered IMU acceleration magnitude, jump-height repeatability, session change, and force-plate relative impulse.",
                "componentIds": ["coach-athlete-comparison"],
                "sourceLineage": [{"files": [SOURCE_FILES[0], SOURCE_FILES[1], SOURCE_FILES[5]]}],
            }
        ],
        "acceleration_correlations": [
            {
                "label": "Acceleration feature association",
                "definition": "Pearson correlation between an IMU acceleration feature and force-plate relative impulse across matched jumps; association does not establish causality.",
                "componentIds": ["scientist-acceleration-associations"],
                "sourceLineage": [{"files": [SOURCE_FILES[1]]}],
            }
        ],
        "model_performance": [
            {
                "label": "Grouped cross-validation",
                "definition": "Five-fold research validation grouped by athlete ID, preventing one athlete from appearing in both train and validation folds. This is not an external deployment test.",
                "componentIds": ["scientist-model-comparison", "scientist-model-r2"],
                "sourceLineage": [{"files": [SOURCE_FILES[3], SOURCE_FILES[4]]}],
            }
        ],
        "quality": [
            {
                "label": "Athlete ID completeness",
                "definition": "Share of matched jump rows with a non-empty athlete ID that exists in the project athlete map.",
                "componentIds": ["scientist-id-coverage"],
                "sourceLineage": [{"files": [SOURCE_FILES[0], SOURCE_FILES[5]]}],
            },
            {
                "label": "Repeat coverage",
                "definition": "Share of athletes represented by at least two accepted sessions. Longitudinal change is available only for these athletes.",
                "componentIds": ["coach-repeat-coverage", "scientist-quality-table"],
                "sourceLineage": [{"files": [SOURCE_FILES[0]]}],
            },
            {
                "label": "Event acceptance yield",
                "definition": "Accepted matched events divided by all force or IMU detections, with rejected counts and low-yield sessions reported separately.",
                "componentIds": ["scientist-quality-table", "scientist-session-quality"],
                "sourceLineage": [{"files": [SOURCE_FILES[2]]}],
            },
            {
                "label": "Flight-time agreement",
                "definition": "Absolute difference between force-plate and IMU flight duration for each accepted matched event.",
                "componentIds": ["scientist-quality-table", "scientist-event-table", "scientist-session-quality"],
                "sourceLineage": [{"files": [SOURCE_FILES[0], SOURCE_FILES[2]]}],
            }
        ],
        "session_quality": [
            {
                "label": "Session matching quality",
                "definition": "Per-session force and IMU detection counts, accepted matches, retention, synchronization, and flight-time agreement.",
                "componentIds": ["scientist-session-quality"],
                "sourceLineage": [{"files": [SOURCE_FILES[2]]}],
            }
        ],
        "model_validation": [
            {
                "label": "Deployment readiness",
                "definition": "A conservative gate requiring locked detector settings and evaluation on an untouched external athlete cohort.",
                "componentIds": ["scientist-model-r2", "scientist-quality-table"],
                "sourceLineage": [{"files": [MODEL_VALIDATION_FILE]}],
            }
        ],
        "signal_traces": [
            {
                "label": "CMJ force signal",
                "definition": "Combined vertical force from the left and right force plates, low-pass filtered at 20 Hz and shown relative to the detected force takeoff.",
                "componentIds": ["scientist-data-review"],
                "sourceLineage": [{"files": raw_signal_files or []}],
            },
            {
                "label": "IMU acceleration signal",
                "definition": "Low-pass filtered vector magnitude of the three-axis IMU acceleration signal, shown relative to the detected IMU takeoff.",
                "componentIds": ["scientist-data-review"],
                "sourceLineage": [{"files": raw_signal_files or []}],
            },
            {
                "label": "Detection markers",
                "definition": "Movement onset, takeoff, freefall end, and landing markers retained by the CMJ/IMU event-detection pipeline for the selected matched jump.",
                "componentIds": ["scientist-data-review"],
                "sourceLineage": [{"files": [SOURCE_FILES[0]]}],
            },
        ],
    }
    source_files = [*SOURCE_FILES, MODEL_VALIDATION_FILE]
    if table == "signal_traces":
        source_files = [SOURCE_FILES[0], *(raw_signal_files or [])]
    return {
        "label": f"KivoMotion SQLite: {table}",
        "files": source_files,
        "sql": f"SELECT * FROM {table}",
        "filters": [coverage],
        "executedAt": generated_at,
        "generatedAt": generated_at,
        "metricDefinitions": definitions.get(table, []),
        "evidenceFlow": [
            {"title": "Project outputs", "detail": "Loaded verified batch CSV outputs and athlete_id_map.csv."},
            *([{"title": "Raw signal review", "detail": "Read the synchronized force-plate and IMU CSVs, applied the project 20 Hz preprocessing, and retained compact review windows around each detected jump."}] if table == "signal_traces" else []),
            {"title": "SQLite transformation", "detail": "Joined jump IDs, derived role-ready summaries, and saved them in kivomotion.sqlite3."},
        ],
    }


def build(
    project_root: Path,
    *,
    athlete_map_path: Path | None = None,
    raw_data_dir: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    matched = pd.read_csv(project_root / SOURCE_FILES[0])
    features = pd.read_csv(project_root / SOURCE_FILES[1])
    status = pd.read_csv(project_root / SOURCE_FILES[2])
    leaderboard = pd.read_csv(project_root / SOURCE_FILES[3])
    predictions_raw = pd.read_csv(project_root / SOURCE_FILES[4])
    athlete_map = pd.read_csv(athlete_map_path or project_root / SOURCE_FILES[5])
    athlete_map, athlete_aliases = prepare_athlete_aliases(athlete_map)
    validation_path = project_root / MODEL_VALIDATION_FILE
    if validation_path.exists():
        validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
    else:
        validation_payload = {
            "threshold_mode": "legacy_unknown",
            "grouped_cross_validation": "Unavailable",
            "research_ready_for_unseen_athletes": False,
            "external_holdout_evaluated": False,
            "deployment_ready": False,
            "deployment_blockers": ["Validation manifest has not been generated."],
        }

    if matched["jump_id"].duplicated().any():
        raise ValueError("all_matched_events.csv contains duplicate jump_id values")
    if features["jump_id"].duplicated().any():
        raise ValueError("all_model_features.csv contains duplicate jump_id values")
    if set(matched["jump_id"]) != set(features["jump_id"]):
        raise ValueError("Matched events and model features do not contain the same jump IDs")
    mapped_ids = set(athlete_map["athlete_id"].dropna().astype(str))
    event_ids = set(matched["athlete_id"].dropna().astype(str))
    if not event_ids.issubset(mapped_ids):
        raise ValueError("Matched events include athlete IDs not present in athlete_id_map.csv")
    conflicting_keys = athlete_map.groupby("athlete_key")["athlete_id"].nunique()
    if (conflicting_keys > 1).any():
        raise ValueError("athlete_id_map.csv maps an athlete_key to more than one athlete_id")
    if athlete_map["cmj_file"].duplicated().any() or athlete_map["imu_file"].duplicated().any():
        raise ValueError("athlete_id_map.csv contains duplicate source-file mappings")
    missing_acceleration = sorted(set(ACCELERATION_FIELDS) - set(features.columns))
    if missing_acceleration:
        raise ValueError(f"Model features are missing acceleration columns: {missing_acceleration}")

    combined = matched.merge(
        features[["jump_id", "target_impulse_ns_kg", *ACCELERATION_FIELDS]],
        on="jump_id",
        how="left",
        validate="one_to_one",
    )
    combined["date"] = pd.to_datetime(combined["recording_date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    combined["renamedAthlete"] = combined["athlete_id"].map(athlete_aliases).replace("", pd.NA)
    combined["athleteLabel"] = combined["renamedAthlete"].fillna(combined["athlete_id"].map(short_label))
    combined["jumpNumber"] = combined.groupby("session_id").cumcount() + 1

    resolved_raw_data_dir = (raw_data_dir or project_root / "Phase 2 Data jumps").resolve()
    signal_traces = build_signal_traces(
        project_root,
        combined,
        athlete_map,
        status,
        resolved_raw_data_dir,
    )
    if len(signal_traces) != len(combined):
        raise ValueError(
            f"Signal traces were built for {len(signal_traces)} of {len(combined)} matched jumps from {resolved_raw_data_dir}"
        )

    events = pd.DataFrame(
        {
            "jumpId": combined["jump_id"],
            "athleteId": combined["athlete_id"],
            "athleteLabel": combined["athleteLabel"],
            "renamedAthlete": combined["renamedAthlete"],
            "sessionId": combined["session_id"],
            "date": combined["date"],
            "testNumber": combined["test_number"],
            "jumpNumber": combined["jumpNumber"],
            "jumpHeightCm": combined["jump_height_impulse_m"] * 100,
            "flightHeightCm": combined["jump_height_flight_m"] * 100,
            "takeoffVelocityMs": combined["takeoff_velocity_m_s"],
            "impulseNsKg": combined["target_impulse_ns_kg"],
            "peakForceN": combined["peak_force_n"],
            "peakForceNkg": combined["peak_force_n"] / combined["mass_kg"],
            "peakLandingForceN": combined["peak_landing_force_n"],
            "syncErrorMs": combined["sync_error_s"].abs() * 1000,
            "forceFlightTimeMs": combined["flight_time_s"] * 1000,
            "imuFlightTimeMs": combined["imu_flight_time_s"] * 1000,
            "flightTimeDiffMs": (combined["flight_time_s"] - combined["imu_flight_time_s"]).abs() * 1000,
            "minAccelerationG": combined["min_acc_mag_g"],
            "massKg": combined["mass_kg"],
        }
    ).sort_values(["date", "sessionId", "jumpId"])

    acceleration_events = pd.DataFrame(
        {
            "jumpId": combined["jump_id"],
            "athleteId": combined["athlete_id"],
            "athleteLabel": combined["athleteLabel"],
            "renamedAthlete": combined["renamedAthlete"],
            "sessionId": combined["session_id"],
            "date": combined["date"],
            "jumpNumber": combined["jumpNumber"],
            "jumpHeightCm": combined["jump_height_impulse_m"] * 100,
            "takeoffVelocityMs": combined["takeoff_velocity_m_s"],
            "impulseNsKg": combined["target_impulse_ns_kg"],
            "syncErrorMs": combined["sync_error_s"].abs() * 1000,
            "flightMinAccelerationG": combined["min_acc_mag_g"],
            **{alias: combined[field] for field, (alias, _) in ACCELERATION_FIELDS.items()},
        }
    ).sort_values(["date", "sessionId", "jumpId"])

    acceleration_session_rows = []
    for session_id, group in acceleration_events.groupby("sessionId", sort=True):
        peak = group["accelerationMagnitudeMaxG"]
        acceleration_session_rows.append(
            {
                "sessionId": session_id,
                "athleteId": group["athleteId"].iloc[0],
                "athleteLabel": group["athleteLabel"].iloc[0],
                "renamedAthlete": group["renamedAthlete"].iloc[0],
                "date": group["date"].iloc[0],
                "jumpCount": len(group),
                "avgMeanAccelerationG": group["accelerationMagnitudeMeanG"].mean(),
                "avgPeakAccelerationG": peak.mean(),
                "maxPeakAccelerationG": peak.max(),
                "avgAccelerationVariabilityG": group["accelerationMagnitudeStdG"].mean(),
                "avgAccelerationRangeG": group["accelerationMagnitudeRangeG"].mean(),
                "avgAccelerationRmsG": group["accelerationMagnitudeRmsG"].mean(),
                "avgAccelerationAreaGs": group["accelerationAreaGs"].mean(),
                "peakAccelerationCvPct": peak.std(ddof=0) / peak.mean() * 100,
            }
        )
    acceleration_sessions = pd.DataFrame(acceleration_session_rows).sort_values(["date", "sessionId"])

    acceleration_athlete_rows = []
    for athlete_id, group in acceleration_events.groupby("athleteId", sort=True):
        athlete_sessions = acceleration_sessions[
            acceleration_sessions["athleteId"] == athlete_id
        ].sort_values(["date", "sessionId"])
        peak = group["accelerationMagnitudeMaxG"]
        acceleration_athlete_rows.append(
            {
                "athleteId": athlete_id,
                "athleteLabel": group["athleteLabel"].iloc[0],
                "renamedAthlete": group["renamedAthlete"].iloc[0],
                "sessionCount": group["sessionId"].nunique(),
                "jumpCount": len(group),
                "avgMeanAccelerationG": group["accelerationMagnitudeMeanG"].mean(),
                "avgPeakAccelerationG": peak.mean(),
                "maxPeakAccelerationG": peak.max(),
                "avgAccelerationVariabilityG": group["accelerationMagnitudeStdG"].mean(),
                "avgAccelerationRangeG": group["accelerationMagnitudeRangeG"].mean(),
                "avgAccelerationRmsG": group["accelerationMagnitudeRmsG"].mean(),
                "avgAccelerationAreaGs": group["accelerationAreaGs"].mean(),
                "peakAccelerationCvPct": peak.std(ddof=0) / peak.mean() * 100,
                "peakAccelerationSessionChangePct": percent_change(
                    athlete_sessions["avgPeakAccelerationG"].iloc[0],
                    athlete_sessions["avgPeakAccelerationG"].iloc[-1],
                ) if len(athlete_sessions) > 1 else None,
            }
        )
    acceleration_athletes = pd.DataFrame(acceleration_athlete_rows).sort_values("athleteLabel")

    acceleration_daily_rows = []
    for date, group in acceleration_events.groupby("date", sort=True):
        acceleration_daily_rows.append(
            {
                "date": date,
                "sessionCount": group["sessionId"].nunique(),
                "athleteCount": group["athleteId"].nunique(),
                "jumpCount": len(group),
                "avgPeakAccelerationG": group["accelerationMagnitudeMaxG"].mean(),
                "maxPeakAccelerationG": group["accelerationMagnitudeMaxG"].max(),
                "avgAccelerationVariabilityG": group["accelerationMagnitudeStdG"].mean(),
                "avgAccelerationRmsG": group["accelerationMagnitudeRmsG"].mean(),
                "avgAccelerationAreaGs": group["accelerationAreaGs"].mean(),
            }
        )
    acceleration_daily = pd.DataFrame(acceleration_daily_rows)

    session_rows = []
    for session_id, group in events.groupby("sessionId", sort=True):
        session_rows.append(
            {
                "sessionId": session_id,
                "athleteId": group["athleteId"].iloc[0],
                "athleteLabel": group["athleteLabel"].iloc[0],
                "renamedAthlete": group["renamedAthlete"].iloc[0],
                "date": group["date"].iloc[0],
                "jumpCount": len(group),
                "avgHeightCm": group["jumpHeightCm"].mean(),
                "bestHeightCm": group["jumpHeightCm"].max(),
                "avgVelocityMs": group["takeoffVelocityMs"].mean(),
                "avgImpulseNsKg": group["impulseNsKg"].mean(),
                "avgPeakForceNkg": group["peakForceNkg"].mean(),
                "medianSyncErrorMs": group["syncErrorMs"].median(),
                "maxSyncErrorMs": group["syncErrorMs"].max(),
                "consistencyCvPct": group["jumpHeightCm"].std(ddof=0) / group["jumpHeightCm"].mean() * 100,
            }
        )
    sessions = pd.DataFrame(session_rows).sort_values(["date", "sessionId"])

    athlete_rows = []
    for athlete_id, group in events.groupby("athleteId", sort=True):
        athlete_sessions = sessions[sessions["athleteId"] == athlete_id].sort_values(["date", "sessionId"])
        athlete_rows.append(
            {
                "athleteId": athlete_id,
                "athleteLabel": group["athleteLabel"].iloc[0],
                "renamedAthlete": group["renamedAthlete"].iloc[0],
                "sessionCount": group["sessionId"].nunique(),
                "jumpCount": len(group),
                "avgHeightCm": group["jumpHeightCm"].mean(),
                "bestHeightCm": group["jumpHeightCm"].max(),
                "avgVelocityMs": group["takeoffVelocityMs"].mean(),
                "avgImpulseNsKg": group["impulseNsKg"].mean(),
                "avgPeakForceNkg": group["peakForceNkg"].mean(),
                "firstDate": group["date"].min(),
                "lastDate": group["date"].max(),
                "consistencyCvPct": group["jumpHeightCm"].std(ddof=0) / group["jumpHeightCm"].mean() * 100,
                "sessionChangePct": percent_change(
                    athlete_sessions["avgHeightCm"].iloc[0], athlete_sessions["avgHeightCm"].iloc[-1]
                ) if len(athlete_sessions) > 1 else None,
            }
        )
    athletes = pd.DataFrame(athlete_rows).sort_values("athleteLabel")

    athlete_comparison = athletes.merge(
        acceleration_athletes[
            [
                "athleteId",
                "avgPeakAccelerationG",
                "maxPeakAccelerationG",
                "avgAccelerationVariabilityG",
                "peakAccelerationSessionChangePct",
            ]
        ],
        on="athleteId",
        how="left",
        validate="one_to_one",
    ).sort_values("athleteLabel")

    daily_rows = []
    for date, group in events.groupby("date", sort=True):
        daily_rows.append(
            {
                "date": date,
                "sessionCount": group["sessionId"].nunique(),
                "athleteCount": group["athleteId"].nunique(),
                "jumpCount": len(group),
                "avgHeightCm": group["jumpHeightCm"].mean(),
                "bestHeightCm": group["jumpHeightCm"].max(),
                "avgImpulseNsKg": group["impulseNsKg"].mean(),
                "medianSyncErrorMs": group["syncErrorMs"].median(),
            }
        )
    daily = pd.DataFrame(daily_rows)

    model_performance = leaderboard.rename(
        columns={"mae": "maeNsKg", "rmse": "rmseNsKg", "r2": "r2"}
    ).sort_values("rmseNsKg")
    best_model_row = model_performance.iloc[0]
    prediction_mask = predictions_raw["model"].eq(best_model_row["model"])
    if "feature_set" in predictions_raw and "feature_set" in best_model_row:
        prediction_mask &= predictions_raw["feature_set"].eq(best_model_row["feature_set"])
    predictions = predictions_raw[prediction_mask].merge(
        events[["jumpId", "athleteId", "athleteLabel", "renamedAthlete", "sessionId", "date"]],
        left_on="jump_id", right_on="jumpId", how="left", validate="one_to_one"
    )[["jumpId", "athleteId", "athleteLabel", "renamedAthlete", "sessionId", "date", "actual", "predicted", "residual"]].rename(
        columns={"actual": "actualImpulseNsKg", "predicted": "predictedImpulseNsKg", "residual": "residualNsKg"}
    )

    correlation_fields = {
        "imu_flight_time_s": "IMU flight time",
        "acc_mag_area_g_s": "Acceleration area",
        "gyro_mag_area_deg": "Rotation area",
        "acc_mag_g_f_max": "Peak acceleration",
        "acc_mag_g_f_std": "Acceleration variability",
        "gyro_mag_dps_f_max": "Peak rotation rate",
        "gyro_mag_dps_f_std": "Rotation variability",
    }
    correlation_rows = []
    for field, label in correlation_fields.items():
        value = features[field].corr(features["target_impulse_ns_kg"])
        correlation_rows.append({"feature": label, "field": field, "correlation": value, "absoluteCorrelation": abs(value)})
    feature_correlations = pd.DataFrame(correlation_rows).sort_values("absoluteCorrelation", ascending=False)

    acceleration_correlation_rows = []
    for field, (_, label) in ACCELERATION_FIELDS.items():
        value = features[field].corr(features["target_impulse_ns_kg"])
        acceleration_correlation_rows.append(
            {
                "feature": label,
                "field": field,
                "correlation": value,
                "absoluteCorrelation": abs(value),
            }
        )
    acceleration_correlations = pd.DataFrame(acceleration_correlation_rows).sort_values(
        "absoluteCorrelation", ascending=False
    )

    selected_metrics = ["jumpHeightCm", "takeoffVelocityMs", "impulseNsKg", "peakForceNkg", "syncErrorMs"]
    acceleration_metrics = list(ACCELERATION_FIELDS.values())
    acceleration_columns = [alias for alias, _ in acceleration_metrics]
    repeated_athletes = int((athletes["sessionCount"] >= 2).sum())
    repeat_coverage_pct = float(repeated_athletes / len(athletes) * 100) if len(athletes) else 0.0
    accepted_status = status[status["status"].eq("accepted")]
    accepted_status = accepted_status.copy()
    qa_defaults = {
        "force_events_rejected": accepted_status["force_events"] - accepted_status["accepted_events"],
        "imu_events_rejected": accepted_status["imu_events"] - accepted_status["accepted_events"],
        "force_event_retention_pct": accepted_status["accepted_events"] / accepted_status["force_events"] * 100,
        "imu_event_retention_pct": accepted_status["accepted_events"] / accepted_status["imu_events"] * 100,
        "median_abs_flight_time_diff_s": pd.Series(float("nan"), index=accepted_status.index),
        "flight_time_diff_over_100ms_pct": pd.Series(float("nan"), index=accepted_status.index),
        "flight_time_diff_over_200ms_pct": pd.Series(float("nan"), index=accepted_status.index),
    }
    for column_name, fallback in qa_defaults.items():
        if column_name not in accepted_status:
            accepted_status[column_name] = fallback

    session_quality = accepted_status[
        [
            "session_id", "force_events", "imu_events", "accepted_events",
            "force_events_rejected", "imu_events_rejected",
            "force_event_retention_pct", "imu_event_retention_pct",
            "median_abs_sync_error_s", "median_abs_flight_time_diff_s",
            "flight_time_diff_over_100ms_pct", "flight_time_diff_over_200ms_pct",
        ]
    ].rename(columns={
        "session_id": "sessionId",
        "force_events": "forceEventsDetected",
        "imu_events": "imuEventsDetected",
        "accepted_events": "acceptedEvents",
        "force_events_rejected": "forceEventsRejected",
        "imu_events_rejected": "imuEventsRejected",
        "force_event_retention_pct": "forceEventRetentionPct",
        "imu_event_retention_pct": "imuEventRetentionPct",
        "median_abs_sync_error_s": "medianSyncErrorS",
        "median_abs_flight_time_diff_s": "medianFlightTimeDiffS",
        "flight_time_diff_over_100ms_pct": "flightTimeDiffOver100msPct",
        "flight_time_diff_over_200ms_pct": "flightTimeDiffOver200msPct",
    }).merge(
        sessions[["sessionId", "athleteId", "athleteLabel", "renamedAthlete", "date"]],
        on="sessionId",
        how="left",
        validate="one_to_one",
    ).sort_values(["date", "sessionId"])
    threshold_modes = (
        accepted_status["imu_threshold_mode"].dropna().astype(str).unique().tolist()
        if "imu_threshold_mode" in accepted_status
        else [validation_payload.get("threshold_mode", "legacy_unknown")]
    )
    threshold_mode = threshold_modes[0] if len(threshold_modes) == 1 else "mixed"
    source_paths = [project_root / path for path in [*SOURCE_FILES, MODEL_VALIDATION_FILE]]
    existing_sources = [path for path in source_paths if path.exists()]
    generated_at = datetime.fromtimestamp(
        max(path.stat().st_mtime for path in existing_sources),
        tz=UTC,
    ).replace(microsecond=0).isoformat()
    coverage = f"Accepted CMJ sessions recorded {events['date'].min()} through {events['date'].max()}."
    quality = pd.DataFrame([{
        "matchedSessions": int(events["sessionId"].nunique()),
        "athletes": int(events["athleteId"].nunique()),
        "matchedJumps": int(len(events)),
        "athleteIdCompletenessPct": float(events["athleteId"].notna().mean() * 100),
        "duplicateJumpIds": int(events["jumpId"].duplicated().sum()),
        "missingMetricCells": int(events[selected_metrics].isna().sum().sum()),
        "missingAccelerationCells": int(acceleration_events[acceleration_columns].isna().sum().sum()),
        "medianSyncErrorMs": float(events["syncErrorMs"].median()),
        "maxSyncErrorMs": float(events["syncErrorMs"].max()),
        "goodSyncPct": float((events["syncErrorMs"] <= 100).mean() * 100),
        "forceEventsDetected": int(accepted_status["force_events"].sum()),
        "imuEventsDetected": int(accepted_status["imu_events"].sum()),
        "acceptedEvents": int(accepted_status["accepted_events"].sum()),
        "forceEventsRejected": int(accepted_status["force_events_rejected"].sum()),
        "imuEventsRejected": int(accepted_status["imu_events_rejected"].sum()),
        "forceEventRetentionPct": float(
            accepted_status["accepted_events"].sum() / accepted_status["force_events"].sum() * 100
        ),
        "imuEventRetentionPct": float(
            accepted_status["accepted_events"].sum() / accepted_status["imu_events"].sum() * 100
        ),
        "lowForceRetentionSessions": int((accepted_status["force_event_retention_pct"] < 80).sum()),
        "medianFlightTimeDiffMs": float(events["flightTimeDiffMs"].median()),
        "flightTimeDiffOver100msPct": float((events["flightTimeDiffMs"] > 100).mean() * 100),
        "flightTimeDiffOver200msPct": float((events["flightTimeDiffMs"] > 200).mean() * 100),
        "repeatedAthletes": repeated_athletes,
        "singleSessionAthletes": int(len(athletes) - repeated_athletes),
        "repeatCoveragePct": repeat_coverage_pct,
        "longitudinalCoverageStatus": "limited" if repeat_coverage_pct < 50 else "established",
        "imuThresholdMode": threshold_mode,
        "externalHoldoutEvaluated": bool(validation_payload.get("external_holdout_evaluated", False)),
        "deploymentReady": bool(validation_payload.get("deployment_ready", False)),
        "dataStartDate": events["date"].min(),
        "dataEndDate": events["date"].max(),
        "snapshotGeneratedAt": generated_at,
    }])

    model_validation = pd.DataFrame([{
        "thresholdMode": validation_payload.get("threshold_mode", threshold_mode),
        "imuThresholdG": validation_payload.get("imu_threshold_g"),
        "imuMinFreefallS": validation_payload.get("imu_min_freefall_s"),
        "groupedCrossValidation": validation_payload.get("grouped_cross_validation", "Unavailable"),
        "researchReadyForUnseenAthletes": bool(validation_payload.get("research_ready_for_unseen_athletes", False)),
        "externalHoldoutEvaluated": bool(validation_payload.get("external_holdout_evaluated", False)),
        "deploymentReady": bool(validation_payload.get("deployment_ready", False)),
        "deploymentBlockers": "; ".join(validation_payload.get("deployment_blockers", [])),
        "modelArtifactVersion": validation_payload.get("model_artifact_version"),
        "modelTrainingRows": validation_payload.get("model_training_rows"),
        "modelFeatureCount": validation_payload.get("model_feature_count"),
        "modelDatasetSha256": validation_payload.get("model_dataset_sha256"),
    }])

    accepted_sessions = status[status["status"].eq("accepted")]["session_id"].dropna().nunique()
    if accepted_sessions != len(sessions):
        raise ValueError(f"Accepted-session count ({accepted_sessions}) does not match derived sessions ({len(sessions)})")

    athlete_alias_table = pd.DataFrame({
        "athleteId": sorted(event_ids),
    })
    athlete_alias_table["generatedAthleteLabel"] = athlete_alias_table["athleteId"].map(short_label)
    athlete_alias_table["renamedAthlete"] = athlete_alias_table["athleteId"].map(athlete_aliases).replace("", pd.NA)
    athlete_alias_table["athleteLabel"] = athlete_alias_table["renamedAthlete"].fillna(
        athlete_alias_table["generatedAthleteLabel"]
    )

    tables = {
        "events": events,
        "signal_traces": signal_traces,
        "acceleration_events": acceleration_events,
        "acceleration_sessions": acceleration_sessions,
        "acceleration_athletes": acceleration_athletes,
        "acceleration_daily": acceleration_daily,
        "acceleration_correlations": acceleration_correlations,
        "athlete_comparison": athlete_comparison,
        "athlete_aliases": athlete_alias_table,
        "sessions": sessions,
        "athletes": athletes,
        "daily": daily,
        "model_performance": model_performance,
        "predictions": predictions,
        "feature_correlations": feature_correlations,
        "quality": quality,
        "session_quality": session_quality,
        "model_validation": model_validation,
        "raw_matched_events": matched,
        "raw_model_features": features,
        "raw_session_status": status,
        "raw_model_leaderboard": leaderboard,
        "raw_out_of_fold_predictions": predictions_raw,
        "athlete_id_map": athlete_map,
    }
    snapshot = {
        "title": "KivoMotion — Performance Intelligence",
        "description": "Role-aware CMJ performance, coaching, model-validation, and data-quality dashboard.",
        "buildStatus": "creating",
        "generatedAt": generated_at,
        "filters": [],
        "queries": {
            name: {
                "rows": records(tables[name]),
                "source": source_metadata(
                    name,
                    coverage,
                    generated_at,
                    sorted(set(signal_traces["forceSourceFile"]).union(signal_traces["imuSourceFile"]))
                    if name == "signal_traces" else None,
                ),
                **({"payloadColumns": ["forceTraceJson", "imuTraceJson"]} if name == "signal_traces" else {}),
            }
            for name in [
                "events",
                "signal_traces",
                "acceleration_events",
                "acceleration_sessions",
                "acceleration_athletes",
                "acceleration_daily",
                "acceleration_correlations",
                "athlete_comparison",
                "sessions",
                "athletes",
                "daily",
                "model_performance",
                "predictions",
                "feature_correlations",
                "quality",
                "session_quality",
                "model_validation",
            ]
        },
    }
    return tables, snapshot


def write_database(database_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = database_path.with_name(f".{database_path.name}.{os.getpid()}.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        with sqlite3.connect(temporary_path) as connection:
            for name, frame in tables.items():
                frame.to_sql(name, connection, if_exists="replace", index=False)
            statements = [
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_jump_id ON events(jumpId)",
                "CREATE INDEX IF NOT EXISTS idx_events_athlete_date ON events(athleteId, date)",
                "CREATE INDEX IF NOT EXISTS idx_events_session ON events(sessionId)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_signal_traces_jump_id ON signal_traces(jumpId)",
                "CREATE INDEX IF NOT EXISTS idx_signal_traces_session ON signal_traces(sessionId)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_acceleration_events_jump_id ON acceleration_events(jumpId)",
                "CREATE INDEX IF NOT EXISTS idx_acceleration_events_athlete_date ON acceleration_events(athleteId, date)",
                "CREATE INDEX IF NOT EXISTS idx_acceleration_events_session ON acceleration_events(sessionId)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_acceleration_sessions_id ON acceleration_sessions(sessionId)",
                "CREATE INDEX IF NOT EXISTS idx_acceleration_sessions_athlete_date ON acceleration_sessions(athleteId, date)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_acceleration_athletes_id ON acceleration_athletes(athleteId)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_athlete_comparison_id ON athlete_comparison(athleteId)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_id ON sessions(sessionId)",
                "CREATE INDEX IF NOT EXISTS idx_sessions_athlete_date ON sessions(athleteId, date)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_athletes_id ON athletes(athleteId)",
                "CREATE INDEX IF NOT EXISTS idx_predictions_athlete ON predictions(athleteId)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_jump_id ON predictions(jumpId)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_session_quality_id ON session_quality(sessionId)",
                "CREATE INDEX IF NOT EXISTS idx_session_quality_athlete ON session_quality(athleteId)",
            ]
            for statement in statements:
                connection.execute(statement)
            connection.execute("PRAGMA optimize")
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        os.replace(temporary_path, database_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_snapshot(snapshot_path: Path, snapshot: dict) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = snapshot_path.with_name(f".{snapshot_path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        json.loads(temporary_path.read_text(encoding="utf-8"))
        os.replace(temporary_path, snapshot_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def refresh_artifacts(
    project_root: Path,
    database_path: Path,
    snapshot_path: Path,
    build_status: str = "complete",
    raw_data_dir: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    tables, snapshot = build(project_root.resolve(), raw_data_dir=raw_data_dir)
    existing_id = None
    if snapshot_path.exists():
        try:
            existing_id = json.loads(snapshot_path.read_text(encoding="utf-8"))["id"]
        except (KeyError, json.JSONDecodeError):
            pass
    if existing_id:
        snapshot["id"] = existing_id
    snapshot["buildStatus"] = build_status
    write_database(database_path.resolve(), tables)
    write_snapshot(snapshot_path.resolve(), snapshot)
    return tables, snapshot


def main() -> None:
    script_path = Path(__file__).resolve()
    default_project = script_path.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=default_project)
    parser.add_argument("--database", type=Path, default=script_path.parents[1] / "data" / "kivomotion.sqlite3")
    parser.add_argument("--snapshot", type=Path, default=script_path.parents[1] / "src" / "data.json")
    parser.add_argument("--raw-data-dir", type=Path, help="Directory containing the synchronized raw CMJ and IMU CSVs")
    parser.add_argument(
        "--build-status",
        choices=["creating", "updating", "paused", "complete"],
        default="complete",
    )
    args = parser.parse_args()

    tables, _ = refresh_artifacts(
        args.project_root.resolve(),
        args.database.resolve(),
        args.snapshot.resolve(),
        args.build_status,
        raw_data_dir=args.raw_data_dir.resolve() if args.raw_data_dir else None,
    )
    print(f"Wrote {args.database.resolve()}")
    print(f"Wrote {args.snapshot.resolve()}")
    print(f"Rows: {len(tables['events'])} jumps, {len(tables['sessions'])} sessions, {len(tables['athletes'])} athletes")


if __name__ == "__main__":
    main()
