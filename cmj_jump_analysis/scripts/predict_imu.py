#!/usr/bin/env python3
"""Predict normalized CMJ impulse/takeoff velocity from one new IMU CSV."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import pandas as pd

from cmj_analysis.events import (
    DEFAULT_IMU_FREEFALL_THRESHOLD_G,
    DEFAULT_IMU_MIN_FREEFALL_S,
    detect_imu_jumps,
)
from cmj_analysis.features import extract_imu_inference_features
from cmj_analysis.io import estimate_sampling_rate, read_imu_csv
from cmj_analysis.reporting import create_imu_prediction_report
from cmj_analysis.signals import assess_imu_quality, preprocess_imu


def predict_imu_recording(
    imu_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    freefall_threshold_g: float = DEFAULT_IMU_FREEFALL_THRESHOLD_G,
    min_freefall_s: float = DEFAULT_IMU_MIN_FREEFALL_S,
    mass_kg: float | None = None,
    expected_jumps: int | None = None,
    save_preprocessed: bool = False,
) -> pd.DataFrame:
    """Run target-free detection, feature extraction and model inference."""
    imu_path = Path(imu_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not imu_path.is_file():
        raise FileNotFoundError(f"IMU file not found: {imu_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run scripts/run_batch.py first."
        )
    if mass_kg is not None and mass_kg <= 0:
        raise ValueError("mass_kg must be positive")

    recording = read_imu_csv(imu_path)
    fs_hz = estimate_sampling_rate(recording.data["time_s"])
    imu = preprocess_imu(recording.data, fs_hz)
    quality = assess_imu_quality(imu, fs_hz)
    events = detect_imu_jumps(
        imu,
        freefall_threshold_g=freefall_threshold_g,
        min_freefall_s=min_freefall_s,
    )
    if events.empty:
        raise RuntimeError(
            "No IMU jumps were detected. Review the signal, placement and detector settings."
        )
    features = extract_imu_inference_features(imu, events)
    if features.empty:
        raise RuntimeError("Detected events did not contain enough samples for feature extraction")

    artifact = joblib.load(model_path)
    required_keys = {"model", "features", "target"}
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise ValueError("Unsupported model artifact: model/features/target metadata are required")
    target = str(artifact["target"])
    if target not in {"target_impulse_ns_kg", "target_takeoff_velocity_m_s"}:
        raise ValueError(
            f"The IMU prediction report currently requires a normalized impulse/velocity model; got {target!r}"
        )
    missing = [name for name in artifact["features"] if name not in features.columns]
    if missing:
        raise ValueError(f"IMU feature extraction is missing model inputs: {missing}")

    prediction = artifact["model"].predict(features[artifact["features"]])
    predictions = features[
        ["jump_id", "jump_type", "imu_takeoff_s", "imu_landing_s", "imu_flight_time_s"]
    ].copy()
    predictions["predicted_normalized_impulse_ns_kg"] = prediction
    predictions["predicted_takeoff_velocity_m_s"] = prediction
    if mass_kg is not None:
        predictions["mass_kg"] = float(mass_kg)
        predictions["predicted_net_impulse_ns"] = prediction * float(mass_kg)

    warnings: list[str] = []
    if not 95.0 <= fs_hz <= 105.0:
        warnings.append(f"Sampling rate {fs_hz:.2f} Hz is outside the validated current-data range.")
    if quality["imu_orientation_review"]:
        warnings.append("IMU resting orientation requires manual review.")
    if quality["imu_negative_filtered_magnitude_samples"]:
        warnings.append(
            f"Filtered magnitude contains {quality['imu_negative_filtered_magnitude_samples']} negative ringing samples; the validated detector was preserved."
        )
    if quality["imu_large_sample_gaps"]:
        warnings.append(f"Detected {quality['imu_large_sample_gaps']} timestamp gaps above the QC threshold.")
    if quality["imu_acceleration_saturation_samples"]:
        warnings.append(
            f"Detected {quality['imu_acceleration_saturation_samples']} acceleration samples near the ±16 g limit."
        )
    if len(events) < 3:
        warnings.append("Fewer than three jumps were detected; inspect the event table manually.")
    if expected_jumps is not None and len(events) != expected_jumps:
        warnings.append(f"Expected {expected_jumps} jumps but detected {len(events)}.")
    if artifact.get("feature_set") != "magnitude_only":
        warnings.append("The loaded model uses orientation-sensitive axis features.")
    trained_detector = artifact.get("imu_detector") or {}
    trained_threshold = trained_detector.get("freefall_threshold_g")
    trained_duration = trained_detector.get("min_freefall_s")
    if isinstance(trained_threshold, (int, float)) and abs(float(trained_threshold) - freefall_threshold_g) > 1e-9:
        warnings.append(
            f"Inference threshold {freefall_threshold_g:g} g differs from training threshold {float(trained_threshold):g} g."
        )
    if isinstance(trained_duration, (int, float)) and abs(float(trained_duration) - min_freefall_s) > 1e-9:
        warnings.append(
            f"Inference free-fall duration {min_freefall_s:g} s differs from training duration {float(trained_duration):g} s."
        )
    if not artifact.get("research_ready", False):
        warnings.append("Model validation is grouped by session, not by verified athlete.")

    quality.update(
        {
            "imu_source": str(imu_path),
            "model_source": str(model_path),
            "sampling_rate_hz": float(fs_hz),
            "cleaned_samples": int(len(imu)),
            "detector_freefall_threshold_g": float(freefall_threshold_g),
            "detector_min_freefall_s": float(min_freefall_s),
            "detected_jumps": int(len(events)),
            "feature_rows": int(len(features)),
            "model_feature_set": str(artifact.get("feature_set", "unknown")),
            "model_target": target,
            "mass_kg": float(mass_kg) if mass_kg is not None else None,
            "warnings": warnings,
        }
    )

    events.to_csv(output_dir / "imu_events.csv", index=False)
    features.to_csv(output_dir / "imu_features.csv", index=False)
    predictions.to_csv(output_dir / "imu_predictions.csv", index=False)
    (output_dir / "imu_quality.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    if save_preprocessed:
        imu.to_csv(output_dir / "imu_preprocessed.csv", index=False)
    create_imu_prediction_report(predictions, quality, artifact, output_dir)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imu", type=Path, required=True, help="New current-format IMU CSV")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "batch_results" / "models" / "best_impulse_model.joblib",
        help="Trained artifact from scripts/run_batch.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: imu_predictions/<input filename>",
    )
    parser.add_argument("--mass-kg", type=float, default=None, help="Optional mass for raw N·s output")
    parser.add_argument("--expected-jumps", type=int, default=None, help="Optional event-count QC check")
    parser.add_argument("--freefall-threshold-g", type=float, default=DEFAULT_IMU_FREEFALL_THRESHOLD_G)
    parser.add_argument("--min-freefall-s", type=float, default=DEFAULT_IMU_MIN_FREEFALL_S)
    parser.add_argument("--save-preprocessed", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or ROOT / "imu_predictions" / args.imu.stem
    predictions = predict_imu_recording(
        args.imu,
        args.model,
        output_dir,
        freefall_threshold_g=args.freefall_threshold_g,
        min_freefall_s=args.min_freefall_s,
        mass_kg=args.mass_kg,
        expected_jumps=args.expected_jumps,
        save_preprocessed=args.save_preprocessed,
    )
    print(f"Detected and predicted {len(predictions)} jumps")
    print(predictions.round(4).to_string(index=False))
    print(f"Outputs: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
