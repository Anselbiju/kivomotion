from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


def lowpass(values: np.ndarray, fs_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 3 * (order + 1):
        return values.copy()
    cutoff_hz = min(cutoff_hz, 0.45 * fs_hz)
    sos = butter(order, cutoff_hz, btype="low", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, values)


def preprocess_force(data: pd.DataFrame, fs_hz: float) -> pd.DataFrame:
    out = data.copy()
    out["force_n"] = lowpass(out["Both"].to_numpy(), fs_hz, cutoff_hz=20.0)
    return out


def preprocess_imu(data: pd.DataFrame, fs_hz: float) -> pd.DataFrame:
    out = data.copy()
    accel_cols = ["ax_g", "ay_g", "az_g"]
    gyro_cols = ["gx_dps", "gy_dps", "gz_dps"]
    for col in accel_cols + gyro_cols:
        out[f"{col}_f"] = lowpass(out[col].to_numpy(), fs_hz, cutoff_hz=20.0)
    out["acc_mag_g"] = np.linalg.norm(out[accel_cols].to_numpy(), axis=1)
    out["acc_mag_g_f"] = lowpass(out["acc_mag_g"].to_numpy(), fs_hz, 20.0)
    out["gyro_mag_dps"] = np.linalg.norm(out[gyro_cols].to_numpy(), axis=1)
    out["gyro_mag_dps_f"] = lowpass(out["gyro_mag_dps"].to_numpy(), fs_hz, 20.0)
    return out


def assess_imu_quality(imu: pd.DataFrame, fs_hz: float) -> dict[str, float | int | str | bool]:
    """Summarize mounting orientation and signal quality without changing data.

    The detector intentionally continues to use the validated scalar-magnitude
    preprocessing. Negative filtered magnitudes are recorded here so a future
    detector change can be retuned against force-plate events rather than being
    introduced silently.
    """
    accel_cols = ["ax_g", "ay_g", "az_g"]
    gyro_cols = ["gx_dps", "gy_dps", "gz_dps"]
    acc = imu[accel_cols].to_numpy(dtype=float)
    gyro = imu[gyro_cols].to_numpy(dtype=float)
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    quiet = (gyro_mag < 5.0) & (acc_mag >= 0.85) & (acc_mag <= 1.15)
    minimum_quiet = max(20, int(round(0.5 * fs_hz)))
    used_fallback = bool(quiet.sum() < minimum_quiet)
    if used_fallback:
        quiet = np.arange(len(imu)) < min(len(imu), max(minimum_quiet, int(round(fs_hz))))

    gravity = np.median(acc[quiet], axis=0)
    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm and np.isfinite(gravity_norm):
        gravity_unit = gravity / gravity_norm
        dominant_idx = int(np.argmax(np.abs(gravity_unit)))
        dominant_axis = "xyz"[dominant_idx]
        dominant_sign = "positive" if gravity_unit[dominant_idx] >= 0 else "negative"
        orientation_group = f"{dominant_axis.upper()}_{dominant_sign}"
        orientation_review = bool(
            used_fallback or dominant_axis != "y" or abs(float(gravity_unit[1])) < 0.85
        )
    else:
        gravity_unit = np.array([np.nan] * 3)
        orientation_group = "UNKNOWN"
        orientation_review = True

    time_col = "time_s" if "time_s" in imu else "Time"
    times = imu[time_col].to_numpy(dtype=float)
    gaps = np.diff(times)
    positive_gaps = gaps[gaps > 0]
    large_gap_threshold_s = max(0.05, 5.0 / fs_hz)
    filtered_negative = (
        int((imu["acc_mag_g_f"] < 0).sum()) if "acc_mag_g_f" in imu else 0
    )
    return {
        "imu_orientation_group": orientation_group,
        "imu_orientation_review": orientation_review,
        "imu_orientation_fallback_used": used_fallback,
        "imu_quiet_samples": int(quiet.sum()),
        "imu_gravity_unit_x": float(gravity_unit[0]),
        "imu_gravity_unit_y": float(gravity_unit[1]),
        "imu_gravity_unit_z": float(gravity_unit[2]),
        "imu_quiet_acceleration_median_g": float(np.median(acc_mag[quiet])),
        "imu_negative_filtered_magnitude_samples": filtered_negative,
        "imu_acceleration_saturation_samples": int((np.abs(acc) >= 15.9).sum()),
        "imu_max_sample_gap_s": float(positive_gaps.max()) if len(positive_gaps) else 0.0,
        "imu_large_sample_gaps": int((positive_gaps > large_gap_threshold_s).sum()),
    }


def contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open [start, end) regions where a Boolean mask is true."""
    mask = np.asarray(mask, dtype=bool)
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [tuple(pair) for pair in boundaries.reshape(-1, 2)]
