from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.ptp(values)),
        f"{prefix}_rms": float(np.sqrt(np.mean(values**2))),
    }


def extract_imu_features(imu: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """Create one ML row per jump using only the IMU window."""
    rows: list[dict[str, float | int | str]] = []
    for event in matched.itertuples(index=False):
        # Include countermovement, takeoff, flight and landing response.
        start_s = event.imu_takeoff_s - 1.25
        end_s = event.imu_landing_s + 0.35
        window = imu[(imu["time_s"] >= start_s) & (imu["time_s"] <= end_s)]
        if len(window) < 20:
            continue
        row: dict[str, float | int | str] = {
            "jump_id": int(event.jump_id),
            "jump_type": str(event.jump_type),
            "imu_flight_time_s": float(event.imu_flight_time_s),
            "target_impulse_ns": float(event.net_impulse_ns),
            "target_impulse_ns_kg": float(event.net_impulse_ns / event.mass_kg),
            "target_takeoff_velocity_m_s": float(event.takeoff_velocity_m_s),
        }
        for col in ["ax_g_f", "ay_g_f", "az_g_f", "acc_mag_g_f", "gx_dps_f", "gy_dps_f", "gz_dps_f", "gyro_mag_dps_f"]:
            row.update(_safe_stats(window[col].to_numpy(dtype=float), col))
        time = window["time_s"].to_numpy(dtype=float)
        row["acc_mag_area_g_s"] = float(np.trapz(np.abs(window["acc_mag_g_f"] - 1.0), time))
        row["gyro_mag_area_deg"] = float(np.trapz(window["gyro_mag_dps_f"], time))
        rows.append(row)
    return pd.DataFrame(rows)
