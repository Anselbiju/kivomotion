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


def contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open [start, end) regions where a Boolean mask is true."""
    mask = np.asarray(mask, dtype=bool)
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [tuple(pair) for pair in boundaries.reshape(-1, 2)]

