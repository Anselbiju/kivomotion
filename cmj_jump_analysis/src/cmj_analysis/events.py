from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import contiguous_regions

G = 9.80665
DEFAULT_IMU_FREEFALL_THRESHOLD_G = 0.65
DEFAULT_IMU_MIN_FREEFALL_S = 0.15


def _merge_short_gaps(
    regions: list[tuple[int, int]],
    max_gap_samples: int,
) -> list[tuple[int, int]]:
    """Join threshold regions separated only by a brief filter-ring gap."""
    merged: list[tuple[int, int]] = []
    for start, end in regions:
        if merged and start - merged[-1][1] <= max_gap_samples:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def estimate_bodyweight(force_n: np.ndarray) -> float:
    """Estimate standing bodyweight while excluding off-plate and impact samples."""
    force_n = np.asarray(force_n, dtype=float)
    positive = force_n[force_n > 200.0]
    if len(positive) < 100:
        raise ValueError("Could not find a stable loaded region to estimate bodyweight")
    initial = float(np.median(positive))
    central = force_n[(force_n > 0.65 * initial) & (force_n < 1.35 * initial)]
    return float(np.median(central)) if len(central) else initial


def _movement_onset(force: np.ndarray, takeoff_idx: int, bw_n: float, fs_hz: float) -> int:
    """Find last quiet standing window before the countermovement."""
    quiet_tol = max(0.05 * bw_n, 25.0)
    quiet_samples = max(5, int(round(0.20 * fs_hz)))
    search_start = max(quiet_samples, takeoff_idx - int(round(2.5 * fs_hz)))
    deviation = np.abs(force - bw_n)
    for idx in range(takeoff_idx - quiet_samples, search_start - 1, -1):
        if np.all(deviation[idx - quiet_samples : idx] < quiet_tol):
            return idx
    return max(0, takeoff_idx - int(round(1.0 * fs_hz)))


def detect_force_jumps(
    force: pd.DataFrame,
    fs_hz: float,
    mass_kg: float | None = None,
    flight_threshold_n: float | None = None,
    min_flight_s: float = 0.20,
    max_flight_gap_s: float = 0.05,
) -> pd.DataFrame:
    """Detect force-plate jumps and compute impulse-derived ground truth."""
    t = force["Time"].to_numpy(dtype=float)
    f = force["force_n"].to_numpy(dtype=float)
    bw_n = estimate_bodyweight(f)
    mass_kg = float(mass_kg or bw_n / G)
    threshold = float(flight_threshold_n or max(20.0, 0.05 * bw_n))
    regions = _merge_short_gaps(
        contiguous_regions(f < threshold),
        max_gap_samples=max(0, int(round(max_flight_gap_s * fs_hz))),
    )

    events: list[dict[str, float | int | str]] = []
    for start, end in regions:
        # Require loaded force immediately before takeoff and after landing. This
        # rejects the initial empty plate and the final walk-off.
        if start == 0 or end >= len(f):
            continue
        duration = t[end - 1] - t[start]
        pre = f[max(0, start - int(fs_hz * 0.10)) : start]
        post = f[end : min(len(f), end + int(fs_hz * 0.10))]
        if duration < min_flight_s or np.median(pre) < 0.4 * bw_n or np.median(post) < 0.4 * bw_n:
            continue

        onset = _movement_onset(f, start, bw_n, fs_hz)
        time_push = t[onset : start + 1]
        net_force = f[onset : start + 1] - bw_n
        net_impulse = float(np.trapezoid(net_force, time_push))
        positive_impulse = float(np.trapezoid(np.maximum(net_force, 0.0), time_push))
        takeoff_velocity = net_impulse / mass_kg
        flight_time = float(t[end] - t[start])
        flight_velocity = G * flight_time / 2.0
        peak_force = float(np.max(f[onset:start]))
        landing_window = f[end : min(len(f), end + int(round(0.25 * fs_hz)))]

        events.append(
            {
                "jump_id": len(events) + 1,
                "jump_type": "CMJ",
                "movement_onset_s": float(t[onset]),
                "takeoff_s": float(t[start]),
                "landing_s": float(t[end]),
                "flight_time_s": flight_time,
                "bodyweight_n": bw_n,
                "mass_kg": mass_kg,
                "net_impulse_ns": net_impulse,
                "positive_impulse_ns": positive_impulse,
                "takeoff_velocity_m_s": takeoff_velocity,
                "flight_velocity_m_s": flight_velocity,
                "jump_height_impulse_m": takeoff_velocity**2 / (2.0 * G),
                "jump_height_flight_m": G * flight_time**2 / 8.0,
                "peak_force_n": peak_force,
                "peak_landing_force_n": float(np.max(landing_window)),
            }
        )
    return pd.DataFrame(events)


def detect_imu_jumps(
    imu: pd.DataFrame,
    freefall_threshold_g: float = DEFAULT_IMU_FREEFALL_THRESHOLD_G,
    min_freefall_s: float = DEFAULT_IMU_MIN_FREEFALL_S,
    min_gap_s: float = 1.5,
) -> pd.DataFrame:
    """Detect jumps from sustained near-zero acceleration during flight."""
    t = imu["time_s"].to_numpy(dtype=float)
    acc = imu["acc_mag_g_f"].to_numpy(dtype=float)
    raw_regions = []
    for start, end in contiguous_regions(acc < freefall_threshold_g):
        duration = t[end - 1] - t[start]
        if duration >= min_freefall_s:
            raw_regions.append((start, end))

    # Keep one plausible flight region in each jump cycle.
    regions: list[tuple[int, int]] = []
    for region in raw_regions:
        if not regions or t[region[0]] - t[regions[-1][0]] >= min_gap_s:
            regions.append(region)
        elif (t[region[1] - 1] - t[region[0]]) > (t[regions[-1][1] - 1] - t[regions[-1][0]]):
            regions[-1] = region

    events = []
    for jump_id, (start, end) in enumerate(regions, start=1):
        landing_search = slice(end, min(len(imu), end + int(0.35 * 100)))
        landing_rel = int(np.argmax(imu["acc_mag_g"].to_numpy()[landing_search]))
        landing_idx = min(len(imu) - 1, end + landing_rel)
        events.append(
            {
                "imu_jump_id": jump_id,
                "imu_takeoff_s": float(t[start]),
                "imu_landing_s": float(t[landing_idx]),
                "imu_freefall_end_s": float(t[end - 1]),
                "imu_flight_time_s": float(t[end - 1] - t[start]),
                "min_acc_mag_g": float(np.min(acc[start:end])),
            }
        )
    return pd.DataFrame(events)
