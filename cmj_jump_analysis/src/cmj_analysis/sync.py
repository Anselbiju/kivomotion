from __future__ import annotations

import numpy as np
import pandas as pd


def match_events(
    force_events: pd.DataFrame,
    imu_events: pd.DataFrame,
    expected_offset_s: float | None = None,
    max_initial_error_s: float = 0.35,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Robustly map IMU elapsed time onto force elapsed time and pair events.

    Order matching is appropriate within one trial. With >=3 events an affine
    fit also estimates small clock-rate drift. Outlier pairs are rejected using
    the median absolute deviation.
    """
    if force_events.empty or imu_events.empty:
        raise ValueError("Both force and IMU events are required for matching")
    force_t = force_events["takeoff_s"].to_numpy(dtype=float)
    imu_t = imu_events["imu_takeoff_s"].to_numpy(dtype=float)

    # Every force/IMU event combination proposes a translation. Score each
    # proposal by the number of unique near-neighbours. The file-clock prior
    # resolves the otherwise ambiguous one-jump shift in periodic trials.
    proposals = []
    for ft in force_t:
        for it in imu_t:
            candidate = float(ft - it)
            if expected_offset_s is None or abs(candidate - expected_offset_s) <= 3.0:
                proposals.append(candidate)
    if not proposals:
        proposals = [float(ft - it) for ft in force_t for it in imu_t]

    best_pairs: list[tuple[int, int, float]] = []
    best_key = None
    for offset in proposals:
        candidates = []
        for fi, ft in enumerate(force_t):
            for ii, it in enumerate(imu_t):
                error = abs(ft - (it + offset))
                if error <= max_initial_error_s:
                    candidates.append((error, fi, ii))
        used_f, used_i, pairs = set(), set(), []
        for error, fi, ii in sorted(candidates):
            if fi not in used_f and ii not in used_i:
                used_f.add(fi); used_i.add(ii); pairs.append((fi, ii, error))
        median_error = float(np.median([p[2] for p in pairs])) if pairs else float("inf")
        prior_error = abs(offset - expected_offset_s) if expected_offset_s is not None else 0.0
        key = (-len(pairs), median_error, prior_error)
        if best_key is None or key < best_key:
            best_key, best_pairs = key, pairs

    if not best_pairs:
        raise ValueError("No force/IMU events could be aligned")
    best_pairs.sort(key=lambda pair: pair[0])
    force_idx = [pair[0] for pair in best_pairs]
    imu_idx = [pair[1] for pair in best_pairs]
    paired_force_t = force_t[force_idx]
    paired_imu_t = imu_t[imu_idx]
    offset0 = float(np.median(paired_force_t - paired_imu_t))
    residual0 = (paired_force_t - paired_imu_t) - offset0
    mad = float(np.median(np.abs(residual0 - np.median(residual0))))
    keep = np.abs(residual0) <= max(0.15, 3.0 * 1.4826 * mad)
    if keep.sum() >= 3:
        slope, intercept = np.polyfit(paired_imu_t[keep], paired_force_t[keep], 1)
    else:
        slope, intercept = 1.0, offset0

    paired = pd.concat(
        [force_events.iloc[force_idx].reset_index(drop=True), imu_events.iloc[imu_idx].reset_index(drop=True)],
        axis=1,
    )
    paired["imu_takeoff_on_force_s"] = slope * paired["imu_takeoff_s"] + intercept
    paired["sync_error_s"] = paired["takeoff_s"] - paired["imu_takeoff_on_force_s"]
    paired["sync_quality"] = np.where(paired["sync_error_s"].abs() <= 0.10, "good", "review")
    return paired, {
        "clock_scale": float(slope),
        "clock_offset_s": float(intercept),
        "median_abs_sync_error_s": float(paired["sync_error_s"].abs().median()),
        "nominal_clock_offset_s": float(expected_offset_s) if expected_offset_s is not None else None,
        "force_events_detected": int(len(force_events)),
        "imu_events_detected": int(len(imu_events)),
        "events_matched": int(len(paired)),
    }
