from __future__ import annotations

"""
KIVO Phase 2 — Single-file application
========================================
This one Python file contains:
  • portable dataset discovery
  • ForceDeck + IMU processing pipeline
  • jump detection / phase processing / metrics / QC
  • demo role-based login
  • Coach, Athlete and Sport Scientist dashboards

Dataset placement (portable): put the folder "Phase 2 Data jumps" beside this file,
or set environment variable KIVO_DATA_DIR to the dataset folder.

Install once:
    python -m pip install fastapi uvicorn pandas numpy scipy plotly python-multipart itsdangerous

Run:
    python kivo_phase2_all_in_one.py

Open:
    http://127.0.0.1:8000

Demo accounts:
    coach01 / kivo123
    scientist01 / kivo123
    ATH001 / kivo123
"""

# ==================== CONFIG ====================
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
SIGNALS_DIR = PROCESSED_DIR / "jump_signals"


def find_data_dir() -> Path:
    """Find the Phase 2 dataset without hard-coding a user's computer path."""
    env = os.getenv("KIVO_DATA_DIR")
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())

    candidates += [
        BASE_DIR / "data" / "Phase_2",
        BASE_DIR / "Phase 2 Data jumps",
        BASE_DIR / "Phase 2 Data jumps" / "Phase 2 Data jumps",
        BASE_DIR.parent / "Phase 2 Data jumps",
        BASE_DIR.parent / "Phase 2 Data jumps" / "Phase 2 Data jumps",
    ]

    # Also search one level around the project folder for a matching dataset folder.
    for parent in [BASE_DIR, BASE_DIR.parent]:
        if parent.exists():
            for p in parent.glob("**/Phase 2 Data jumps"):
                if len(p.parts) - len(parent.parts) <= 3:
                    candidates.append(p)

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked or not candidate.exists():
            continue
        checked.append(candidate)
        # Files may be directly inside candidate or in one nested folder.
        if list(candidate.glob("*_CMJ_Test*.csv")) and list(candidate.glob("*_IMU_Test*.csv")):
            return candidate
        nested = candidate / "Phase 2 Data jumps"
        if nested.exists() and list(nested.glob("*_CMJ_Test*.csv")):
            return nested

    raise FileNotFoundError(
        "Phase 2 data folder not found. Put 'Phase 2 Data jumps' beside this project, "
        "inside data/Phase_2, or set the KIVO_DATA_DIR environment variable."
    )


DATA_DIR = None
try:
    DATA_DIR = find_data_dir()
except FileNotFoundError:
    pass

JUMP_METRICS_FILE = PROCESSED_DIR / "jump_metrics.csv"
ALIGNMENT_FILE = PROCESSED_DIR / "alignment_summary.csv"
QC_FILE = PROCESSED_DIR / "qc_results.csv"
MAPPING_FILE = PROCESSED_DIR / "athlete_mapping.csv"

G = 9.81


# ==================== PROCESSING PIPELINE ====================
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.signal import butter, sosfiltfilt, correlate
from scipy.spatial.transform import Rotation


PAIR_RE = re.compile(r"(?P<date>\d{8})_(?P<kind>CMJ|IMU)_Test(?P<test>\d+)\.csv$", re.I)


def _lowpass(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 20:
        return x.copy()
    cutoff = min(cutoff, fs * 0.45)
    sos = butter(4, cutoff, btype="low", fs=fs, output="sos")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sosfiltfilt(sos, x)


def discover_pairs(data_dir: Path) -> list[dict]:
    found: dict[tuple[str, int], dict] = {}
    for path in data_dir.glob("*.csv"):
        m = PAIR_RE.match(path.name)
        if not m:
            continue
        key = (m.group("date"), int(m.group("test")))
        found.setdefault(key, {"date": key[0], "test_number": key[1]})
        found[key][m.group("kind").lower()] = path

    pairs = []
    for (date, test), rec in sorted(found.items()):
        if "cmj" in rec and "imu" in rec:
            rec["test_id"] = f"{date}_Test{test:02d}"
            pairs.append(rec)
    return pairs


def read_force_file(path: Path) -> tuple[dict, pd.DataFrame]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Time,"))
    meta = {}
    for line in lines[:header_idx]:
        if "," in line:
            key, value = line.split(",", 1)
            meta[key.strip()] = value.strip().strip('"')
    df = pd.read_csv(path, skiprows=header_idx)
    for c in ["Time", "Left", "Right", "Both"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Time", "Both"]).reset_index(drop=True)
    return meta, df


def read_imu_file(path: Path) -> tuple[dict, pd.DataFrame]:
    meta = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                break
            text = line[1:].strip()
            if "=" in text:
                k, v = text.split("=", 1)
                meta[k.strip()] = v.strip()
    df = pd.read_csv(path, comment="#")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return meta, df.dropna(subset=["t_ms"]).reset_index(drop=True)


def process_imu(df: pd.DataFrame, cutoff_hz: float = 12.0) -> pd.DataFrame:
    required = {"t_ms", "ax_g", "ay_g", "az_g", "qw", "qx", "qy", "qz"}
    if not required.issubset(df.columns):
        raise ValueError(f"IMU file missing columns: {sorted(required - set(df.columns))}")

    out = df.copy()
    t = (out["t_ms"].to_numpy(float) - float(out["t_ms"].iloc[0])) / 1000.0
    q = out[["qx", "qy", "qz", "qw"]].to_numpy(float)
    a_device = out[["ax_g", "ay_g", "az_g"]].to_numpy(float)

    # Dataset quaternions rotate device acceleration into a world frame in which +Z ~= gravity.
    world = Rotation.from_quat(q).apply(a_device)
    vertical_g = world[:, 2]
    linear_vertical = (vertical_g - 1.0) * G

    dt = np.diff(t)
    fs = 1.0 / np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 100.0
    filtered = _lowpass(linear_vertical, fs, cutoff_hz)

    out["time_rel_s"] = t
    out["world_ax_g"] = world[:, 0]
    out["world_ay_g"] = world[:, 1]
    out["world_az_g"] = world[:, 2]
    out["vertical_acc_ms2"] = linear_vertical
    out["vertical_acc_filt_ms2"] = filtered
    return out


def align_imu_to_force(force_t, force_acc, imu_t, imu_acc, target_fs: float = 100.0):
    """Return the ForceDeck-time offset of IMU t=0 and a correlation quality score."""
    if len(force_t) < 100 or len(imu_t) < 100:
        return 0.0, np.nan, 1.0

    ft = np.arange(float(force_t[0]), float(force_t[-1]), 1.0 / target_fs)
    force_u = np.interp(ft, force_t, force_acc)
    it = np.arange(0.0, float(imu_t[-1]), 1.0 / target_fs)
    imu_u = np.interp(it, imu_t, imu_acc)

    force_u = _lowpass(force_u, target_fs, 10.0)
    imu_u = _lowpass(imu_u, target_fs, 10.0)

    # Robust clipping reduces influence of landing spikes.
    def norm(x):
        x = x - np.nanmedian(x)
        lo, hi = np.nanquantile(x, [0.01, 0.99])
        x = np.clip(x, lo, hi)
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)

    A, B = norm(force_u), norm(imu_u)
    if len(A) < len(B):
        return 0.0, np.nan, 1.0
    c = correlate(A, B, mode="valid", method="fft")
    idx = int(np.nanargmax(np.abs(c)))
    sign = 1.0 if c[idx] >= 0 else -1.0
    segment = A[idx:idx + len(B)]
    if len(segment) == len(B) and np.nanstd(segment) > 1e-9 and np.nanstd(B) > 1e-9:
        quality = float(abs(np.corrcoef(segment, B)[0, 1]))
    else:
        quality = np.nan
    offset = float(ft[idx])
    return offset, quality, sign


def detect_flight_phases(time_s, force_filt, bodyweight_N, fs):
    low_threshold = max(30.0, 0.05 * bodyweight_N)
    low = np.asarray(force_filt) < low_threshold
    edges = np.diff(np.r_[False, low, False].astype(int))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    flights = []
    for a, b in zip(starts, ends):
        dur = (b - a) / fs
        if not (0.12 <= dur <= 0.80):
            continue
        pre = force_filt[max(0, a - int(0.30 * fs)):a]
        post = force_filt[b:min(len(force_filt), b + int(0.30 * fs))]
        if len(pre) < int(0.08 * fs) or len(post) < int(0.08 * fs):
            continue
        if np.nanmedian(pre) > 0.50 * bodyweight_N and np.nanmedian(post) > 0.50 * bodyweight_N:
            flights.append((a, b))
    return flights


def find_onset(force_filt, bodyweight_N, takeoff_idx, fs):
    start = max(0, takeoff_idx - int(2.0 * fs))
    stop = max(start + 1, takeoff_idx - int(0.12 * fs))
    quiet = np.abs(force_filt - bodyweight_N) < 0.035 * bodyweight_N
    win = max(5, int(0.10 * fs))
    q = quiet[start:stop].astype(float)
    if len(q) < win:
        return max(start, takeoff_idx - int(0.8 * fs))
    roll = np.convolve(q, np.ones(win), mode="valid") >= win * 0.95
    indices = np.where(roll)[0]
    if len(indices):
        # End of the last quiet window before the movement.
        onset = start + int(indices[-1]) + win
        return min(onset, takeoff_idx - int(0.08 * fs))
    return max(start, takeoff_idx - int(0.8 * fs))


def build_mapping(pairs: list[dict]) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if MAPPING_FILE.exists():
        old = pd.read_csv(MAPPING_FILE, dtype=str)
    else:
        old = pd.DataFrame(columns=["test_id", "athlete_id", "display_name"])

    old_map = {str(r.test_id): (str(r.athlete_id), str(r.display_name)) for r in old.itertuples(index=False)}
    rows = []
    next_num = 1
    used = {v[0] for v in old_map.values()}
    for p in pairs:
        tid = p["test_id"]
        if tid in old_map:
            athlete_id, display_name = old_map[tid]
        else:
            while f"ATH{next_num:03d}" in used:
                next_num += 1
            athlete_id = f"ATH{next_num:03d}"
            display_name = athlete_id
            used.add(athlete_id)
            next_num += 1
        rows.append({"test_id": tid, "athlete_id": athlete_id, "display_name": display_name})
    mapping = pd.DataFrame(rows)
    mapping.to_csv(MAPPING_FILE, index=False)
    return mapping


def process_pair(pair: dict, athlete_id: str, display_name: str):
    meta, force_df = read_force_file(pair["cmj"])
    imu_meta, imu_raw = read_imu_file(pair["imu"])
    imu = process_imu(imu_raw)

    fs = float(meta.get("Frequency", 500.0))
    mass = float(meta.get("Weight", np.nan))
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError(f"Invalid ForceDeck Weight metadata in {pair['cmj'].name}")
    bw = mass * G

    t = force_df["Time"].to_numpy(float)
    force = force_df["Both"].to_numpy(float)
    force_filt = _lowpass(force, fs, 20.0)
    net_force = force_filt - bw
    fd_acc = net_force / mass

    offset_s, align_corr, imu_sign = align_imu_to_force(
        t, fd_acc,
        imu["time_rel_s"].to_numpy(float),
        imu["vertical_acc_filt_ms2"].to_numpy(float),
    )
    imu_force_time = offset_s + imu["time_rel_s"].to_numpy(float)
    imu_acc = imu_sign * imu["vertical_acc_filt_ms2"].to_numpy(float)

    flights = detect_flight_phases(t, force_filt, bw, fs)
    metrics, qc_rows = [], []
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    for jump_idx, (takeoff_idx, landing_idx) in enumerate(flights, start=1):
        onset_idx = find_onset(force_filt, bw, takeoff_idx, fs)
        landing_idx = min(landing_idx, len(t) - 1)
        event_end = min(len(t) - 1, landing_idx + int(0.50 * fs))
        event_start = max(0, onset_idx - int(0.40 * fs))

        move_t = t[onset_idx:event_end + 1]
        move_acc = fd_acc[onset_idx:event_end + 1]
        velocity = cumulative_trapezoid(move_acc, move_t, initial=0.0)
        local_takeoff = takeoff_idx - onset_idx
        local_landing = landing_idx - onset_idx
        pre_take_vel = velocity[:local_takeoff + 1]
        min_local = int(np.nanargmin(pre_take_vel)) if len(pre_take_vel) else 0
        braking_end_idx = onset_idx + min_local

        takeoff_velocity = float(velocity[local_takeoff])
        flight_time = float(t[landing_idx] - t[takeoff_idx])
        velocity_height_m = takeoff_velocity ** 2 / (2 * G) if takeoff_velocity > 0 else np.nan
        flight_height_m = G * flight_time ** 2 / 8.0
        # Use velocity-derived height when physically valid; otherwise retain the flight-time estimate
        # for review rather than squaring a negative take-off velocity.
        jump_height_m = velocity_height_m if np.isfinite(velocity_height_m) else flight_height_m
        time_to_takeoff = float(t[takeoff_idx] - t[onset_idx])

        pre_slice = slice(onset_idx, takeoff_idx + 1)
        braking_slice = slice(onset_idx, braking_end_idx + 1)
        prop_slice = slice(braking_end_idx, takeoff_idx + 1)

        braking_imp = float(trapezoid(net_force[braking_slice], t[braking_slice])) if braking_end_idx > onset_idx else np.nan
        prop_imp = float(trapezoid(net_force[prop_slice], t[prop_slice])) if takeoff_idx > braking_end_idx else np.nan
        total_imp = float(trapezoid(net_force[pre_slice], t[pre_slice]))

        # Velocity aligned to force samples for the jump signal file.
        sig_t = t[event_start:event_end + 1]
        sig_force = force_filt[event_start:event_end + 1]
        sig_net = net_force[event_start:event_end + 1]
        sig_fd_acc = fd_acc[event_start:event_end + 1]
        sig_vel = np.full_like(sig_t, np.nan, dtype=float)
        vel_start = onset_idx - event_start
        sig_vel[vel_start:] = velocity[:len(sig_vel) - vel_start]
        sig_power = sig_force * sig_vel
        sig_imu = np.interp(sig_t, imu_force_time, imu_acc, left=np.nan, right=np.nan)

        pre_vel = sig_vel[vel_start:vel_start + (takeoff_idx - onset_idx + 1)]
        pre_power = sig_power[vel_start:vel_start + (takeoff_idx - onset_idx + 1)]
        imu_pre = sig_imu[vel_start:vel_start + (takeoff_idx - onset_idx + 1)]

        peak_force = float(np.nanmax(force_filt[pre_slice]))
        mean_force = float(np.nanmean(force_filt[pre_slice]))
        peak_velocity = float(np.nanmax(pre_take_vel))
        peak_power = float(np.nanmax(pre_power)) if np.isfinite(pre_power).any() else np.nan
        mean_power = float(np.nanmean(pre_power)) if np.isfinite(pre_power).any() else np.nan
        imu_peak_acc = float(np.nanmax(imu_pre)) if np.isfinite(imu_pre).any() else np.nan
        imu_mean_acc = float(np.nanmean(imu_pre)) if np.isfinite(imu_pre).any() else np.nan

        braking_duration = float(t[braking_end_idx] - t[onset_idx])
        prop_duration = float(t[takeoff_idx] - t[braking_end_idx])
        rsi = jump_height_m / time_to_takeoff if time_to_takeoff > 0 else np.nan

        warnings_list = []
        if not np.isfinite(align_corr) or align_corr < 0.25:
            warnings_list.append("ALIGNMENT_WARNING")
        if not (0.15 <= flight_time <= 1.0):
            warnings_list.append("FLIGHT_TIME_WARNING")
        if not (0.05 <= jump_height_m <= 1.0):
            warnings_list.append("JUMP_HEIGHT_WARNING")
        if takeoff_velocity <= 0:
            warnings_list.append("VELOCITY_WARNING")
        if braking_end_idx <= onset_idx or braking_end_idx >= takeoff_idx:
            warnings_list.append("PHASE_WARNING")
        if not np.isfinite(imu_peak_acc):
            warnings_list.append("IMU_MISSING")

        qc_status = "VALID" if not warnings_list else "REVIEW"
        jump_key = f"{pair['test_id']}_J{jump_idx:02d}"

        row = {
            "jump_key": jump_key,
            "athlete_id": athlete_id,
            "display_name": display_name,
            "session_date": pair["date"],
            "test_id": pair["test_id"],
            "test_number": pair["test_number"],
            "jump_id": jump_idx,
            "body_mass_kg": mass,
            "onset_s": float(t[onset_idx]),
            "braking_end_s": float(t[braking_end_idx]),
            "takeoff_s": float(t[takeoff_idx]),
            "landing_s": float(t[landing_idx]),
            "flight_time_s": flight_time,
            "time_to_takeoff_s": time_to_takeoff,
            "eccentric_duration_s": braking_duration,
            "concentric_duration_s": prop_duration,
            "jump_height_cm": jump_height_m * 100.0,
            "takeoff_velocity_jump_height_cm": velocity_height_m * 100.0 if np.isfinite(velocity_height_m) else np.nan,
            "flight_time_jump_height_cm": flight_height_m * 100.0,
            "rsi_mod": rsi,
            "peak_force_N": peak_force,
            "mean_force_N": mean_force,
            "relative_peak_force_N_kg": peak_force / mass,
            "relative_mean_force_N_kg": mean_force / mass,
            "braking_impulse_Ns": braking_imp,
            "braking_impulse_magnitude_Ns": abs(braking_imp) if np.isfinite(braking_imp) else np.nan,
            "propulsive_impulse_Ns": prop_imp,
            "total_net_impulse_Ns": total_imp,
            "relative_impulse_Ns_kg": total_imp / mass,
            "ecc_con_impulse_ratio": abs(braking_imp) / abs(prop_imp) if np.isfinite(braking_imp) and np.isfinite(prop_imp) and prop_imp != 0 else np.nan,
            "peak_velocity_ms": peak_velocity,
            "takeoff_velocity_ms": takeoff_velocity,
            "peak_power_W": peak_power,
            "mean_power_W": mean_power,
            "relative_peak_power_W_kg": peak_power / mass if np.isfinite(peak_power) else np.nan,
            "imu_peak_acceleration_ms2": imu_peak_acc,
            "imu_mean_acceleration_ms2": imu_mean_acc,
            "alignment_shift_s": offset_s,
            "alignment_corr": align_corr,
            "quality_status": qc_status,
            "quality_flags": "|".join(warnings_list) if warnings_list else "OK",
        }
        metrics.append(row)
        qc_rows.append({
            "jump_key": jump_key, "athlete_id": athlete_id, "test_id": pair["test_id"],
            "jump_id": jump_idx, "quality_status": qc_status,
            "quality_flags": row["quality_flags"], "alignment_corr": align_corr,
        })

        sig = pd.DataFrame({
            "time_s": sig_t,
            "time_from_onset_s": sig_t - float(t[onset_idx]),
            "force_N": sig_force,
            "net_force_N": sig_net,
            "fd_acceleration_ms2": sig_fd_acc,
            "velocity_ms": sig_vel,
            "power_W": sig_power,
            "imu_acceleration_ms2": sig_imu,
        })
        sig.to_csv(SIGNALS_DIR / f"{jump_key}.csv.gz", index=False, compression="gzip")

    alignment_row = {
        "athlete_id": athlete_id,
        "display_name": display_name,
        "session_date": pair["date"],
        "test_id": pair["test_id"],
        "force_file": pair["cmj"].name,
        "imu_file": pair["imu"].name,
        "body_mass_kg": mass,
        "jump_count": len(metrics),
        "imu_to_force_shift_s": offset_s,
        "alignment_corr": align_corr,
        "alignment_sign": imu_sign,
    }
    return metrics, qc_rows, alignment_row


def run_pipeline(force_rebuild: bool = False) -> dict:
    data_dir = find_data_dir()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    if JUMP_METRICS_FILE.exists() and ALIGNMENT_FILE.exists() and not force_rebuild:
        existing = pd.read_csv(JUMP_METRICS_FILE)
        return {
            "status": "existing",
            "data_dir": str(data_dir),
            "jumps": len(existing),
            "tests": existing["test_id"].nunique() if "test_id" in existing.columns else 0,
        }

    pairs = discover_pairs(data_dir)
    if not pairs:
        raise RuntimeError(f"No matched CMJ/IMU file pairs found in {data_dir}")
    mapping = build_mapping(pairs)
    map_lookup = mapping.set_index("test_id").to_dict("index")

    all_metrics, all_qc, all_align = [], [], []
    for i, pair in enumerate(pairs, start=1):
        m = map_lookup[pair["test_id"]]
        metrics, qc, alignment = process_pair(pair, m["athlete_id"], m["display_name"])
        all_metrics.extend(metrics)
        all_qc.extend(qc)
        all_align.append(alignment)
        print(f"[{i:02d}/{len(pairs)}] {pair['test_id']}: {len(metrics)} jumps")

    metrics_df = pd.DataFrame(all_metrics)
    qc_df = pd.DataFrame(all_qc)
    align_df = pd.DataFrame(all_align)
    metrics_df.to_csv(JUMP_METRICS_FILE, index=False)
    qc_df.to_csv(QC_FILE, index=False)
    align_df.to_csv(ALIGNMENT_FILE, index=False)

    return {
        "status": "rebuilt",
        "data_dir": str(data_dir),
        "jumps": len(metrics_df),
        "tests": len(pairs),
        "athletes": metrics_df["athlete_id"].nunique() if not metrics_df.empty else 0,
    }




# ==================== DASHBOARD ====================
from html import escape
from pathlib import Path
from urllib.parse import quote
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware


app = FastAPI(title="KIVO Phase 2")
app.add_middleware(
    SessionMiddleware,
    secret_key="KIVO-DEMO-SESSION-KEY-CHANGE-FOR-PRODUCTION",
    same_site="lax",
    https_only=False,
)

DEMO_PASSWORD = "kivo123"


def ensure_processed():
    if not JUMP_METRICS_FILE.exists():
        run_pipeline(force_rebuild=True)


def load_metrics() -> pd.DataFrame:
    ensure_processed()
    df = pd.read_csv(JUMP_METRICS_FILE)
    for c in ["athlete_id", "display_name", "test_id", "quality_status", "quality_flags"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def load_alignment() -> pd.DataFrame:
    if not ALIGNMENT_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(ALIGNMENT_FILE)


def safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def fmt(x, d=1, suffix=""):
    x = safe_float(x)
    return "—" if not np.isfinite(x) else f"{x:.{d}f}{suffix}"


def get_top_performer(df, metric):
    """
    Return the athlete with the highest median VALID value for a metric.
    Falls back to all rows only when no VALID rows exist.
    """
    if df.empty or metric not in df.columns or "athlete_id" not in df.columns:
        return None, np.nan

    d = df.copy()

    if "quality_status" in d.columns:
        valid = d[d["quality_status"].astype(str).str.upper() == "VALID"].copy()
        if not valid.empty:
            d = valid

    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=["athlete_id", metric])

    if d.empty:
        return None, np.nan

    summary = (
        d.groupby("athlete_id")[metric]
        .median()
        .sort_values(ascending=False)
    )

    if summary.empty:
        return None, np.nan

    return str(summary.index[0]), float(summary.iloc[0])


def median(df, col):
    if df.empty or col not in df.columns:
        return np.nan
    return pd.to_numeric(df[col], errors="coerce").median()


def user_role(request: Request):
    return request.session.get("role")


def user_name(request: Request):
    return request.session.get("username")


def require_role(request: Request, *roles):
    role = user_role(request)
    if not role:
        return RedirectResponse("/login", status_code=303)
    if role not in roles:
        raise HTTPException(status_code=403, detail="This demo account does not have access to this page.")
    return None


def athlete_allowed(request: Request, athlete_id: str):
    role = user_role(request)
    if role in {"coach", "scientist"}:
        return True
    return role == "athlete" and request.session.get("athlete_id") == athlete_id


def plot_html(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(family="Inter, Arial, sans-serif", color="#cfcfcf"),
        margin=dict(l=55, r=25, t=55, b=55),
        xaxis=dict(gridcolor="#242424", linecolor="#3a3a3a", zerolinecolor="#353535"),
        yaxis=dict(gridcolor="#242424", linecolor="#3a3a3a", zerolinecolor="#353535"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#111111", font_color="#ffffff", bordercolor="#555555"),
    )
    return pio.to_html(
        fig, full_html=False, include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
    )


CSS = r"""
:root {
    --bg:#070d18;
    --sidebar:#08111f;
    --panel:#0d1626;
    --panel2:#101b2d;
    --border:#1a2940;
    --border2:#22334d;
    --text:#f4f7fb;
    --muted:#71849e;
    --soft:#aebdd0;
    --cyan:#22d3ee;
    --purple:#705cff;
    --green:#20d7a7;
    --red:#ff6b81;
    --orange:#ff9f43;
}

*{box-sizing:border-box}

html,body{
    margin:0;
    min-height:100%;
    background:var(--bg);
    color:var(--text);
    font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}

a{text-decoration:none;color:inherit}

.app{
    display:grid;
    grid-template-columns:235px minmax(0,1fr);
    min-height:100vh;
}

.sidebar{
    position:sticky;
    top:0;
    height:100vh;
    border-right:1px solid #142139;
    padding:26px 16px;
    background:
        radial-gradient(circle at 10% 2%,rgba(34,211,238,.08),transparent 27%),
        var(--sidebar);
}

.brand{
    display:flex;
    align-items:center;
    gap:11px;
    padding:0 10px 30px;
    font-weight:900;
    font-size:22px;
    letter-spacing:1.2px;
}

.brand-mark{
    width:34px;
    height:34px;
    display:grid;
    place-items:center;
    border-radius:10px;
    color:#07111c;
    background:linear-gradient(135deg,var(--cyan),var(--purple));
    box-shadow:0 0 22px rgba(34,211,238,.12);
    font-size:14px;
}

.nav-label{
    color:#50627c;
    font-size:9px;
    font-weight:800;
    letter-spacing:1.6px;
    text-transform:uppercase;
    padding:8px 12px;
}

.nav{
    display:flex;
    gap:10px;
    align-items:center;
    margin:4px 0;
    padding:11px 12px;
    border-radius:9px;
    color:#8395ae;
    font-size:12px;
    font-weight:700;
}

.nav:hover{
    color:#fff;
    background:#101c2f;
}

.nav.active{
    color:#e7fbff;
    border:1px solid rgba(34,211,238,.16);
    background:linear-gradient(90deg,rgba(34,211,238,.13),rgba(112,92,255,.07));
}

.sidebar-foot{
    position:absolute;
    bottom:22px;
    left:16px;
    right:16px;
    border-top:1px solid #17263c;
    padding-top:14px;
    color:#61748d;
    font-size:9px;
    line-height:1.7;
}

.main{min-width:0}

.topbar{
    height:76px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:16px 30px;
    border-bottom:1px solid #162239;
    background:rgba(7,13,24,.94);
    position:sticky;
    top:0;
    z-index:20;
    backdrop-filter:blur(10px);
}

.title{
    font-size:20px;
    font-weight:850;
    letter-spacing:-.3px;
}

.subtitle{
    font-size:10px;
    color:#71839a;
    margin-top:5px;
}

.userbox{
    display:flex;
    gap:10px;
    align-items:center;
    color:#8da0b8;
    font-size:10px;
}

.role{
    padding:5px 8px;
    border:1px solid rgba(34,211,238,.22);
    border-radius:99px;
    color:#bdf6ff;
    background:rgba(34,211,238,.06);
    text-transform:uppercase;
    font-size:8px;
    font-weight:900;
    letter-spacing:.7px;
}

.logout{
    padding:6px 10px;
    border:1px solid #24405f;
    border-radius:7px;
    color:#bdebf4;
    background:#0f2037;
}

.logout:hover{
    border-color:rgba(34,211,238,.45);
}

.content{
    max-width:1650px;
    margin:auto;
    padding:26px 30px 50px;
}

.kpis{
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:12px;
    margin-bottom:16px;
}

.kpi{
    position:relative;
    overflow:hidden;
    background:linear-gradient(180deg,#101b2d,#0b1524);
    border:1px solid #1b2a40;
    border-radius:12px;
    padding:18px;
    min-height:108px;
}

.kpi:after{
    content:"";
    position:absolute;
    left:0;
    right:0;
    bottom:0;
    height:2px;
    background:linear-gradient(90deg,transparent,rgba(34,211,238,.38),transparent);
}

.kpi-label{
    font-size:8px;
    color:#6d8099;
    text-transform:uppercase;
    font-weight:900;
    letter-spacing:1px;
}

.kpi-value{
    font-size:28px;
    font-weight:850;
    margin-top:10px;
}

.kpi-foot{
    font-size:9px;
    color:#687a91;
    margin-top:7px;
}

.panel{
    background:linear-gradient(180deg,#0e1829,#0a1321);
    border:1px solid #1a2940;
    border-radius:12px;
    overflow:hidden;
}

.panel-head{
    padding:14px 16px;
    border-bottom:1px solid #19273c;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
}

.panel-title{
    color:#dfe8f4;
    font-size:10px;
    text-transform:uppercase;
    font-weight:900;
    letter-spacing:.9px;
}

.panel-sub{
    font-size:9px;
    color:#61748d;
    margin-top:4px;
}

.panel-body{padding:16px}

.grid-2{
    display:grid;
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:14px;
}

.grid-main{
    display:grid;
    grid-template-columns:minmax(0,1.5fr) minmax(320px,.8fr);
    gap:14px;
}

.section{margin:24px 0 12px}

.section-title{
    color:#dbe5f2;
    font-size:10px;
    text-transform:uppercase;
    letter-spacing:1.1px;
    font-weight:900;
}

.section-note{
    font-size:9px;
    color:#60738c;
    margin-top:4px;
}

.tabs{
    display:flex;
    gap:6px;
    flex-wrap:wrap;
}

.tab,.btn{
    border:1px solid #21324b;
    background:#0b1524;
    color:#788ba4;
    border-radius:7px;
    padding:7px 10px;
    font-size:9px;
    font-weight:800;
    cursor:pointer;
}

.tab.active,.tab:hover,.btn:hover{
    color:#dffbff;
    border-color:rgba(34,211,238,.42);
    background:rgba(34,211,238,.08);
}

.table-wrap{overflow:auto}

table{
    width:100%;
    border-collapse:collapse;
    min-width:850px;
}

th{
    background:#0a1423;
    color:#5f728c;
    text-align:left;
    padding:10px 12px;
    border-bottom:1px solid #1a2940;
    font-size:8px;
    text-transform:uppercase;
    letter-spacing:.7px;
}

td{
    padding:11px 12px;
    border-bottom:1px solid #152338;
    color:#b9c6d7;
    font-size:10px;
}

tr.clickable{cursor:pointer}

tr.clickable:hover{
    background:rgba(34,211,238,.04);
}

.strong{
    font-weight:850;
    color:#fff;
}

.status{
    display:inline-block;
    padding:4px 7px;
    border-radius:99px;
    border:1px solid rgba(32,215,167,.18);
    background:rgba(32,215,167,.07);
    font-size:8px;
    font-weight:850;
    color:#8bf0d2;
}

.status.review{
    border-color:rgba(255,107,129,.22);
    background:rgba(255,107,129,.07);
    color:#ff9baf;
}

.metric-grid{
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:10px;
}

.metric-card{
    border:1px solid #1b2d44;
    border-radius:9px;
    background:#0a1524;
    padding:13px;
}

.metric-card .label{
    font-size:8px;
    color:#6d8099;
    text-transform:uppercase;
    font-weight:900;
}

.metric-card .value{
    font-size:17px;
    font-weight:850;
    margin-top:7px;
}

.note{
    border:1px solid rgba(34,211,238,.15);
    border-radius:9px;
    background:rgba(34,211,238,.035);
    padding:12px 14px;
    color:#8498b0;
    font-size:9px;
    line-height:1.6;
}

.filters{
    display:flex;
    gap:10px;
    flex-wrap:wrap;
    padding:14px 16px;
    border-bottom:1px solid #19273c;
    background:#0a1423;
}

.filter{
    display:flex;
    flex-direction:column;
    gap:5px;
    min-width:150px;
}

.filter label{
    font-size:8px;
    color:#647892;
    font-weight:850;
    text-transform:uppercase;
}

.filter select{
    background:#0b1524;
    border:1px solid #21324b;
    color:#dce8f5;
    padding:8px;
    border-radius:7px;
}

.athlete-grid{
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:13px;
}

.athlete-card{
    border:1px solid #1b2b42;
    border-radius:11px;
    background:linear-gradient(180deg,#101b2d,#0a1423);
    padding:16px;
    transition:.15s;
}

.athlete-card:hover{
    border-color:rgba(34,211,238,.45);
    transform:translateY(-2px);
    background:#101d30;
}

.athlete-name{
    font-size:17px;
    font-weight:850;
}

.athlete-meta{
    font-size:9px;
    color:#687b94;
    margin-top:5px;
}

.mini-grid{
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:8px;
    margin-top:14px;
}

.mini{
    border-top:1px solid #17283e;
    padding-top:9px;
}

.mini span{
    display:block;
    color:#657a94;
    font-size:8px;
    text-transform:uppercase;
}

.mini strong{
    display:block;
    margin-top:4px;
    font-size:12px;
}

.breadcrumb{
    font-size:9px;
    color:#6a7d95;
    margin-bottom:15px;
}

.breadcrumb a:hover{color:var(--cyan)}

.empty{
    max-width:720px;
    margin:70px auto;
    border:1px solid #1e3049;
    border-radius:12px;
    padding:30px;
    text-align:center;
    background:#0c1727;
}

.empty p{
    color:#7b8ea6;
    font-size:10px;
    line-height:1.7;
}

.login-wrap{
    min-height:100vh;
    display:grid;
    place-items:center;
    background:
        radial-gradient(circle at 50% 0,rgba(34,211,238,.12),transparent 28%),
        radial-gradient(circle at 72% 30%,rgba(112,92,255,.12),transparent 28%),
        #070d18;
}

.login-card{
    width:min(430px,92vw);
    border:1px solid #1b2d44;
    border-radius:16px;
    background:linear-gradient(180deg,#0f1a2b,#091321);
    padding:32px;
    box-shadow:0 30px 80px rgba(0,0,0,.45);
}

.login-logo{
    width:52px;
    height:52px;
    border-radius:14px;
    display:grid;
    place-items:center;
    margin:auto;
    font-weight:900;
    color:#07111c;
    background:linear-gradient(135deg,var(--cyan),var(--purple));
    box-shadow:0 0 30px rgba(34,211,238,.14);
}

.login-title{
    text-align:center;
    margin-top:18px;
    font-size:26px;
    font-weight:900;
    letter-spacing:3px;
}

.login-sub{
    text-align:center;
    color:#71849e;
    font-size:10px;
    margin:8px 0 24px;
}

.login-card input{
    width:100%;
    padding:12px;
    background:#08111f;
    border:1px solid #21324b;
    border-radius:8px;
    color:#fff;
    margin:6px 0 12px;
    outline:none;
}

.login-card input:focus{
    border-color:rgba(34,211,238,.55);
    box-shadow:0 0 0 3px rgba(34,211,238,.06);
}

.login-card button{
    width:100%;
    padding:12px;
    color:#07111c;
    border:0;
    border-radius:8px;
    font-weight:900;
    cursor:pointer;
    background:linear-gradient(90deg,var(--cyan),#79e9f7);
}

.login-card button:hover{
    filter:brightness(1.05);
}

.demo{
    margin-top:18px;
    padding-top:16px;
    border-top:1px solid #17283e;
    color:#71849e;
    font-size:9px;
    line-height:1.8;
}

.error{
    color:#ffb3c0;
    border:1px solid rgba(255,107,129,.3);
    background:rgba(255,107,129,.06);
    padding:9px;
    border-radius:7px;
    margin-bottom:12px;
    font-size:9px;
}

.chart-empty{
    padding:55px 20px;
    text-align:center;
    color:#6f829b;
    font-size:10px;
}

@media(max-width:1100px){
    .app{grid-template-columns:76px minmax(0,1fr)}
    .brand{justify-content:center;padding-left:0;padding-right:0}
    .brand span:last-child,.nav-label,.nav span:not(.nav-icon),.sidebar-foot{display:none}
    .nav{justify-content:center}
    .kpis{grid-template-columns:repeat(2,minmax(0,1fr))}
    .grid-main,.grid-2{grid-template-columns:1fr}
    .metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
    .athlete-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
}

@media(max-width:720px){
    .app{display:block}
    .sidebar{
        position:relative;
        width:100%;
        height:auto;
        padding:10px;
        display:flex;
        align-items:center;
        gap:5px;
        overflow:auto;
        border-right:none;
        border-bottom:1px solid #17253a;
    }
    .brand{padding:0 7px 0 0;font-size:16px}
    .brand span:last-child,.nav span:not(.nav-icon){display:inline}
    .nav-label,.sidebar-foot{display:none}
    .nav{white-space:nowrap;margin:0}
    .topbar{position:relative;padding:14px}
    .content{padding:17px 14px 40px}
    .kpis,.metric-grid,.athlete-grid{grid-template-columns:1fr}
    .filters{align-items:stretch}
    .filter{min-width:100%}
}
"""


def sidebar(active, request: Request):
    role = user_role(request)
    items = []
    if role == "coach":
        items += [
            ("/coach", "Squad Dashboard", "coach"),
            ("/athletes", "Athletes", "athletes"),
        ]
    elif role == "athlete":
        aid = request.session.get("athlete_id", "")
        items += [(f"/athlete/{quote(aid)}", "My Performance", "athlete")]
    elif role == "scientist":
        items += [
            ("/scientist", "Sport Scientist", "scientist"),
            ("/athletes", "Athletes", "athletes"),
        ]
    nav = "".join(
        f'<a class="nav {"active" if active == key else ""}" href="{href}"><span>◻</span><span>{escape(label)}</span></a>'
        for href, label, key in items
    )
    return f"""
    <aside class="sidebar">
      <a href="/" class="brand"><span class="brand-mark">K</span><span>KIVO</span></a>
      <div class="nav-label">Navigation</div>{nav}
      <div class="sidebar-foot">KIVO Phase 2<br>ForceDeck + IMU<br><span style="color:#bbb">Demo environment</span></div>
    </aside>"""


def page(title, subtitle, body, active, request: Request):
    role = user_role(request) or ""
    username = user_name(request) or ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)} · KIVO</title><style>{CSS}</style></head><body>
    <div class="app">{sidebar(active, request)}<main class="main"><header class="topbar"><div><div class="title">{escape(title)}</div><div class="subtitle">{escape(subtitle)}</div></div><div class="userbox"><span>{escape(username)}</span><span class="role">{escape(role)}</span><a class="logout" href="/logout">Logout</a></div></header><section class="content">{body}</section></main></div></body></html>"""


def empty_data(message):
    return f'<div class="empty"><h2>Data unavailable</h2><p>{escape(message)}</p></div>'


def load_signal(jump_key):
    p = SIGNALS_DIR / f"{jump_key}.csv.gz"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def chart_history(d, metric, label, unit, decimals=1):
    if d.empty or metric not in d.columns:
        return '<div class="chart-empty">No history available.</div>'

    xdf = d.sort_values(["session_date", "test_id", "jump_id"]).copy()
    y = pd.to_numeric(xdf[metric], errors="coerce")
    labels = [
        f"{r.test_id} · J{int(r.jump_id)}"
        for r in xdf.itertuples()
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=y,
            mode="lines+markers",
            line=dict(
                color="#22d3ee",
                width=3,
            ),
            marker=dict(
                color="#22d3ee",
                size=9,
                line=dict(
                    color="#07101d",
                    width=2,
                ),
            ),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,.055)",
            hovertemplate=(
                "<b>%{x}</b><br>"
                + label
                + f": %{{y:.{decimals}f}} "
                + unit
                + "<extra></extra>"
            ),
            name=label,
        )
    )

    if y.notna().sum() > 1:
        avg = float(y.mean())
        fig.add_hline(
            y=avg,
            line_dash="dash",
            line_color="#705cff",
            annotation_text=f"Average {avg:.{decimals}f}",
            annotation_font_color="#a99cff",
            annotation_bgcolor="rgba(13,22,38,.85)",
        )

    fig.update_layout(
        title=dict(
            text=f"{label} History",
            font=dict(color="white", size=16),
        ),
        xaxis_title="Session / Jump",
        yaxis_title=f"{label} ({unit})" if unit else label,
        showlegend=False,
    )

    return plot_html(fig, 370)


def quadrant_chart(df, athlete_a="", athlete_b=""):
    d = df.dropna(
        subset=["rsi_mod", "jump_height_cm"]
    ).copy()

    if d.empty:
        return '<div class="chart-empty">No quadrant data.</div>'

    xm = d["rsi_mod"].median()
    ym = d["jump_height_cm"].median()

    fig = go.Figure()

    others = d[
        ~d["athlete_id"].isin([athlete_a, athlete_b])
    ]

    if not others.empty:
        fig.add_trace(
            go.Scatter(
                x=others["rsi_mod"],
                y=others["jump_height_cm"],
                mode="markers",
                name="Squad",
                marker=dict(
                    color="#71849e",
                    size=8,
                    opacity=.38,
                    line=dict(
                        color="#0d1626",
                        width=1,
                    ),
                ),
                customdata=np.stack(
                    [
                        others["athlete_id"].astype(str),
                        others["jump_id"].astype(str),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> · Jump %{customdata[1]}"
                    "<br>RSI-mod: %{x:.2f}"
                    "<br>Jump height: %{y:.1f} cm"
                    "<extra></extra>"
                ),
            )
        )

    compare_styles = [
        (athlete_a, "Athlete A", "#22d3ee", "circle"),
        (athlete_b, "Athlete B", "#ff9f43", "diamond"),
    ]

    for aid, name, color, symbol in compare_styles:
        g = d[d["athlete_id"] == aid]

        if aid and not g.empty:
            fig.add_trace(
                go.Scatter(
                    x=g["rsi_mod"],
                    y=g["jump_height_cm"],
                    mode="markers",
                    name=f"{name}: {aid}",
                    marker=dict(
                        color=color,
                        size=13,
                        symbol=symbol,
                        line=dict(
                            color="#07101d",
                            width=2,
                        ),
                    ),
                    customdata=np.stack(
                        [
                            g["athlete_id"].astype(str),
                            g["jump_id"].astype(str),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> · Jump %{customdata[1]}"
                        "<br>RSI-mod: %{x:.2f}"
                        "<br>Jump height: %{y:.1f} cm"
                        "<extra></extra>"
                    ),
                )
            )

    fig.add_vline(
        x=xm,
        line_dash="dash",
        line_color="#705cff",
        opacity=.8,
    )
    fig.add_hline(
        y=ym,
        line_dash="dash",
        line_color="#705cff",
        opacity=.8,
    )

    fig.add_annotation(
        x=.99, y=.98,
        xref="paper", yref="paper",
        text="High RSI + High Jump",
        showarrow=False,
        font=dict(size=10, color="#20d7a7"),
    )
    fig.add_annotation(
        x=.99, y=.03,
        xref="paper", yref="paper",
        text="High RSI + Low Jump",
        showarrow=False,
        font=dict(size=10, color="#aebdd0"),
    )
    fig.add_annotation(
        x=.01, y=.98,
        xref="paper", yref="paper",
        text="Low RSI + High Jump",
        showarrow=False,
        xanchor="left",
        font=dict(size=10, color="#aebdd0"),
    )
    fig.add_annotation(
        x=.01, y=.03,
        xref="paper", yref="paper",
        text="Low RSI + Low Jump",
        showarrow=False,
        xanchor="left",
        font=dict(size=10, color="#ff9baf"),
    )

    # Keep the chart title outside the Plotly canvas so it cannot
    # overlap with the legend on smaller screens.
    fig.update_layout(
        title=None,
        xaxis_title="RSI Modified",
        yaxis_title="Jump Height (cm)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
    )

    return plot_html(fig, 470)


def signal_chart(sig, xcol, ycol, title, ylabel, events=None):
    if sig.empty or ycol not in sig.columns:
        return '<div class="chart-empty">Signal unavailable.</div>'

    d = sig[[xcol, ycol]].dropna()

    if d.empty:
        return '<div class="chart-empty">Signal unavailable.</div>'

    title_lower = title.lower()

    if "force" in title_lower:
        line_color = "#22d3ee"
        fill_color = "rgba(34,211,238,.045)"
    elif "acceleration" in title_lower:
        line_color = "#705cff"
        fill_color = "rgba(112,92,255,.045)"
    elif "velocity" in title_lower:
        line_color = "#20d7a7"
        fill_color = "rgba(32,215,167,.045)"
    elif "power" in title_lower:
        line_color = "#ff9f43"
        fill_color = "rgba(255,159,67,.045)"
    else:
        line_color = "#22d3ee"
        fill_color = "rgba(34,211,238,.045)"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d[xcol],
            y=d[ycol],
            mode="lines",
            line=dict(
                color=line_color,
                width=2.5,
            ),
            fill="tozeroy",
            fillcolor=fill_color,
            hovertemplate=(
                "Time: %{x:.3f} s"
                f"<br>{ylabel}: %{{y:.2f}}"
                "<extra></extra>"
            ),
            name=title,
        )
    )

    event_colors = {
        "Movement onset": "#aebdd0",
        "Onset": "#aebdd0",
        "Minimum velocity": "#705cff",
        "Braking end": "#ff9f43",
        "Concentric start": "#ff9f43",
        "Take-off": "#20d7a7",
        "Landing": "#ff6b81",
    }

    for label, x in (events or []):
        if np.isfinite(safe_float(x)):
            color = event_colors.get(
                label,
                "#71849e",
            )

            fig.add_vline(
                x=float(x),
                line_dash="dash",
                line_color=color,
                line_width=1.5,
                opacity=.9,
                annotation_text=label,
                annotation_font_color=color,
                annotation_position="top",
                annotation_bgcolor="rgba(13,22,38,.85)",
            )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                color="white",
                size=16,
            ),
        ),
        xaxis_title="Time from onset (s)",
        yaxis_title=ylabel,
        showlegend=False,
    )

    return plot_html(fig, 365)


@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    if user_role(request):
        return RedirectResponse("/", status_code=303)
    error = request.query_params.get("error")
    err_html = '<div class="error">Invalid demo login ID or password.</div>' if error else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KIVO Login</title><style>{CSS}</style></head><body><div class="login-wrap"><div class="login-card"><div class="login-logo">K</div><div class="login-title">KIVO</div><div class="login-sub">COUNTERMOVEMENT JUMP PERFORMANCE PLATFORM</div>{err_html}<form method="post" action="/login"><label style="font-size:9px;color:#777">LOGIN ID</label><input name="username" placeholder="coach01 / scientist01 / ATH001" required><label style="font-size:9px;color:#777">PASSWORD</label><input name="password" type="password" placeholder="Password" required><button type="submit">SIGN IN</button></form><div class="demo"><b style="color:#bbb">Demo accounts</b><br>Coach: <b>coach01</b> / kivo123<br>Sport Scientist: <b>scientist01</b> / kivo123<br>Athlete: <b>ATH001</b> / kivo123 (ATH002, ATH003, ...)<br><br>This is a demonstration login, not production authentication.</div></div></div></body></html>"""


@app.post("/login")
async def login_post(request: Request):
    form = await request.form()
    username = str(form.get("username", "")).strip()
    password = str(form.get("password", ""))
    if password != DEMO_PASSWORD:
        return RedirectResponse("/login?error=1", status_code=303)
    lower = username.lower()
    if lower == "coach01":
        request.session.update({"username": "coach01", "role": "coach"})
        return RedirectResponse("/coach", status_code=303)
    if lower == "scientist01":
        request.session.update({"username": "scientist01", "role": "scientist"})
        return RedirectResponse("/scientist", status_code=303)
    ensure_processed()
    mapping = pd.read_csv(MAPPING_FILE, dtype=str) if MAPPING_FILE.exists() else pd.DataFrame()
    ids = set(mapping.athlete_id.astype(str).str.upper()) if not mapping.empty else set()
    aid = username.upper()
    if aid in ids:
        request.session.update({"username": aid, "role": "athlete", "athlete_id": aid})
        return RedirectResponse(f"/athlete/{quote(aid)}", status_code=303)
    return RedirectResponse("/login?error=1", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/")
def home(request: Request):
    role = user_role(request)
    if not role:
        return RedirectResponse("/login", status_code=303)
    if role == "coach":
        return RedirectResponse("/coach", status_code=303)
    if role == "scientist":
        return RedirectResponse("/scientist", status_code=303)
    return RedirectResponse(f"/athlete/{quote(request.session.get('athlete_id',''))}", status_code=303)


@app.get("/coach", response_class=HTMLResponse)
def coach_dashboard(request: Request, athlete_a: str = "", athlete_b: str = ""):
    redirect = require_role(request, "coach")
    if redirect:
        return redirect

    df = load_metrics()

    perf = df[df["quality_status"].astype(str).str.upper() == "VALID"].copy()
    if perf.empty:
        perf = df.copy()

    athletes = sorted(df["athlete_id"].dropna().astype(str).unique().tolist())

    athlete_a = (
        athlete_a
        if athlete_a in athletes
        else (athletes[0] if athletes else "")
    )

    athlete_b = (
        athlete_b
        if athlete_b in athletes
        else (
            athletes[1]
            if len(athletes) > 1
            else athlete_a
        )
    )

    summary_rows = []

    for aid, g_all in df.groupby("athlete_id"):
        g = g_all[
            g_all["quality_status"].astype(str).str.upper() == "VALID"
        ].copy()

        if g.empty:
            g = g_all.copy()

        jump_values = pd.to_numeric(
            g["jump_height_cm"],
            errors="coerce",
        )

        variability = 0.0
        if jump_values.notna().sum() > 1:
            mean_jump = jump_values.mean()
            if np.isfinite(mean_jump) and abs(mean_jump) > 1e-9:
                variability = (
                    jump_values.std(ddof=1)
                    / abs(mean_jump)
                    * 100
                )

        review_count = int(
            (
                g_all["quality_status"]
                .astype(str)
                .str.upper()
                == "REVIEW"
            ).sum()
        )

        summary_rows.append(
            {
                "athlete_id": str(aid),
                "display_name": (
                    g["display_name"].iloc[0]
                    if "display_name" in g.columns
                    else str(aid)
                ),
                "jumps": len(g),
                "mass": median(g, "body_mass_kg"),
                "jump": median(g, "jump_height_cm"),
                "rsi": median(g, "rsi_mod"),
                "peak_force": median(g, "peak_force_N"),
                "rel_force": median(g, "relative_peak_force_N_kg"),
                "impulse": median(g, "relative_impulse_Ns_kg"),
                "velocity": median(g, "takeoff_velocity_ms"),
                "power": median(g, "peak_power_W"),
                "rel_power": median(g, "relative_peak_power_W_kg"),
                "variability": variability,
                "review": review_count,
            }
        )

    s = pd.DataFrame(summary_rows)

    # --------------------------------------------------------
    # Coach-facing squad KPIs
    # --------------------------------------------------------
    squad_avg_jump = pd.to_numeric(
        perf.get("jump_height_cm"),
        errors="coerce",
    ).mean()

    squad_avg_rsi = pd.to_numeric(
        perf.get("rsi_mod"),
        errors="coerce",
    ).mean()

    squad_avg_rel_force = pd.to_numeric(
        perf.get("relative_peak_force_N_kg"),
        errors="coerce",
    ).mean()

    athletes_requiring_review = 0
    if not df.empty and "quality_status" in df.columns:
        review_df = df[
            df["quality_status"]
            .astype(str)
            .str.upper()
            == "REVIEW"
        ]
        athletes_requiring_review = int(
            review_df["athlete_id"].nunique()
        )

    opts_a = "".join(
        f'<option {"selected" if a == athlete_a else ""}>{escape(a)}</option>'
        for a in athletes
    )

    opts_b = "".join(
        f'<option {"selected" if a == athlete_b else ""}>{escape(a)}</option>'
        for a in athletes
    )

    rows = "".join(
        f"""
        <tr
            class="clickable"
            onclick="location.href='/athlete/{quote(r.athlete_id)}'"
        >
            <td class="strong">{escape(r.athlete_id)}</td>
            <td>{fmt(r.mass,1," kg")}</td>
            <td>{fmt(r.jump,1," cm")}</td>
            <td>{fmt(r.rsi,2)}</td>
            <td>{fmt(r.peak_force,0," N")}</td>
            <td>{fmt(r.rel_force,2," N/kg")}</td>
            <td>{fmt(r.impulse,2," N·s/kg")}</td>
            <td>{fmt(r.velocity,2," m/s")}</td>
            <td>{fmt(r.power,0," W")}</td>
            <td>{fmt(r.rel_power,1," W/kg")}</td>
            <td>{fmt(r.variability,1," %")}</td>
            <td>
                <span class="status {"review" if r.review else ""}">
                    {"Review" if r.review else "Valid"}
                </span>
            </td>
        </tr>
        """
        for r in s.itertuples()
    )

    form = f"""
    <form class="filters" method="get">

        <div class="filter">
            <label>Athlete A</label>
            <select name="athlete_a">
                {opts_a}
            </select>
        </div>

        <div class="filter">
            <label>Athlete B</label>
            <select name="athlete_b">
                {opts_b}
            </select>
        </div>

        <button class="btn" type="submit">
            Compare
        </button>

    </form>
    """

    body = f"""
    <div class="kpis">

        <div class="kpi">
            <div class="kpi-label">
                Squad Average Jump Height
            </div>
            <div class="kpi-value">
                {fmt(squad_avg_jump,1,' cm')}
            </div>
            <div class="kpi-foot">
                Current squad performance
            </div>
        </div>

        <div class="kpi">
            <div class="kpi-label">
                Squad Average RSI-mod
            </div>
            <div class="kpi-value">
                {fmt(squad_avg_rsi,2)}
            </div>
            <div class="kpi-foot">
                Average reactive performance
            </div>
        </div>

        <div class="kpi">
            <div class="kpi-label">
                Average Relative Peak Force
            </div>
            <div class="kpi-value">
                {fmt(squad_avg_rel_force,2,' N/kg')}
            </div>
            <div class="kpi-foot">
                Body-mass normalised comparison
            </div>
        </div>

        <div class="kpi">
            <div class="kpi-label">
                Athletes Requiring Review
            </div>
            <div class="kpi-value">
                {athletes_requiring_review}
            </div>
            <div class="kpi-foot">
                Based on current QC status
            </div>
        </div>

    </div>


    <div class="panel">

        <div class="panel-head">
            <div>
                <div class="panel-title">
                    Athlete Comparison Quadrant
                </div>
                <div class="panel-sub">
                    RSI-mod vs Jump Height · compare two athletes against the squad
                </div>
            </div>
        </div>

        {form}

        <div class="panel-body">
            {quadrant_chart(
                perf,
                athlete_a,
                athlete_b,
            )}
        </div>

    </div>


    <div class="section">
        <div class="section-title">
            Squad Performance
        </div>
        <div class="section-note">
            Practical performance, normalised metrics and quality status
        </div>
    </div>


    <div class="panel">
        <div class="table-wrap">

            <table>

                <thead>
                    <tr>
                        <th>Athlete</th>
                        <th>Mass</th>
                        <th>Jump Height</th>
                        <th>RSI</th>
                        <th>Peak Force</th>
                        <th>Rel. Peak Force</th>
                        <th>Rel. Impulse</th>
                        <th>Take-off Vel.</th>
                        <th>Peak Power</th>
                        <th>Rel. Power</th>
                        <th>Variability</th>
                        <th>Status</th>
                    </tr>
                </thead>

                <tbody>
                    {rows}
                </tbody>

            </table>

        </div>
    </div>
    """

    return page(
        "Coach Dashboard",
        "Squad performance · comparison · normalised metrics",
        body,
        "coach",
        request,
    )



@app.get("/athletes", response_class=HTMLResponse)
def athletes_page(request: Request):
    redirect = require_role(
        request,
        "coach",
        "scientist",
    )

    if redirect:
        return redirect

    df = load_metrics()

    # --------------------------------------------------------
    # Top performers
    # --------------------------------------------------------
    best_jump_athlete, best_jump = get_top_performer(
        df,
        "jump_height_cm",
    )

    best_rsi_athlete, best_rsi = get_top_performer(
        df,
        "rsi_mod",
    )

    best_force_athlete, best_force = get_top_performer(
        df,
        "relative_peak_force_N_kg",
    )

    best_power_athlete, best_power = get_top_performer(
        df,
        "relative_peak_power_W_kg",
    )

    def performer_card(
        title,
        athlete_id,
        value,
        decimals,
        suffix,
    ):
        if athlete_id:
            link = (
                f"/scientist/{quote(athlete_id)}"
                if user_role(request) == "scientist"
                else f"/athlete/{quote(athlete_id)}"
            )
            athlete_text = escape(str(athlete_id))
            open_text = "Open athlete →"
        else:
            link = "#"
            athlete_text = "Unavailable"
            open_text = ""

        return f"""
        <a class="kpi" href="{link}" style="display:block;">

            <div class="kpi-label">
                {escape(title)}
            </div>

            <div class="kpi-value">
                {fmt(value, decimals, suffix)}
            </div>

            <div class="kpi-foot">
                <span style="color:#dfe8f4;font-weight:800;">
                    {athlete_text}
                </span>
                {" · " + open_text if open_text else ""}
            </div>

        </a>
        """

    top_performers_html = f"""
    <div class="section">

        <div class="section-title">
            Top Performers
        </div>

        <div class="section-note">
            Quick view of the leading athlete for each key performance metric
        </div>

    </div>


    <div class="kpis">

        {performer_card(
            "Best Jump Height",
            best_jump_athlete,
            best_jump,
            1,
            " cm",
        )}

        {performer_card(
            "Best RSI-mod",
            best_rsi_athlete,
            best_rsi,
            2,
            "",
        )}

        {performer_card(
            "Highest Relative Peak Force",
            best_force_athlete,
            best_force,
            2,
            " N/kg",
        )}

        {performer_card(
            "Highest Relative Power",
            best_power_athlete,
            best_power,
            1,
            " W/kg",
        )}

    </div>
    """

    # --------------------------------------------------------
    # Athlete cards
    # --------------------------------------------------------
    cards = []

    for aid, g_all in df.groupby("athlete_id"):

        g = g_all[
            g_all["quality_status"]
            .astype(str)
            .str.upper()
            == "VALID"
        ].copy()

        if g.empty:
            g = g_all.copy()

        link = (
            f"/scientist/{quote(str(aid))}"
            if user_role(request) == "scientist"
            else f"/athlete/{quote(str(aid))}"
        )

        cards.append(
            f"""
            <a
                class="athlete-card"
                href="{link}"
            >

                <div class="athlete-name">
                    {escape(str(aid))}
                </div>

                <div class="athlete-meta">
                    {len(g)} jumps
                </div>


                <div class="mini-grid">

                    <div class="mini">
                        <span>Jump Height</span>
                        <strong>
                            {fmt(
                                median(
                                    g,
                                    "jump_height_cm",
                                ),
                                1,
                                " cm",
                            )}
                        </strong>
                    </div>

                    <div class="mini">
                        <span>RSI-mod</span>
                        <strong>
                            {fmt(
                                median(
                                    g,
                                    "rsi_mod",
                                ),
                                2,
                            )}
                        </strong>
                    </div>

                    <div class="mini">
                        <span>Peak Force</span>
                        <strong>
                            {fmt(
                                median(
                                    g,
                                    "peak_force_N",
                                ),
                                0,
                                " N",
                            )}
                        </strong>
                    </div>

                    <div class="mini">
                        <span>Peak Power</span>
                        <strong>
                            {fmt(
                                median(
                                    g,
                                    "peak_power_W",
                                ),
                                0,
                                " W",
                            )}
                        </strong>
                    </div>

                </div>


                <div
                    style="
                        margin-top:14px;
                        padding-top:10px;
                        border-top:1px solid #17283e;
                        color:#22d3ee;
                        font-size:9px;
                        font-weight:850;
                        text-align:right;
                    "
                >
                    View Athlete →
                </div>

            </a>
            """
        )

    body = f"""

    {top_performers_html}


    <div class="section">

        <div class="section-title">
            Participants / Athletes
        </div>

        <div class="section-note">
            Select an athlete to view individual performance
        </div>

    </div>


    <div class="athlete-grid">
        {"".join(cards)}
    </div>
    """

    return page(
        "Athletes",
        "Select an athlete or review the current top performers",
        body,
        "athletes",
        request,
    )


@app.get("/athlete/{athlete_id}", response_class=HTMLResponse)
def athlete_page(request: Request, athlete_id: str, metric: str="jump_height"):
    if not user_role(request): return RedirectResponse("/login",status_code=303)
    athlete_id=athlete_id.strip().upper()
    if not athlete_allowed(request,athlete_id): raise HTTPException(403,"Athlete accounts can only view their own performance.")
    df=load_metrics(); d_all=df[df.athlete_id==athlete_id].copy()
    if d_all.empty: raise HTTPException(404,"Athlete not found")
    d=d_all[d_all.quality_status == "VALID"].copy()
    if d.empty:
        d=d_all.copy()
    metric_map={"jump_height":("jump_height_cm","Jump Height","cm",1),"rsi":("rsi_mod","RSI Modified","",2),"velocity":("takeoff_velocity_ms","Take-off Velocity","m/s",2),"force":("peak_force_N","Peak Force","N",0),"power":("peak_power_W","Peak Power","W",0)}
    col,label,unit,dec=metric_map.get(metric,metric_map["jump_height"])
    tabs="".join(f'<a class="tab {"active" if metric==k else ""}" href="/athlete/{quote(athlete_id)}?metric={k}">{v[1]}</a>' for k,v in metric_map.items())
    vals=pd.to_numeric(d.jump_height_cm,errors="coerce")
    latest=d.sort_values(["session_date","test_id","jump_id"]).iloc[-1]
    rows="".join(f'<tr><td>{escape(str(r.test_id))}</td><td>J{int(r.jump_id)}</td><td>{fmt(r.jump_height_cm,1," cm")}</td><td>{fmt(r.rsi_mod,2)}</td><td>{fmt(r.takeoff_velocity_ms,2," m/s")}</td><td>{fmt(r.peak_force_N,0," N")}</td><td>{fmt(r.peak_power_W,0," W")}</td></tr>' for r in d.sort_values(["test_id","jump_id"]).itertuples())
    advanced = f'<a class="btn" href="/scientist/{quote(athlete_id)}">Advanced Analysis →</a>' if user_role(request) in {"coach","scientist"} else ''
    body=f"""<div class="breadcrumb"><a href="/">Home</a> › {escape(athlete_id)}</div><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px"><div><h2 style="margin:0">{escape(athlete_id)}</h2><div class="subtitle">Simple athlete performance feedback</div></div>{advanced}</div>
    <div class="kpis"><div class="kpi"><div class="kpi-label">Latest Jump Height</div><div class="kpi-value">{fmt(latest.jump_height_cm,1,' cm')}</div></div><div class="kpi"><div class="kpi-label">Latest RSI-mod</div><div class="kpi-value">{fmt(latest.rsi_mod,2)}</div></div><div class="kpi"><div class="kpi-label">Best Jump</div><div class="kpi-value">{fmt(vals.max(),1,' cm')}</div></div><div class="kpi"><div class="kpi-label">Average Jump</div><div class="kpi-value">{fmt(vals.mean(),1,' cm')}</div></div></div>
    <div class="panel"><div class="panel-head"><div><div class="panel-title">Performance History</div><div class="panel-sub">Individual jump progress</div></div><div class="tabs">{tabs}</div></div><div class="panel-body">{chart_history(d,col,label,unit,dec)}</div></div>
    <div class="section"><div class="section-title">Recent Jumps</div></div><div class="panel"><div class="table-wrap"><table><thead><tr><th>Test</th><th>Jump</th><th>Jump Height</th><th>RSI</th><th>Take-off Velocity</th><th>Peak Force</th><th>Peak Power</th></tr></thead><tbody>{rows}</tbody></table></div></div>"""
    return page(f"Athlete · {athlete_id}", "Performance feedback", body, "athlete" if user_role(request)=="athlete" else "athletes", request)


@app.get("/scientist", response_class=HTMLResponse)
def scientist_home(request: Request):
    redirect = require_role(request, "scientist")
    if redirect:
        return redirect

    df = load_metrics()
    align = load_alignment()

    if df.empty:
        return page(
            "Sport Scientist",
            "Biomechanics · signals · alignment · quality control",
            '<div class="empty"><p>No processed jump data are currently available.</p></div>',
            "scientist",
            request,
        )

    valid_df = df[
        df["quality_status"].astype(str).str.upper() == "VALID"
    ].copy()

    review_df = df[
        df["quality_status"].astype(str).str.upper() == "REVIEW"
    ].copy()

    performance_df = valid_df if not valid_df.empty else df.copy()

    processed_tests = int(df["test_id"].nunique()) if "test_id" in df.columns else 0
    detected_jumps = int(len(df))
    valid_jumps = int(len(valid_df))
    review_jumps = int(len(review_df))

    athlete_count = int(df["athlete_id"].nunique()) if "athlete_id" in df.columns else 0
    athletes_with_review = (
        int(review_df["athlete_id"].nunique())
        if not review_df.empty and "athlete_id" in review_df.columns
        else 0
    )

    review_rate = (
        review_jumps / detected_jumps * 100
        if detected_jumps
        else np.nan
    )

    mean_corr = (
        pd.to_numeric(
            align["alignment_corr"],
            errors="coerce",
        ).mean()
        if not align.empty and "alignment_corr" in align.columns
        else pd.to_numeric(
            df.get("alignment_corr"),
            errors="coerce",
        ).mean()
    )

    mean_shift = pd.to_numeric(
        df.get("alignment_shift_s"),
        errors="coerce",
    ).abs().mean()

    avg_peak_force = pd.to_numeric(
        performance_df.get("peak_force_N"),
        errors="coerce",
    ).mean()

    avg_rel_peak_force = pd.to_numeric(
        performance_df.get("relative_peak_force_N_kg"),
        errors="coerce",
    ).mean()

    avg_takeoff_velocity = pd.to_numeric(
        performance_df.get("takeoff_velocity_ms"),
        errors="coerce",
    ).mean()

    avg_peak_power = pd.to_numeric(
        performance_df.get("peak_power_W"),
        errors="coerce",
    ).mean()

    avg_rel_power = pd.to_numeric(
        performance_df.get("relative_peak_power_W_kg"),
        errors="coerce",
    ).mean()

    avg_rel_impulse = pd.to_numeric(
        performance_df.get("relative_impulse_Ns_kg"),
        errors="coerce",
    ).mean()

    # --------------------------------------------------------
    # Athlete selector
    # --------------------------------------------------------
    athlete_options = "".join(
        f'<option value="{quote(str(aid))}">{escape(str(aid))}</option>'
        for aid in sorted(
            df["athlete_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )

    athlete_selector = f"""
    <div class="panel">

        <div class="panel-head">
            <div>
                <div class="panel-title">
                    Athlete Analysis
                </div>
                <div class="panel-sub">
                    Open advanced biomechanical and signal analysis for an athlete
                </div>
            </div>
        </div>

        <div class="filters">

            <div class="filter" style="min-width:280px;">
                <label>Select Athlete</label>

                <select
                    id="scientist-athlete-select"
                    onchange="
                        if(this.value) {{
                            window.location.href='/scientist/' + this.value;
                        }}
                    "
                >
                    <option value="">
                        Choose an athlete...
                    </option>

                    {athlete_options}

                </select>
            </div>

            <a class="btn" href="/athletes">
                Browse All Athletes
            </a>

        </div>

    </div>
    """

    # --------------------------------------------------------
    # Quality flag summary
    # --------------------------------------------------------
    flag_counts = {}

    if "quality_flags" in df.columns:
        for value in df["quality_flags"].fillna("OK").astype(str):
            flags = [
                f.strip()
                for f in value.split("|")
                if f.strip() and f.strip().upper() != "OK"
            ]

            for flag in flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

    if flag_counts:
        flag_rows = "".join(
            f"""
            <tr>
                <td>{escape(flag)}</td>
                <td class="strong">{count}</td>
            </tr>
            """
            for flag, count in sorted(
                flag_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:8]
        )
    else:
        flag_rows = """
        <tr>
            <td>No active quality flags</td>
            <td class="strong">0</td>
        </tr>
        """

    quality_table = f"""
    <div class="panel">

        <div class="panel-head">
            <div>
                <div class="panel-title">
                    Quality Control Summary
                </div>
                <div class="panel-sub">
                    Current signal, alignment and processing checks
                </div>
            </div>
        </div>

        <div class="panel-body">

            <div class="metric-grid">

                <div class="metric-card">
                    <div class="label">
                        Review Rate
                    </div>
                    <div class="value">
                        {fmt(review_rate,1," %")}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="label">
                        Athletes With Review Flags
                    </div>
                    <div class="value">
                        {athletes_with_review}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="label">
                        Mean Alignment Correlation
                    </div>
                    <div class="value">
                        {fmt(mean_corr,2)}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="label">
                        Mean |Alignment Shift|
                    </div>
                    <div class="value">
                        {fmt(mean_shift,3," s")}
                    </div>
                </div>

            </div>

            <div style="height:16px"></div>

            <div class="table-wrap">

                <table style="min-width:0;">

                    <thead>
                        <tr>
                            <th>Quality Flag</th>
                            <th>Occurrences</th>
                        </tr>
                    </thead>

                    <tbody>
                        {flag_rows}
                    </tbody>

                </table>

            </div>

        </div>

    </div>
    """

    # --------------------------------------------------------
    # Scientist overview charts
    # --------------------------------------------------------
    scatter_df = performance_df.copy()

    scatter_df["rsi_mod"] = pd.to_numeric(
        scatter_df.get("rsi_mod"),
        errors="coerce",
    )

    scatter_df["jump_height_cm"] = pd.to_numeric(
        scatter_df.get("jump_height_cm"),
        errors="coerce",
    )

    scatter_df = scatter_df.dropna(
        subset=["rsi_mod", "jump_height_cm"]
    )

    performance_fig = go.Figure()

    if not scatter_df.empty:
        performance_fig.add_trace(
            go.Scatter(
                x=scatter_df["rsi_mod"],
                y=scatter_df["jump_height_cm"],
                mode="markers",
                marker=dict(
                    color="#22d3ee",
                    size=8,
                    opacity=0.60,
                ),
                text=scatter_df["athlete_id"].astype(str),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "RSI-mod: %{x:.2f}<br>"
                    "Jump Height: %{y:.1f} cm"
                    "<extra></extra>"
                ),
                name="Valid Jumps",
            )
        )

    performance_fig.update_layout(
        title=None,
        xaxis_title="RSI Modified",
        yaxis_title="Jump Height (cm)",
        showlegend=False,
    )

    performance_chart = plot_html(
        performance_fig,
        height=340,
    )

    alignment_fig = go.Figure()

    align_plot = df.copy()

    if "alignment_corr" in align_plot.columns:
        align_plot["alignment_corr"] = pd.to_numeric(
            align_plot["alignment_corr"],
            errors="coerce",
        )

        align_plot = align_plot.dropna(
            subset=["alignment_corr"]
        )

        if not align_plot.empty:
            align_plot = align_plot.reset_index(drop=True)

            alignment_fig.add_trace(
                go.Scatter(
                    x=np.arange(1, len(align_plot) + 1),
                    y=align_plot["alignment_corr"],
                    mode="markers",
                    marker=dict(
                        color="#705cff",
                        size=7,
                        opacity=0.70,
                    ),
                    text=align_plot["athlete_id"].astype(str),
                    customdata=align_plot["test_id"].astype(str),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Test: %{customdata}<br>"
                        "Alignment: %{y:.2f}"
                        "<extra></extra>"
                    ),
                    name="Alignment",
                )
            )

    alignment_fig.update_layout(
        title=None,
        xaxis_title="Processed Jump",
        yaxis_title="Alignment Correlation",
        showlegend=False,
    )

    alignment_chart = plot_html(
        alignment_fig,
        height=340,
    )

    body = f"""

    <div class="kpis">

        <div class="kpi">
            <div class="kpi-label">
                Processed Tests
            </div>
            <div class="kpi-value">
                {processed_tests}
            </div>
            <div class="kpi-foot">
                ForceDeck + IMU test files
            </div>
        </div>

        <div class="kpi">
            <div class="kpi-label">
                Detected Jumps
            </div>
            <div class="kpi-value">
                {detected_jumps}
            </div>
            <div class="kpi-foot">
                Detected CMJ events
            </div>
        </div>

        <div class="kpi">
            <div class="kpi-label">
                Valid Jumps
            </div>
            <div class="kpi-value">
                {valid_jumps}
            </div>
            <div class="kpi-foot">
                Passed current QC checks
            </div>
        </div>

        <div class="kpi">
            <div class="kpi-label">
                Review Jumps
            </div>
            <div class="kpi-value">
                {review_jumps}
            </div>
            <div class="kpi-foot">
                Require scientist review
            </div>
        </div>

    </div>


    <div class="section">
        <div class="section-title">
            Advanced Performance Summary
        </div>
        <div class="section-note">
            Squad-level summary using valid jumps
        </div>
    </div>


    <div class="metric-grid">

        <div class="metric-card">
            <div class="label">
                Average Peak Force
            </div>
            <div class="value">
                {fmt(avg_peak_force,0," N")}
            </div>
        </div>

        <div class="metric-card">
            <div class="label">
                Average Relative Peak Force
            </div>
            <div class="value">
                {fmt(avg_rel_peak_force,2," N/kg")}
            </div>
        </div>

        <div class="metric-card">
            <div class="label">
                Average Take-off Velocity
            </div>
            <div class="value">
                {fmt(avg_takeoff_velocity,2," m/s")}
            </div>
        </div>

        <div class="metric-card">
            <div class="label">
                Average Peak Power
            </div>
            <div class="value">
                {fmt(avg_peak_power,0," W")}
            </div>
        </div>

        <div class="metric-card">
            <div class="label">
                Average Relative Power
            </div>
            <div class="value">
                {fmt(avg_rel_power,1," W/kg")}
            </div>
        </div>

        <div class="metric-card">
            <div class="label">
                Average Relative Impulse
            </div>
            <div class="value">
                {fmt(avg_rel_impulse,2," N·s/kg")}
            </div>
        </div>

        <div class="metric-card">
            <div class="label">
                Athletes Analysed
            </div>
            <div class="value">
                {athlete_count}
            </div>
        </div>

        <div class="metric-card">
            <div class="label">
                Mean IMU ↔ Force Alignment
            </div>
            <div class="value">
                {fmt(mean_corr,2)}
            </div>
        </div>

    </div>


    <div style="height:20px"></div>

    {athlete_selector}


    <div class="section">
        <div class="section-title">
            Squad Biomechanics Overview
        </div>
        <div class="section-note">
            Valid jump performance and alignment distributions
        </div>
    </div>


    <div class="grid-2">

        <div class="panel">

            <div class="panel-head">
                <div>
                    <div class="panel-title">
                        RSI-mod vs Jump Height
                    </div>
                    <div class="panel-sub">
                        Valid jumps across the squad
                    </div>
                </div>
            </div>

            <div class="panel-body">
                {performance_chart}
            </div>

        </div>


        <div class="panel">

            <div class="panel-head">
                <div>
                    <div class="panel-title">
                        IMU ↔ Force Alignment
                    </div>
                    <div class="panel-sub">
                        Cross-correlation quality across processed jumps
                    </div>
                </div>
            </div>

            <div class="panel-body">
                {alignment_chart}
            </div>

        </div>

    </div>


    <div style="height:20px"></div>

    {quality_table}


    <div class="section">
        <div class="section-title">
            Individual Signal Analysis
        </div>
        <div class="section-note">
            After selecting an athlete, inspect Force-Time, IMU Acceleration-Time,
            Velocity-Time and Power-Time curves with phase markers, advanced metrics,
            jump strategy, QC flags and alignment information.
        </div>
    </div>


    <div class="note">
        Use the Athlete Analysis selector above to open the detailed scientist view.
        The detailed page includes movement onset, minimum-velocity/braking transition,
        take-off and landing markers together with force, acceleration, velocity,
        power and phase-specific metrics.
    </div>

    """

    return page(
        "Sport Scientist",
        "Biomechanics · signals · alignment · quality control",
        body,
        "scientist",
        request,
    )


@app.get("/scientist/{athlete_id}", response_class=HTMLResponse)
def scientist_athlete(request: Request, athlete_id: str, jump: int|None=None):
    redirect=require_role(request,"scientist")
    if redirect:return redirect
    athlete_id=athlete_id.strip().upper(); df=load_metrics(); d=df[df.athlete_id==athlete_id].sort_values(["test_id","jump_id"])
    if d.empty:raise HTTPException(404,"Athlete not found")
    if jump is None or jump not in set(d.jump_id.astype(int)):
        selected=d.iloc[0]
    else:
        selected=d[d.jump_id.astype(int)==jump].iloc[0]
    sig=load_signal(selected.jump_key)
    # Signal time is converted to time-from-onset for all graphs.
    events=[("Onset",0.0),("Min velocity",selected.braking_end_s-selected.onset_s),("Take-off",selected.takeoff_s-selected.onset_s),("Landing",selected.landing_s-selected.onset_s)]
    jump_opts="".join(f'<option value="{int(r.jump_id)}" {"selected" if int(r.jump_id)==int(selected.jump_id) else ""}>Jump {int(r.jump_id)}</option>' for r in d.itertuples())
    filters=f'<form class="filters" method="get"><div class="filter"><label>Jump</label><select name="jump">{jump_opts}</select></div><button class="btn" type="submit">Load Jump</button></form>'
    metrics=[("Body Mass",selected.body_mass_kg,1," kg"),("Jump Height",selected.jump_height_cm,1," cm"),("RSI Modified",selected.rsi_mod,2,""),("Peak Force",selected.peak_force_N,0," N"),("Mean Force",selected.mean_force_N,0," N"),("Relative Peak Force",selected.relative_peak_force_N_kg,2," N/kg"),("Braking Impulse",abs(selected.braking_impulse_Ns),2," N·s"),("Propulsive Impulse",selected.propulsive_impulse_Ns,2," N·s"),("Net Impulse",selected.total_net_impulse_Ns,2," N·s"),("Relative Impulse",selected.relative_impulse_Ns_kg,2," N·s/kg"),("Peak Acceleration",selected.imu_peak_acceleration_ms2,2," m/s²"),("Mean Acceleration",selected.imu_mean_acceleration_ms2,2," m/s²"),("Peak Velocity",selected.peak_velocity_ms,2," m/s"),("Take-off Velocity",selected.takeoff_velocity_ms,2," m/s"),("Peak Power",selected.peak_power_W,0," W"),("Relative Power",selected.relative_peak_power_W_kg,1," W/kg"),("Eccentric Duration",selected.eccentric_duration_s,3," s"),("Concentric Duration",selected.concentric_duration_s,3," s")]
    cards="".join(f'<div class="metric-card"><div class="label">{escape(n)}</div><div class="value">{fmt(v,dg,suf)}</div></div>' for n,v,dg,suf in metrics)
    force=signal_chart(sig,"time_from_onset_s","force_N","Force-Time Curve","Force (N)",events)
    acc=signal_chart(sig,"time_from_onset_s","imu_acceleration_ms2","IMU Acceleration-Time Curve","Acceleration (m/s²)",events)
    vel=signal_chart(sig,"time_from_onset_s","velocity_ms","Velocity-Time Curve","Velocity (m/s)",events)
    power=signal_chart(sig,"time_from_onset_s","power_W","Power-Time Curve","Power (W)",events)
    strategy="Longer eccentric strategy – the athlete uses a longer countermovement period before the propulsive phase." if selected.eccentric_duration_s>selected.concentric_duration_s*1.15 else ("Shorter eccentric strategy – the athlete moves relatively quickly through the eccentric phase before propulsion." if selected.eccentric_duration_s<selected.concentric_duration_s*.85 else "Balanced phase-duration strategy – eccentric and concentric durations are relatively similar.")
    body=f"""<div class="breadcrumb"><a href="/scientist">Sport Scientist</a> › <a href="/athletes">Athletes</a> › {escape(athlete_id)}</div><div class="panel">{filters}</div><div class="section"><div class="section-title">Advanced Metrics</div><div class="section-note">Jump {int(selected.jump_id)} · {escape(selected.test_id)}</div></div><div class="metric-grid">{cards}</div><div class="section"><div class="section-title">Signal Curves & Phase Markers</div><div class="section-note">Real ForceDeck + quaternion-oriented IMU data</div></div><div class="panel"><div class="panel-body">{force}</div></div><div style="height:14px"></div><div class="grid-2"><div class="panel"><div class="panel-body">{acc}</div></div><div class="panel"><div class="panel-body">{vel}</div></div></div><div style="height:14px"></div><div class="panel"><div class="panel-body">{power}</div></div><div class="section"><div class="section-title">Jump Strategy & Quality</div></div><div class="grid-2"><div class="panel"><div class="panel-body"><div class="note">{escape(strategy)} This is a movement strategy description, not a good/bad classification.</div></div></div><div class="panel"><div class="panel-body"><div class="metric-card"><div class="label">Quality Status</div><div class="value">{escape(selected.quality_status)}</div></div><div style="height:8px"></div><div class="note">Flags: {escape(selected.quality_flags)}<br>IMU ↔ Force shift: {fmt(selected.alignment_shift_s,2,' s')}<br>Alignment correlation: {fmt(selected.alignment_corr,2)}</div></div></div></div>"""
    return page(f"Sport Scientist · {athlete_id}", "Advanced biomechanical and signal analysis", body, "scientist", request)


@app.get("/health")
def health():
    try: data_dir=str(find_data_dir())
    except Exception as e: data_dir=str(e)
    return JSONResponse({"status":"ok","data_dir":data_dir,"processed":JUMP_METRICS_FILE.exists(),"signals_dir":str(SIGNALS_DIR)})


if __name__ == "__main__":
    import uvicorn
    print("KIVO Phase 2 starting...")
    print("RUNNING FILE:", __file__)
    print("Login: coach01 / kivo123 | scientist01 / kivo123 | ATH001 / kivo123")
    uvicorn.run(app, host="127.0.0.1", port=8000)
