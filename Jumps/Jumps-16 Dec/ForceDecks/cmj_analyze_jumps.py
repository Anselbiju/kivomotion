"""
cmj_analyze_jumps.py
====================

Standalone CMJ (counter-movement jump) phase-detection and impulse analysis
for VALD ForceDecks raw-force CSV exports.

USAGE
-----
1. Put this script in a folder that contains one or more ForceDecks raw CSV
   exports named like:   FD_<ATHLETEID>_<anything>.csv
   (e.g. FD_AK_16_12.csv, FD_CL_16_12_02.csv, FD_AM_03_07.csv ...).

2. Run:
       python cmj_analyze_jumps.py

   Optional command-line arguments:
       --input-dir   PATH    folder to scan (default: same folder as this script)
       --output-dir  PATH    where to write outputs (default: ./output)
       --no-plots            skip per-file PNG generation (CSV only)
       --pattern     GLOB    filename glob to match (default: FD_*.csv)

3. Outputs (in ./output by default):
       all_athlete_force_jump_events_impulse.csv
       FD_<ID>_..._events.png        (one per input file)

WHAT IT COMPUTES
----------------
Each detected CMJ in each file gets a row with:

    athlete_id, trial_id, jump_id, body_mass_kg,
    onset_s, braking_end_s, takeoff_s, landing_s, flight_time_s,
    braking_impulse_Ns, propulsive_impulse_Ns, total_impulse_Ns,
    fd_file

Phase definitions (4-event CMJ model):

    onset_s        : start of unweighting / onset of contraction
                     = first sustained sample where F < BW - max(5*SD_quiet, 20 N)
    braking_end_s  : end of unweighting (= velocity minimum)
                     = first sample after the dip where CoM velocity is
                       at its most negative; equivalently F returns to BW
                       going upward
    takeoff_s      : feet leave the plate
                     = first sample where F < 20 N going down
    landing_s      : feet touch the plate after flight
                     = first sample where F > 20 N coming back up

Impulses are time-integrals of (F - BW):
    braking_impulse_Ns    = |integral from onset to braking_end|  (positive)
    propulsive_impulse_Ns = integral from braking_end to takeoff
    total_impulse_Ns      = net impulse from onset to takeoff
                            (= m * v_takeoff; determines jump height)

ASSUMPTIONS
-----------
* Raw ForceDecks export format: 8 metadata lines, blank line, then
  "Time,Left,Right,Both" header on line 10, then samples.
* Sampling rate is read from the "Frequency" metadata field (commonly 500 Hz).
* Each file contains a single trial with one or more CMJs separated by
  brief standing periods.
* The athlete is standing still on the plate at bodyweight before each jump.

DEPENDENCIES
------------
    numpy, pandas, matplotlib
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Tunable parameters (sensible defaults; rarely need changing)
# ---------------------------------------------------------------------------
G                       = 9.80665
QUIET_WINDOW_S          = 1.0      # length of the BW estimation window
THR_FORCE_N             = 20.0     # min absolute threshold (N) for onset / takeoff / landing
THR_SD_MULT             = 5.0      # onset threshold = max(THR_SD_MULT * SD_quiet, THR_FORCE_N)
TAKEOFF_THR_N           = 20.0     # F < this -> in flight
MIN_FLIGHT_S            = 0.10     # ignore flight gaps shorter than this
STABLE_WIN_S            = 0.30     # length of the pre-jump "standing" window
STABLE_MEAN_TOL_N       = 50.0     # window mean must be within this of BW
STABLE_SD_MAX_N         = 25.0     # window SD must be under this
ONSET_SUSTAIN_S         = 0.020    # onset deviation must persist this long


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class JumpEvents:
    jump_id: int
    body_mass_kg: float
    onset_s: float
    braking_end_s: float
    takeoff_s: float
    landing_s: float
    flight_time_s: float
    braking_impulse_Ns: float
    propulsive_impulse_Ns: float
    total_impulse_Ns: float


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------
def read_fd_file(path: str) -> Tuple[pd.DataFrame, float, dict]:
    """Read a ForceDecks raw CSV. Returns (df, sampling_freq_hz, metadata)."""
    meta: dict = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for _ in range(9):
            line = f.readline().strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2 and parts[0]:
                meta[parts[0]] = parts[1]

    df = pd.read_csv(path, skiprows=9)
    df.columns = [c.strip() for c in df.columns]
    if "Time" not in df.columns or "Both" not in df.columns:
        raise ValueError(
            f"{os.path.basename(path)}: expected 'Time' and 'Both' columns "
            f"on line 10; got {list(df.columns)}"
        )
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Both"] = pd.to_numeric(df["Both"], errors="coerce")
    df = df.dropna(subset=["Time", "Both"]).reset_index(drop=True)

    freq = float(meta.get("Frequency", 500))
    if freq <= 0:
        raise ValueError(f"{os.path.basename(path)}: invalid frequency {freq}")
    return df, freq, meta


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def find_quiet_window(force: np.ndarray, dt: float) -> Tuple[int, int, float]:
    """Slide QUIET_WINDOW_S across the first 30 s and return the window with
    the lowest SD (mean must look like standing, i.e. > 100 N).
    Returns (start_idx, end_idx, sd_of_window).
    """
    n_win = max(10, int(round(QUIET_WINDOW_S / dt)))
    search_end = min(len(force), int(round(30.0 / dt)))
    if search_end - n_win <= 0:
        search_end = len(force)

    best_sd = np.inf
    best_i  = 0
    step    = max(1, n_win // 4)
    for i in range(0, search_end - n_win, step):
        seg = force[i:i + n_win]
        if seg.mean() < 100:
            continue
        sd = seg.std()
        if sd < best_sd:
            best_sd, best_i = sd, i
    return best_i, best_i + n_win, best_sd


def detect_jumps(force: np.ndarray, dt: float) -> List[Tuple[int, int]]:
    """Find (takeoff_idx, landing_idx) pairs: flight gaps where the trace
    sits below TAKEOFF_THR_N for at least MIN_FLIGHT_S."""
    below = force < TAKEOFF_THR_N
    jumps: List[Tuple[int, int]] = []
    i, n = 0, len(force)
    min_flight = int(round(MIN_FLIGHT_S / dt))
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if (j - i) >= min_flight and j < n:
                jumps.append((i, j))
            i = j
        else:
            i += 1
    return jumps


def find_pre_jump_stable_anchor(
    force: np.ndarray,
    bw_N: float,
    takeoff_idx: int,
    lower_bound_idx: int,
    dt: float,
) -> int:
    """Walk backwards from just before takeoff in STABLE_WIN_S windows; return
    the END index of the first window (going back) that looks like quiet
    standing (mean within BW±tol and SD < max). This anchor sits between the
    previous jump's recovery noise and the start of the current jump's dip,
    so the onset search won't false-trigger on landing oscillations."""
    n_win = max(20, int(round(STABLE_WIN_S / dt)))
    step  = max(1, n_win // 4)
    end   = takeoff_idx - int(round(0.05 / dt))
    if end - n_win <= lower_bound_idx:
        return lower_bound_idx
    while end - n_win >= lower_bound_idx:
        seg = force[end - n_win:end]
        if abs(seg.mean() - bw_N) < STABLE_MEAN_TOL_N and seg.std() < STABLE_SD_MAX_N:
            return end
        end -= step
    return lower_bound_idx


def find_onset(
    force: np.ndarray,
    bw_N: float,
    sd_quiet: float,
    takeoff_idx: int,
    search_start_idx: int,
    dt: float,
) -> Optional[int]:
    """First sample at or after search_start_idx where (BW - F) exceeds the
    onset threshold for at least ONSET_SUSTAIN_S consecutive samples."""
    thr = max(THR_SD_MULT * sd_quiet, THR_FORCE_N)
    n_consec = max(3, int(round(ONSET_SUSTAIN_S / dt)))
    start = max(0, search_start_idx)
    end   = takeoff_idx
    if end - start < n_consec:
        return None
    f = force[start:end]
    below = (bw_N - f) > thr
    for i in range(len(below) - n_consec):
        if below[i:i + n_consec].all():
            return start + i
    return None


def find_velocity_min(
    force: np.ndarray,
    bw_N: float,
    mass_kg: float,
    onset_idx: int,
    takeoff_idx: int,
    dt: float,
) -> int:
    """End-of-unweighting = velocity minimum. Integrate (F - BW)/m from onset
    and return the index of the most-negative velocity."""
    f_seg = force[onset_idx:takeoff_idx + 1]
    a_seg = (f_seg - bw_N) / mass_kg
    v_seg = np.cumsum(a_seg) * dt
    return onset_idx + int(np.argmin(v_seg))


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------
def analyze_file(path: str) -> Tuple[List[JumpEvents], float, np.ndarray, np.ndarray]:
    """Detect all CMJs in one file. Returns (events, bw_N, time, force)."""
    df, freq, _meta = read_fd_file(path)
    time  = df["Time"].to_numpy()
    force = df["Both"].to_numpy()
    dt    = 1.0 / freq

    q_start, q_end, sd_quiet = find_quiet_window(force, dt)
    bw_N    = float(force[q_start:q_end].mean())
    mass_kg = bw_N / G

    jumps = detect_jumps(force, dt)

    events: List[JumpEvents] = []
    prev_landing_idx = q_end

    for to_i, ld_i in jumps:
        lower   = max(prev_landing_idx, q_end)
        anchor  = find_pre_jump_stable_anchor(force, bw_N, to_i, lower, dt)
        onset_i = find_onset(force, bw_N, sd_quiet, to_i, anchor, dt)
        if onset_i is None:
            prev_landing_idx = ld_i
            continue
        be_i = find_velocity_min(force, bw_N, mass_kg, onset_i, to_i, dt)

        onset_s        = float(time[onset_i])
        braking_end_s  = float(time[be_i])
        takeoff_s      = float(time[to_i])
        landing_s      = float(time[ld_i])
        flight_time_s  = landing_s - takeoff_s

        f_uw    = force[onset_i:be_i + 1] - bw_N
        f_push  = force[be_i:to_i + 1]    - bw_N
        f_total = force[onset_i:to_i + 1] - bw_N
        braking_impulse_Ns    = float(abs(np.sum(f_uw)    * dt))
        propulsive_impulse_Ns = float(np.sum(f_push)  * dt)
        total_impulse_Ns      = float(np.sum(f_total) * dt)

        events.append(JumpEvents(
            jump_id              = 0,
            body_mass_kg         = round(mass_kg, 3),
            onset_s              = round(onset_s, 3),
            braking_end_s        = round(braking_end_s, 3),
            takeoff_s            = round(takeoff_s, 3),
            landing_s            = round(landing_s, 3),
            flight_time_s        = round(flight_time_s, 3),
            braking_impulse_Ns   = round(braking_impulse_Ns, 3),
            propulsive_impulse_Ns= round(propulsive_impulse_Ns, 3),
            total_impulse_Ns     = round(total_impulse_Ns, 3),
        ))
        prev_landing_idx = ld_i

    # number jumps 1..N in chronological order
    for new_id, ev in enumerate(events, start=1):
        ev.jump_id = new_id
    return events, bw_N, time, force


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
EVENT_COLORS = {
    "onset_s":       "#2E86AB",
    "braking_end_s": "#F18F01",
    "takeoff_s":     "#06A77D",
    "landing_s":     "#D62246",
}
EVENT_LABELS = {
    "onset_s":       "Onset of contraction",
    "braking_end_s": "End of unweighting (v min)",
    "takeoff_s":     "Takeoff",
    "landing_s":     "Landing",
}


def plot_file(
    fname: str,
    athlete_id: str,
    time: np.ndarray,
    force: np.ndarray,
    bw_N: float,
    events: List[JumpEvents],
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt  # imported here so --no-plots still works

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time, force, color="#333", linewidth=0.6, label="Force (Both)", zorder=2)
    ax.axhline(bw_N, color="#888", linestyle=":", linewidth=0.9,
               label=f"BW ≈ {bw_N:.0f} N", zorder=1)

    seen = set()
    for ev in events:
        for col in ("onset_s", "braking_end_s", "takeoff_s", "landing_s"):
            t   = getattr(ev, col)
            lab = EVENT_LABELS[col] if EVENT_LABELS[col] not in seen else None
            seen.add(EVENT_LABELS[col])
            ax.axvline(t, color=EVENT_COLORS[col], linestyle="--",
                       linewidth=1.4, alpha=0.9, label=lab, zorder=3)
        ax.axvspan(ev.onset_s, ev.landing_s, color="#FFD166", alpha=0.12, zorder=1)
        mid = (ev.onset_s + ev.landing_s) / 2
        ax.text(mid, 0.97, f"Jump {ev.jump_id}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color="#555",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.85))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical Force - Both (N)")
    ax.set_title(f"{athlete_id} — {fname}  |  CMJ event markers")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.92, ncol=3, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------
def parse_athlete_trial(fname: str) -> Tuple[str, str]:
    """FD_AK_16_12.csv -> ('AK', 'AK_16_12').
       FD_KV_16_12_02.csv -> ('KV', 'KV_16_12_02').
       FD_foo.csv -> ('foo', 'foo')."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    if stem.startswith("FD_"):
        stem = stem[3:]
    parts = stem.split("_")
    athlete = parts[0] if parts else stem
    return athlete, stem


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="CMJ phase detection for ForceDecks raw CSV exports.")
    parser.add_argument("--input-dir",  default=script_dir,
                        help="Folder containing FD_*.csv files (default: script folder)")
    parser.add_argument("--output-dir", default=os.path.join(script_dir, "output"),
                        help="Folder to write the summary CSV and PNGs (default: ./output)")
    parser.add_argument("--no-plots",   action="store_true",
                        help="Skip generating per-file PNG plots")
    parser.add_argument("--pattern",    default="FD_*.csv",
                        help="Glob pattern for files to process (default: FD_*.csv)")
    args = parser.parse_args()

    in_dir  = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, args.pattern)))
    if not files:
        print(f"No files matched {args.pattern!r} in {in_dir}", file=sys.stderr)
        return 1

    print(f"Scanning {in_dir}")
    print(f"Found {len(files)} file(s) matching {args.pattern!r}")
    print(f"Writing outputs to {out_dir}\n")

    all_rows: List[dict] = []
    n_ok = n_err = 0

    for path in files:
        fname = os.path.basename(path)
        try:
            athlete, trial = parse_athlete_trial(fname)
            events, bw_N, time, force = analyze_file(path)
            for ev in events:
                row = dict(
                    athlete_id            = athlete,
                    trial_id              = trial,
                    jump_id               = ev.jump_id,
                    body_mass_kg          = ev.body_mass_kg,
                    onset_s               = ev.onset_s,
                    braking_end_s         = ev.braking_end_s,
                    takeoff_s             = ev.takeoff_s,
                    landing_s             = ev.landing_s,
                    flight_time_s         = ev.flight_time_s,
                    braking_impulse_Ns    = ev.braking_impulse_Ns,
                    propulsive_impulse_Ns = ev.propulsive_impulse_Ns,
                    total_impulse_Ns      = ev.total_impulse_Ns,
                    fd_file               = fname,
                )
                all_rows.append(row)

            if not args.no_plots:
                png_path = os.path.join(out_dir, fname.replace(".csv", "_events.png"))
                plot_file(fname, athlete, time, force, bw_N, events, png_path)
                print(f"  ✓ {fname}: {len(events)} jump(s)  ->  {os.path.basename(png_path)}")
            else:
                print(f"  ✓ {fname}: {len(events)} jump(s)")
            n_ok += 1
        except Exception as e:
            print(f"  ✗ {fname}: {type(e).__name__}: {e}", file=sys.stderr)
            n_err += 1

    if not all_rows:
        print("\nNo jumps detected in any file. Nothing to write.", file=sys.stderr)
        return 2

    out_df = pd.DataFrame(all_rows, columns=[
        "athlete_id","trial_id","jump_id","body_mass_kg",
        "onset_s","braking_end_s","takeoff_s","landing_s","flight_time_s",
        "braking_impulse_Ns","propulsive_impulse_Ns","total_impulse_Ns","fd_file",
    ])
    csv_path = os.path.join(out_dir, "all_athlete_force_jump_events_impulse.csv")
    out_df.to_csv(csv_path, index=False)

    print(f"\nDone. Processed {n_ok} file(s); {n_err} error(s).")
    print(f"Total jumps: {len(out_df)}")
    print(f"Summary CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
