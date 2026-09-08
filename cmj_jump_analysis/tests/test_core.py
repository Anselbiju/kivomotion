import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmj_analysis.signals import contiguous_regions
from cmj_analysis.sync import match_events
from cmj_analysis.io import Recording, matching_athlete_id
from cmj_analysis.events import detect_force_jumps
from scripts.run_batch import discover_pairs


def test_contiguous_regions():
    assert contiguous_regions(np.array([False, True, True, False, True])) == [(1, 3), (4, 5)]


def test_affine_event_sync():
    imu = pd.DataFrame({"imu_jump_id": [1, 2, 3], "imu_takeoff_s": [1.0, 3.0, 5.0]})
    force = pd.DataFrame({"jump_id": [1, 2, 3], "takeoff_s": [11.01, 13.01, 15.01]})
    matched, params = match_events(force, imu)
    assert params["median_abs_sync_error_s"] < 1e-9
    assert matched["sync_quality"].eq("good").all()


def test_matching_athlete_id():
    athlete_id = "e80acbfc-6a23-4d74-947c-772cb5972224"
    force = Recording(pd.DataFrame(), {"AthleteId": athlete_id}, Path("force.csv"))
    imu = Recording(pd.DataFrame(), {"athleteId": athlete_id}, Path("imu.csv"))
    assert matching_athlete_id(force, imu) == athlete_id


def test_mismatched_athlete_id_is_rejected():
    force = Recording(
        pd.DataFrame(),
        {"AthleteId": "e80acbfc-6a23-4d74-947c-772cb5972224"},
        Path("force.csv"),
    )
    imu = Recording(
        pd.DataFrame(),
        {"athleteId": "d2767b47-5655-4a37-8fdf-3e662ee27a06"},
        Path("imu.csv"),
    )
    try:
        matching_athlete_id(force, imu)
    except ValueError as exc:
        assert "athlete ID mismatch" in str(exc)
    else:
        raise AssertionError("Expected a mismatched athlete ID to be rejected")


def test_discover_pairs_accepts_repeat_athlete_suffix(tmp_path):
    (tmp_path / "20260730_CMJ_Test9_A.csv").touch()
    (tmp_path / "20260730_IMU_Test9_A.csv").touch()
    pairs, ignored = discover_pairs(tmp_path)
    assert not ignored
    assert len(pairs) == 1
    assert pairs[0]["session_id"] == "20260730_Test9_A"
    assert pairs[0]["suffix"] == "A"


def test_force_detection_bridges_brief_filter_ring_gap():
    fs_hz = 100.0
    force = np.full(400, 800.0)
    force[150:190] = 0.0
    force[175:177] = 100.0
    force[190:205] = 1_200.0
    frame = pd.DataFrame({"Time": np.arange(400) / fs_hz, "force_n": force})
    events = detect_force_jumps(frame, fs_hz, mass_kg=80.0)
    assert len(events) == 1
    assert events.iloc[0]["flight_time_s"] >= 0.39
