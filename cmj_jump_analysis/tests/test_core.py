import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmj_analysis.signals import contiguous_regions
from cmj_analysis.sync import match_events


def test_contiguous_regions():
    assert contiguous_regions(np.array([False, True, True, False, True])) == [(1, 3), (4, 5)]


def test_affine_event_sync():
    imu = pd.DataFrame({"imu_jump_id": [1, 2, 3], "imu_takeoff_s": [1.0, 3.0, 5.0]})
    force = pd.DataFrame({"jump_id": [1, 2, 3], "takeoff_s": [11.01, 13.01, 15.01]})
    matched, params = match_events(force, imu)
    assert params["median_abs_sync_error_s"] < 1e-9
    assert matched["sync_quality"].eq("good").all()

