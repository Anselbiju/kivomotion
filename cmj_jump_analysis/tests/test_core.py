import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmj_analysis.modeling import select_feature_names
from cmj_analysis.features import extract_imu_inference_features
from cmj_analysis.signals import assess_imu_quality, contiguous_regions
from cmj_analysis.sync import match_events


def test_contiguous_regions():
    assert contiguous_regions(np.array([False, True, True, False, True])) == [(1, 3), (4, 5)]


def test_affine_event_sync():
    imu = pd.DataFrame({"imu_jump_id": [1, 2, 3], "imu_takeoff_s": [1.0, 3.0, 5.0]})
    force = pd.DataFrame({"jump_id": [1, 2, 3], "takeoff_s": [11.01, 13.01, 15.01]})
    matched, params = match_events(force, imu)
    assert params["median_abs_sync_error_s"] < 1e-9
    assert matched["sync_quality"].eq("good").all()


def test_administrative_columns_never_become_features():
    table = pd.DataFrame(
        {
            "jump_id": [1, 2],
            "session_id": ["s1", "s2"],
            "test_number": [1, 2],
            "recording_date": [20260730, 20260731],
            "target_impulse_ns_kg": [2.1, 2.2],
            "ax_g_f_mean": [0.1, 0.2],
            "acc_mag_g_f_mean": [1.1, 1.2],
            "gyro_mag_area_deg": [10.0, 11.0],
        }
    )
    all_axes = select_feature_names(table, "all_axes")
    assert "test_number" not in all_axes
    assert "recording_date" not in all_axes
    assert "ax_g_f_mean" in all_axes

    magnitude = select_feature_names(table, "magnitude_only")
    assert magnitude == ["acc_mag_g_f_mean", "gyro_mag_area_deg"]


def test_orientation_quality_labels_y_axis_sign_without_modifying_signal():
    time = np.arange(100, dtype=float) / 100.0
    base = pd.DataFrame(
        {
            "time_s": time,
            "ax_g": np.zeros(100),
            "ay_g": np.ones(100),
            "az_g": np.zeros(100),
            "gx_dps": np.zeros(100),
            "gy_dps": np.zeros(100),
            "gz_dps": np.zeros(100),
            "acc_mag_g_f": np.ones(100),
        }
    )
    positive = assess_imu_quality(base, 100.0)
    negative_data = base.copy()
    negative_data["ay_g"] = -1.0
    negative = assess_imu_quality(negative_data, 100.0)

    assert positive["imu_orientation_group"] == "Y_positive"
    assert negative["imu_orientation_group"] == "Y_negative"
    assert not positive["imu_orientation_review"]
    assert not negative["imu_orientation_review"]


def test_target_free_inference_features_match_model_schema():
    time = np.arange(300, dtype=float) / 100.0
    imu = pd.DataFrame({"time_s": time})
    for column in [
        "ax_g_f", "ay_g_f", "az_g_f", "acc_mag_g_f",
        "gx_dps_f", "gy_dps_f", "gz_dps_f", "gyro_mag_dps_f",
    ]:
        imu[column] = 1.0 + 0.1 * np.sin(time)
    events = pd.DataFrame(
        {
            "imu_jump_id": [1],
            "imu_takeoff_s": [1.5],
            "imu_landing_s": [2.0],
            "imu_flight_time_s": [0.4],
        }
    )
    features = extract_imu_inference_features(imu, events)

    assert len(features) == 1
    assert features.loc[0, "jump_id"] == 1
    assert "acc_mag_g_f_mean" in features
    assert "gyro_mag_area_deg" in features
    assert not any(column.startswith("target_") for column in features.columns)
