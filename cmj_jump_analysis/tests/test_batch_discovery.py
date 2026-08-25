import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_batch", ROOT / "scripts" / "run_batch.py")
RUN_BATCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUN_BATCH)


def test_discover_pairs_accepts_optional_session_suffix(tmp_path):
    for name in [
        "20260730_CMJ_Test1.csv",
        "20260730_IMU_Test1.csv",
        "20260730_CMJ_Test1_C.csv",
        "20260730_IMU_Test1_C.csv",
    ]:
        (tmp_path / name).touch()

    pairs, ignored = RUN_BATCH.discover_pairs(tmp_path)

    assert [pair["session_id"] for pair in pairs] == [
        "20260730_Test1",
        "20260730_Test1_C",
    ]
    assert ignored == []


def test_discover_pairs_reports_unrelated_and_incomplete_files(tmp_path):
    (tmp_path / "same_person_candidates.csv").touch()
    (tmp_path / "20260731_IMU_Test2_A.csv").touch()

    pairs, ignored = RUN_BATCH.discover_pairs(tmp_path)

    assert pairs == []
    assert len(ignored) == 2
    assert {row["reason"] for row in ignored} == {
        "filename does not match expected pattern",
        "missing force or IMU file",
    }
