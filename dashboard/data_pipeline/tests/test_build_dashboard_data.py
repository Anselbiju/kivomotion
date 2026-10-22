from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PROJECT_ROOT / "dashboard" / "data_pipeline" / "build_dashboard_data.py"
SPEC = importlib.util.spec_from_file_location("build_dashboard_data", MODULE_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


def test_dashboard_build_reconciles_project_outputs():
    tables, snapshot = builder.build(PROJECT_ROOT)
    quality = tables["quality"].iloc[0]

    assert len(tables["events"]) == len(tables["acceleration_events"])
    assert len(tables["events"]) == quality["matchedJumps"]
    assert len(tables["sessions"]) == quality["matchedSessions"]
    assert len(tables["athletes"]) == quality["athletes"]
    assert len(tables["athlete_comparison"]) == quality["athletes"]
    assert tables["athlete_comparison"]["athleteId"].is_unique
    assert tables["athlete_comparison"]["avgPeakAccelerationG"].notna().all()
    assert "renamedAthlete" in tables["athletes"].columns
    assert set(tables["athlete_aliases"].columns) == {
        "athleteId", "generatedAthleteLabel", "renamedAthlete", "athleteLabel"
    }
    assert len(tables["athlete_aliases"]) == quality["athletes"]
    assert quality["athleteIdCompletenessPct"] == 100
    assert quality["duplicateJumpIds"] == 0
    assert quality["missingAccelerationCells"] == 0
    assert tables["acceleration_events"]["accelerationMagnitudeMinG"].ge(0).all()
    assert tables["acceleration_events"]["flightMinAccelerationG"].ge(0).all()
    assert quality["acceptedEvents"] == quality["matchedJumps"]
    assert quality["forceEventsDetected"] == quality["acceptedEvents"] + quality["forceEventsRejected"]
    assert quality["imuEventsDetected"] == quality["acceptedEvents"] + quality["imuEventsRejected"]
    assert 0 <= quality["forceEventRetentionPct"] <= 100
    assert 0 <= quality["imuEventRetentionPct"] <= 100
    assert len(tables["session_quality"]) == quality["matchedSessions"]
    assert tables["session_quality"]["sessionId"].is_unique
    assert quality["repeatedAthletes"] + quality["singleSessionAthletes"] == quality["athletes"]
    validation = snapshot["queries"]["model_validation"]["rows"][0]
    assert validation["deploymentReady"] is False
    assert validation["modelArtifactVersion"] == 2
    assert validation["modelTrainingRows"] == quality["matchedJumps"]
    assert validation["modelFeatureCount"] > 0
    assert len(validation["modelDatasetSha256"]) == 64


def test_refresh_is_atomic_and_preserves_dashboard_identity(tmp_path):
    database = tmp_path / "kivomotion.sqlite3"
    snapshot = tmp_path / "data.json"
    snapshot.write_text(json.dumps({"id": "dashboard:test-id"}), encoding="utf-8")

    tables, payload = builder.refresh_artifacts(
        PROJECT_ROOT,
        database,
        snapshot,
        "complete",
    )

    assert payload["id"] == "dashboard:test-id"
    assert json.loads(snapshot.read_text(encoding="utf-8"))["buildStatus"] == "complete"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == len(tables["events"])
        assert connection.execute("SELECT COUNT(*) FROM athlete_aliases").fetchone()[0] == len(tables["athletes"])
