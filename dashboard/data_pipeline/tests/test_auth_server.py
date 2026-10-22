from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path


DASHBOARD_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DASHBOARD_DIR))

from auth_server import (  # noqa: E402
    _commit_staged_files,
    _login_page,
    authenticate_user,
    filter_snapshot_for_athlete,
    hash_password,
    load_scientist_reviews,
    normalize_athlete_label,
    persist_athlete_rename,
    persist_scientist_review,
    update_athlete_map,
    verify_password,
)


def test_login_page_preserves_brand_and_accessible_motion() -> None:
    page = _login_page().decode("utf-8")
    assert '<svg class="brand-mark"' in page
    assert "KIVO</strong><span>MOTION" in page
    assert "Decoding movement. Empowering athletes." in page
    assert "prefers-reduced-motion" in page
    assert 'aria-label="Sign in"' in page
    assert 'name="email"' in page and 'name="password"' in page
    assert "<h2>" not in page
    assert 'class="performance-heading"' in page
    assert 'id="login-error"' in _login_page("Bad login").decode("utf-8")


def dashboard_snapshot() -> dict:
    return json.loads((DASHBOARD_DIR / "src" / "data.json").read_text(encoding="utf-8"))


def test_password_hash_round_trip_and_rejection() -> None:
    encoded = hash_password("A-secure-test-password!2026")
    assert verify_password("A-secure-test-password!2026", encoded)
    assert not verify_password("incorrect-password", encoded)


def test_authenticate_user_uses_same_result_shape_for_unknown_account() -> None:
    users = {
        "coach@example.test": {
            "email": "coach@example.test",
            "passwordHash": hash_password("A-secure-test-password!2026"),
        }
    }
    assert authenticate_user(users, "coach@example.test", "A-secure-test-password!2026") == users["coach@example.test"]
    assert authenticate_user(users, "missing@example.test", "A-secure-test-password!2026") is None


def test_staged_file_commit_rolls_back_every_target(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"first-original")
    second.write_bytes(b"second-original")
    first_stage = tmp_path / "first.stage"
    first_stage.write_bytes(b"first-new")
    missing_stage = tmp_path / "missing.stage"
    originals = {first: first.read_bytes(), second: second.read_bytes()}

    try:
        _commit_staged_files([(first_stage, first), (missing_stage, second)], originals)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing staged artifact must abort the coordinated commit")

    assert first.read_bytes() == b"first-original"
    assert second.read_bytes() == b"second-original"


def test_athlete_snapshot_contains_only_assigned_athlete() -> None:
    source = dashboard_snapshot()
    athlete = max(source["queries"]["athletes"]["rows"], key=lambda row: row["sessionCount"])
    athlete_id = athlete["athleteId"]
    private = filter_snapshot_for_athlete(source, athlete_id)

    for query_name in (
        "events",
        "acceleration_events",
        "acceleration_sessions",
        "acceleration_athletes",
        "session_quality",
        "sessions",
        "athletes",
    ):
        rows = private["queries"][query_name]["rows"]
        assert rows
        assert {row["athleteId"] for row in rows} == {athlete_id}

    for query_name in (
        "daily",
        "acceleration_daily",
        "model_performance",
        "predictions",
        "feature_correlations",
        "acceleration_correlations",
        "model_validation",
    ):
        assert private["queries"][query_name]["rows"] == []

    quality = private["queries"]["quality"]["rows"][0]
    assert quality["athletes"] == 1
    assert quality["matchedJumps"] == athlete["jumpCount"]
    assert quality["matchedSessions"] == athlete["sessionCount"]
    assert quality["athleteIdCompletenessPct"] == 100.0


def test_athlete_snapshot_does_not_mutate_team_snapshot() -> None:
    source = dashboard_snapshot()
    original_count = len(source["queries"]["events"]["rows"])
    athlete_id = source["queries"]["athletes"]["rows"][0]["athleteId"]
    filter_snapshot_for_athlete(source, athlete_id)
    assert len(source["queries"]["events"]["rows"]) == original_count


def test_athlete_rename_updates_every_mapping_row(tmp_path: Path) -> None:
    mapping = tmp_path / "athlete_id_map.csv"
    mapping.write_text(
        "athlete_key,athlete_id,date,renamed_athlete\n"
        "A,athlete-1,20260101,\n"
        "A2,athlete-1,20260102,\n"
        "B,athlete-2,20260101,Second athlete\n",
        encoding="utf-8",
    )

    original = update_athlete_map(mapping, "athlete-1", "  Casey   Morgan  ")
    assert b"renamed_athlete" in original
    updated = mapping.read_text(encoding="utf-8")
    assert updated.count("Casey   Morgan") == 2
    assert "Second athlete" in updated
    assert normalize_athlete_label("  Casey   Morgan  ") == "Casey Morgan"


def test_athlete_rename_rejects_unknown_id(tmp_path: Path) -> None:
    mapping = tmp_path / "athlete_id_map.csv"
    mapping.write_text("athlete_id\nathlete-1\n", encoding="utf-8")
    try:
        update_athlete_map(mapping, "unknown", "Name")
    except ValueError as exc:
        assert "not present" in str(exc)
    else:
        raise AssertionError("Unknown athlete IDs must be rejected")


def test_persistent_rename_rebuilds_csv_database_and_snapshot(tmp_path: Path) -> None:
    project = tmp_path / "project"
    phase_data = project / "Phase 2 Data jumps"
    dashboard = project / "dashboard"
    phase_data.mkdir(parents=True)
    dashboard.mkdir()
    os.symlink(DASHBOARD_DIR.parent / "cmj_jump_analysis", project / "cmj_jump_analysis", target_is_directory=True)
    mapping = phase_data / "athlete_id_map.csv"
    shutil.copy2(DASHBOARD_DIR.parent / "Phase 2 Data jumps" / "athlete_id_map.csv", mapping)
    athlete_id = dashboard_snapshot()["queries"]["athletes"]["rows"][0]["athleteId"]
    database = dashboard / "kivomotion.sqlite3"
    snapshot_path = dashboard / "data.json"

    snapshot, renamed, effective = persist_athlete_rename(
        athlete_id,
        "Casey Morgan",
        project_root=project,
        athlete_map_path=mapping,
        database_path=database,
        snapshot_path=snapshot_path,
    )

    assert renamed == effective == "Casey Morgan"
    assert next(row for row in snapshot["queries"]["athletes"]["rows"] if row["athleteId"] == athlete_id)["renamedAthlete"] == "Casey Morgan"
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT renamedAthlete FROM athlete_aliases WHERE athleteId = ?", (athlete_id,)
        ).fetchone()[0] == "Casey Morgan"


def test_scientist_review_ledger_preserves_original_values_and_audit_identity(tmp_path: Path) -> None:
    snapshot = dashboard_snapshot()
    event = next(row for row in snapshot["queries"]["events"]["rows"] if row["flightTimeDiffMs"] > 100)
    ledger = tmp_path / "scientist_reviews.json"
    reviewer = {"email": "scientist@example.test", "displayName": "Test Scientist"}
    payload = {
        "jumpId": event["jumpId"],
        "proposedImuFlightTimeMs": event["forceFlightTimeMs"] - 12,
        "decision": "accepted",
        "reason": "event_alignment",
        "outcome": "correction_verified",
        "note": "Reviewed force and IMU flight intervals.",
        "status": "draft",
    }

    draft = persist_scientist_review(payload, reviewer, snapshot, path=ledger)
    assert draft["original"]["imuFlightTimeMs"] == event["imuFlightTimeMs"]
    assert draft["proposed"]["flightTimeDiffMs"] == 12
    assert draft["reviewer"] == reviewer
    assert load_scientist_reviews(ledger) == [draft]

    submitted = persist_scientist_review({**payload, "status": "submitted"}, reviewer, snapshot, path=ledger)
    assert submitted["reviewId"] == draft["reviewId"]
    assert submitted["createdAt"] == draft["createdAt"]
    assert submitted["status"] == "submitted"
    assert len(load_scientist_reviews(ledger)) == 1


def test_scientist_review_rejects_unknown_jump_and_blank_submission_note(tmp_path: Path) -> None:
    snapshot = dashboard_snapshot()
    reviewer = {"email": "scientist@example.test", "displayName": "Test Scientist"}
    base = {
        "proposedImuFlightTimeMs": 400,
        "decision": "accepted",
        "reason": "event_alignment",
        "outcome": "correction_verified",
        "note": "Reviewed source intervals.",
        "status": "submitted",
    }
    try:
        persist_scientist_review({**base, "jumpId": 999_999}, reviewer, snapshot, path=tmp_path / "ledger.json")
    except ValueError as exc:
        assert "not present" in str(exc)
    else:
        raise AssertionError("Unknown jump IDs must be rejected")

    event = snapshot["queries"]["events"]["rows"][0]
    try:
        persist_scientist_review({**base, "jumpId": event["jumpId"], "note": ""}, reviewer, snapshot, path=tmp_path / "ledger.json")
    except ValueError as exc:
        assert "reviewer note" in str(exc).lower()
    else:
        raise AssertionError("Submitted corrections require a reviewer note")
