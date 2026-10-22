from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PIPELINE = PROJECT_ROOT / "dashboard" / "data_pipeline"
sys.path.insert(0, str(DATA_PIPELINE))
SPEC = importlib.util.spec_from_file_location("refresh_all", DATA_PIPELINE / "refresh_all.py")
REFRESH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REFRESH)


def write_map(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["athlete_key", "athlete_id", "renamed_athlete"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_raw_fingerprint_changes_when_a_synchronized_csv_changes(tmp_path: Path) -> None:
    source = tmp_path / "Phase 2 Data"
    source.mkdir()
    csv_path = source / "jump.csv"
    csv_path.write_text("value\n1\n", encoding="utf-8")
    before = REFRESH.raw_fingerprint(source)

    csv_path.write_text("value\n100\n", encoding="utf-8")
    after = REFRESH.raw_fingerprint(source)

    assert before != after
    assert after["dataDir"] == str(source.resolve())


def test_mirror_athlete_map_preserves_local_display_name(tmp_path: Path) -> None:
    source = tmp_path / "onedrive"
    project = tmp_path / "project"
    write_map(
        source / "athlete_id_map.csv",
        [{"athlete_key": "one", "athlete_id": "athlete-1", "renamed_athlete": ""}],
    )
    write_map(
        project / "Phase 2 Data jumps" / "athlete_id_map.csv",
        [{"athlete_key": "one", "athlete_id": "athlete-1", "renamed_athlete": "Alex"}],
    )

    REFRESH.mirror_athlete_map(source, project)

    mirrored = project / "Phase 2 Data jumps" / "athlete_id_map.csv"
    first_modified_ns = mirrored.stat().st_mtime_ns
    REFRESH.mirror_athlete_map(source, project)
    assert mirrored.stat().st_mtime_ns == first_modified_ns

    with mirrored.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["renamed_athlete"] == "Alex"


def test_explicit_data_directory_takes_priority(tmp_path: Path) -> None:
    source = tmp_path / "synced"
    source.mkdir()
    assert REFRESH.resolve_data_dir(tmp_path, source) == source.resolve()


def test_completed_dashboard_artifacts_skip_raw_signal_reload(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path
    dashboard_root = project / "dashboard"
    database = dashboard_root / "data" / "kivomotion.sqlite3"
    snapshot = dashboard_root / "src" / "data.json"
    database.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    monkeypatch.setattr(REFRESH, "DASHBOARD_ROOT", dashboard_root)

    summaries = project / "cmj_jump_analysis" / "batch_results"
    for relative in REFRESH.SUMMARY_FILES:
        path = summaries / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("reviewed output", encoding="utf-8")
    athlete_map = project / "Phase 2 Data jumps" / "athlete_id_map.csv"
    write_map(athlete_map, [{"athlete_key": "one", "athlete_id": "athlete-1", "renamed_athlete": "Alex"}])

    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE events (jumpId INTEGER)")
        connection.execute("INSERT INTO events VALUES (7)")
        connection.execute("CREATE TABLE signal_traces (jumpId INTEGER)")
        connection.execute("INSERT INTO signal_traces VALUES (7)")
        connection.execute("CREATE TABLE athletes (athleteId TEXT, renamedAthlete TEXT)")
        connection.execute("INSERT INTO athletes VALUES ('athlete-1', 'Alex')")
    snapshot.write_text(json.dumps({
        "buildStatus": "complete",
        "queries": {
            "events": {"rows": [{"jumpId": 7}]},
            "signal_traces": {"rows": [{"jumpId": 7}]},
            "athletes": {"rows": [{"athleteId": "athlete-1", "renamedAthlete": "Alex"}]},
        },
    }), encoding="utf-8")

    assert REFRESH.dashboard_artifacts_current(project)

    write_map(athlete_map, [{"athlete_key": "one", "athlete_id": "athlete-1", "renamed_athlete": "Sam"}])
    assert not REFRESH.dashboard_artifacts_current(project)

    write_map(athlete_map, [{"athlete_key": "one", "athlete_id": "athlete-1", "renamed_athlete": "Alex"}])
    changed_summary = summaries / REFRESH.SUMMARY_FILES[0]
    newer = max(database.stat().st_mtime_ns, snapshot.stat().st_mtime_ns) + 1_000_000_000
    os.utime(changed_summary, ns=(newer, newer))
    assert not REFRESH.dashboard_artifacts_current(project)
