#!/usr/bin/env python3
"""Rebuild the CMJ summaries and atomically refresh dashboard artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile

from build_dashboard_data import refresh_artifacts


DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DASHBOARD_ROOT.parent
SUMMARY_FILES = [
    "all_matched_events.csv",
    "all_model_features.csv",
    "batch_session_status.csv",
    "model_validation.json",
    "reports/model_leaderboard.csv",
    "reports/out_of_fold_predictions.csv",
]
PROMOTED_FILES = [
    *SUMMARY_FILES,
    "models/best_impulse_model.joblib",
    "models/best_all_axes_model.joblib",
    "models/best_magnitude_only_model.joblib",
    "reports/prediction_report.html",
    "reports/prediction_report.png",
]
SOURCE_STATE_FILE = DASHBOARD_ROOT / "data" / "source_state.json"


def _csv_count(path: Path) -> int:
    return sum(1 for item in path.glob("*.csv") if item.is_file())


def resolve_data_dir(project_root: Path, requested: Path | None = None) -> Path:
    """Select an explicit source, a configured source, or the best local OneDrive shortcut."""
    configured = requested or (
        Path(os.environ["KIVOMOTION_DATA_DIR"])
        if os.environ.get("KIVOMOTION_DATA_DIR")
        else None
    )
    if configured is not None:
        source = configured.expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Configured jump data directory does not exist: {source}")
        return source

    cloud_root = Path.home() / "Library" / "CloudStorage"
    one_drive_candidates = [
        path
        for path in cloud_root.glob(
            "OneDrive*/Shortcuts/T17 Shared Folder/Phase 2 Data"
        )
        if path.is_dir() and (path / "athlete_id_map.csv").is_file()
    ]
    if one_drive_candidates:
        return max(
            one_drive_candidates,
            key=lambda path: (_csv_count(path), path.stat().st_mtime_ns),
        ).resolve()

    fallback = (project_root / "Phase 2 Data jumps").resolve()
    if not fallback.is_dir():
        raise FileNotFoundError(
            "No synchronized Phase 2 Data folder was found. Refresh the OneDrive shortcut "
            "or set KIVOMOTION_DATA_DIR to its local Phase 2 Data path."
        )
    return fallback


def raw_fingerprint(data_dir: Path) -> dict:
    files = []
    for path in sorted(data_dir.glob("*.csv")):
        stat = path.stat()
        files.append({
            "name": path.name,
            "size": stat.st_size,
            "modifiedNs": stat.st_mtime_ns,
        })
    if not files:
        raise FileNotFoundError(f"No CSV files are available in the synchronized folder: {data_dir}")
    return {"dataDir": str(data_dir.resolve()), "files": files}


def pipeline_fingerprint(project_root: Path) -> str:
    """Hash processing code so algorithm changes trigger a fresh analysis."""
    analysis_root = project_root / "cmj_jump_analysis"
    paths = [
        *sorted((analysis_root / "src").rglob("*.py")),
        *sorted((analysis_root / "scripts").glob("*.py")),
        analysis_root / "run_pipeline.py",
        DASHBOARD_ROOT / "data_pipeline" / "build_dashboard_data.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def processing_state(project_root: Path, data_dir: Path) -> dict:
    return {
        "source": raw_fingerprint(data_dir),
        "pipelineSha256": pipeline_fingerprint(project_root),
    }


def read_source_state(path: Path = SOURCE_STATE_FILE) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_source_state(state: dict, path: Path = SOURCE_STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def mirror_athlete_map(source_dir: Path, project_root: Path) -> None:
    """Mirror identity mappings while preserving dashboard-managed display names."""
    source = source_dir / "athlete_id_map.csv"
    if not source.exists():
        raise FileNotFoundError(f"OneDrive data source is missing athlete_id_map.csv: {source_dir}")
    target = project_root / "Phase 2 Data jumps" / "athlete_id_map.csv"
    target.parent.mkdir(parents=True, exist_ok=True)

    preserved_names: dict[str, str] = {}
    existing_fieldnames: list[str] = []
    existing_rows: list[dict[str, str]] = []
    if target.exists():
        with target.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)
            for row in existing_rows:
                athlete_id = str(row.get("athlete_id", "")).strip()
                renamed = str(row.get("renamed_athlete", "")).strip()
                if athlete_id and renamed:
                    preserved_names[athlete_id] = renamed

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "athlete_id" not in fieldnames:
        raise ValueError("OneDrive athlete_id_map.csv is missing the athlete_id column")
    if "renamed_athlete" not in fieldnames:
        fieldnames.append("renamed_athlete")
    for row in rows:
        athlete_id = str(row.get("athlete_id", "")).strip()
        row["renamed_athlete"] = str(row.get("renamed_athlete", "") or "").strip() or preserved_names.get(athlete_id, "")

    if existing_fieldnames == fieldnames and existing_rows == rows:
        return

    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def dashboard_artifacts_current(project_root: Path) -> bool:
    """Reuse a completed local snapshot when its inputs and athlete names match."""
    database = DASHBOARD_ROOT / "data" / "kivomotion.sqlite3"
    snapshot_path = DASHBOARD_ROOT / "src" / "data.json"
    athlete_map = project_root / "Phase 2 Data jumps" / "athlete_id_map.csv"
    if not all(path.is_file() for path in (database, snapshot_path, athlete_map)):
        return False
    try:
        artifact_time = min(database.stat().st_mtime_ns, snapshot_path.stat().st_mtime_ns)
        summaries = project_root / "cmj_jump_analysis" / "batch_results"
        if any((summaries / relative).stat().st_mtime_ns > artifact_time for relative in SUMMARY_FILES):
            return False

        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot.get("buildStatus") != "complete":
            return False
        queries = snapshot["queries"]
        event_rows = queries["events"]["rows"]
        trace_rows = queries["signal_traces"]["rows"]
        athlete_rows = queries["athletes"]["rows"]
        event_ids = {row["jumpId"] for row in event_rows}
        trace_ids = {row["jumpId"] for row in trace_rows}
        if not event_ids or len(event_ids) != len(event_rows) or trace_ids != event_ids:
            return False

        with athlete_map.open("r", encoding="utf-8", newline="") as handle:
            names = {
                str(row["athlete_id"]): str(row.get("renamed_athlete") or "").strip()
                for row in csv.DictReader(handle)
            }
        if any(str(row.get("renamedAthlete") or "").strip() != names.get(str(row["athleteId"]), "")
               for row in athlete_rows):
            return False

        with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as connection:
            database_event_ids = {row[0] for row in connection.execute("SELECT jumpId FROM events")}
            database_trace_ids = {row[0] for row in connection.execute("SELECT jumpId FROM signal_traces")}
            database_names = {
                str(athlete_id): str(renamed or "").strip()
                for athlete_id, renamed in connection.execute("SELECT athleteId, renamedAthlete FROM athletes")
            }
        return (database_event_ids == event_ids and database_trace_ids == event_ids
                and all(database_names.get(str(row["athleteId"]), "") == names.get(str(row["athleteId"]), "")
                        for row in athlete_rows))
    except (OSError, ValueError, KeyError, TypeError, csv.Error, sqlite3.DatabaseError):
        return False


def promote_summaries(staging: Path, destination: Path) -> None:
    missing = [relative for relative in PROMOTED_FILES if not (staging / relative).exists()]
    if missing:
        raise RuntimeError(f"Batch refresh did not produce required outputs: {missing}")
    for relative in PROMOTED_FILES:
        source = staging / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def run_full_refresh(
    project_root: Path,
    data_dir: Path | None = None,
    state_path: Path = SOURCE_STATE_FILE,
) -> None:
    project_root = project_root.resolve()
    source_dir = resolve_data_dir(project_root, data_dir)
    source_state = processing_state(project_root, source_dir)
    analysis_root = project_root / "cmj_jump_analysis"
    destination = analysis_root / "batch_results"
    print(f"Using synchronized jump data: {source_dir}", flush=True)
    with tempfile.TemporaryDirectory(prefix="kivomotion-batch-") as temporary_dir:
        staging = Path(temporary_dir)
        subprocess.run(
            [
                sys.executable,
                str(analysis_root / "scripts" / "run_batch.py"),
                "--data-dir",
                str(source_dir),
                "--output-dir",
                str(staging),
            ],
            check=True,
            cwd=project_root,
        )
        promote_summaries(staging, destination)

    mirror_athlete_map(source_dir, project_root)
    tables, _ = refresh_artifacts(
        project_root,
        DASHBOARD_ROOT / "data" / "kivomotion.sqlite3",
        DASHBOARD_ROOT / "src" / "data.json",
        "complete",
        raw_data_dir=source_dir,
    )
    write_source_state(source_state, state_path)
    print(
        "Full project refresh complete: "
        f"{len(tables['events'])} jumps, {len(tables['sessions'])} sessions, "
        f"{len(tables['athletes'])} athletes"
    )


def ensure_current(
    project_root: Path,
    data_dir: Path | None = None,
    state_path: Path = SOURCE_STATE_FILE,
) -> bool:
    """Reprocess changed OneDrive inputs; otherwise refresh only dashboard artifacts."""
    project_root = project_root.resolve()
    source_dir = resolve_data_dir(project_root, data_dir)
    current_state = processing_state(project_root, source_dir)
    destination = project_root / "cmj_jump_analysis" / "batch_results"
    outputs_ready = all((destination / relative).exists() for relative in SUMMARY_FILES)
    previous_state = read_source_state(state_path)
    if not outputs_ready or previous_state != current_state:
        if not outputs_ready:
            reason = "analysis outputs are missing"
        elif previous_state and previous_state.get("source") == current_state["source"]:
            reason = "impulse-processing code changed"
        else:
            reason = "synchronized CSV files changed"
        print(f"Running impulse analysis because {reason}.", flush=True)
        run_full_refresh(project_root, source_dir, state_path)
        return True

    mirror_athlete_map(source_dir, project_root)
    if dashboard_artifacts_current(project_root):
        print("Impulse outputs and dashboard artifacts are current; skipped refresh.", flush=True)
        return False

    print("Refreshing dashboard artifacts, including force and IMU review traces. This may take several minutes when OneDrive files need to download.", flush=True)
    tables, _ = refresh_artifacts(
        project_root,
        DASHBOARD_ROOT / "data" / "kivomotion.sqlite3",
        DASHBOARD_ROOT / "src" / "data.json",
        "complete",
        raw_data_dir=source_dir,
    )
    print(
        "Impulse outputs are current; refreshed dashboard artifacts: "
        f"{len(tables['events'])} jumps, {len(tables['sessions'])} sessions, "
        f"{len(tables['athletes'])} athletes",
        flush=True,
    )
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-dir", type=Path, help="Local synchronized OneDrive Phase 2 Data folder")
    parser.add_argument("--if-needed", action="store_true", help="Skip impulse reprocessing when source files are unchanged")
    args = parser.parse_args()
    if args.if_needed:
        ensure_current(args.project_root, args.data_dir)
    else:
        run_full_refresh(args.project_root, args.data_dir)


if __name__ == "__main__":
    main()
