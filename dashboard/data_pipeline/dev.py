#!/usr/bin/env python3
"""Refresh dashboard data automatically and run the local Vite preview."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

from build_dashboard_data import MODEL_VALIDATION_FILE, SOURCE_FILES, refresh_artifacts
from refresh_all import ensure_current, raw_fingerprint, resolve_data_dir, run_full_refresh


DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DASHBOARD_ROOT.parent


def source_fingerprint(project_root: Path) -> tuple[tuple[str, int, int], ...]:
    rows = []
    for relative in [*SOURCE_FILES, MODEL_VALIDATION_FILE]:
        path = project_root / relative
        if not path.exists():
            raise FileNotFoundError(
                f"Required dashboard source is missing: {path}. "
                "Run the CMJ batch pipeline first."
            )
        stat = path.stat()
        rows.append((relative, stat.st_mtime_ns, stat.st_size))
    return tuple(rows)


def refresh(project_root: Path) -> None:
    tables, _ = refresh_artifacts(
        project_root,
        DASHBOARD_ROOT / "data" / "kivomotion.sqlite3",
        DASHBOARD_ROOT / "src" / "data.json",
        "complete",
    )
    print(
        "Dashboard data refreshed: "
        f"{len(tables['events'])} jumps, {len(tables['sessions'])} sessions, "
        f"{len(tables['athletes'])} athletes",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--data-dir", type=Path, help="Local synchronized OneDrive Phase 2 Data folder")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_dir = resolve_data_dir(project_root, args.data_dir)
    ensure_current(project_root, data_dir)
    fingerprint = source_fingerprint(project_root)
    raw_state = raw_fingerprint(data_dir)
    command = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    server = subprocess.Popen(command, cwd=DASHBOARD_ROOT)
    try:
        while server.poll() is None:
            time.sleep(max(args.interval, 0.5))
            current_raw = raw_fingerprint(data_dir)
            if current_raw != raw_state:
                try:
                    run_full_refresh(project_root, data_dir)
                    raw_state = current_raw
                    fingerprint = source_fingerprint(project_root)
                except Exception as exc:
                    print(f"Full analysis refresh failed; keeping the last good outputs: {exc}", file=sys.stderr)
                continue
            current = source_fingerprint(project_root)
            if current == fingerprint:
                continue
            try:
                refresh(project_root)
                fingerprint = current
            except Exception as exc:
                print(f"Dashboard refresh failed; keeping the last good snapshot: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        raise SystemExit(server.returncode or 0)


if __name__ == "__main__":
    main()
