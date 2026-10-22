#!/usr/bin/env python3
"""Refresh synchronized jump data, build the app, and start the login server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from refresh_all import PROJECT_ROOT, ensure_current, resolve_data_dir


DASHBOARD_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-dir", type=Path, help="Local synchronized OneDrive Phase 2 Data folder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4180)
    parser.add_argument("--secure-cookies", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_dir = resolve_data_dir(project_root, args.data_dir)
    print(f"KivoMotion data source: {data_dir}", flush=True)
    ensure_current(project_root, data_dir)

    subprocess.run(["npm", "run", "build:auth"], check=True, cwd=DASHBOARD_ROOT)

    server_command = [
        sys.executable,
        str(DASHBOARD_ROOT / "auth_server.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.secure_cookies:
        server_command.append("--secure-cookies")
    os.execv(sys.executable, server_command)


if __name__ == "__main__":
    main()
