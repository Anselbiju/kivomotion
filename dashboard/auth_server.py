#!/usr/bin/env python3
"""Authenticated local server for the KivoMotion dashboard.

Coach sessions receive the complete reviewed snapshot. Athlete sessions receive
only rows matching the athlete ID assigned to their account. The browser never
receives the full team snapshot for an athlete session.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import hmac
import html
import io
import json
import os
import secrets
import tempfile
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from data_pipeline.build_dashboard_data import build, short_label, write_database, write_snapshot


DASHBOARD_DIR = Path(__file__).resolve().parent
AUTH_DIST_DIR = DASHBOARD_DIR / "auth-dist"
DEFAULT_USERS_FILE = DASHBOARD_DIR / "auth" / "users.local.json"
ATHLETE_MAP_FILE = DASHBOARD_DIR.parent / "Phase 2 Data jumps" / "athlete_id_map.csv"
DATABASE_FILE = DASHBOARD_DIR / "data" / "kivomotion.sqlite3"
SNAPSHOT_FILE = DASHBOARD_DIR / "src" / "data.json"
SCIENTIST_REVIEWS_FILE = DASHBOARD_DIR / "data" / "scientist_reviews.json"
SESSION_COOKIE = "kivo_session"
SESSION_TTL_SECONDS = 8 * 60 * 60
MAX_LOGIN_BODY = 8_192
LOGIN_WINDOW_SECONDS = 15 * 60
MAX_LOGIN_ATTEMPTS = 8
MAX_RENAME_BODY = 4_096
MAX_ATHLETE_LABEL_LENGTH = 48
MAX_SCIENTIST_REVIEW_BODY = 16_384
MAX_REVIEW_NOTE_LENGTH = 800
REVIEW_REASONS = {
    "event_alignment": "Event-alignment correction",
    "athlete_assignment": "Athlete assignment correction",
    "source_transcription": "Source transcription correction",
    "exclude_invalid_trial": "Exclude invalid trial",
}
REVIEW_OUTCOMES = {
    "correction_verified": "Correction verified",
    "second_review": "Needs second review",
    "reprocess": "Return for reprocessing",
}
RECORD_DECISIONS = {"accepted", "excluded", "needs_reprocessing"}


def normalize_athlete_label(value: object) -> str:
    label = " ".join(str(value or "").split())
    if len(label) > MAX_ATHLETE_LABEL_LENGTH:
        raise ValueError(f"Athlete display names must contain {MAX_ATHLETE_LABEL_LENGTH} characters or fewer.")
    return label


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_scientist_reviews(path: Path = SCIENTIST_REVIEWS_FILE) -> list[dict]:
    """Load the auditable correction ledger without changing source measurements."""
    if not path.exists():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    reviews = document.get("reviews", [])
    if document.get("version") != 1 or not isinstance(reviews, list):
        raise ValueError("The scientist review ledger has an unsupported format.")
    return reviews


def persist_scientist_review(
    payload: dict,
    reviewer: dict,
    snapshot: dict,
    *,
    path: Path = SCIENTIST_REVIEWS_FILE,
) -> dict:
    """Validate and atomically upsert one scientist review ledger entry."""
    try:
        jump_id = int(payload.get("jumpId"))
    except (TypeError, ValueError):
        raise ValueError("A valid jump ID is required.") from None
    events = snapshot.get("queries", {}).get("events", {}).get("rows", [])
    event = next((item for item in events if item.get("jumpId") == jump_id), None)
    if event is None:
        raise ValueError("The supplied jump ID is not present in the reviewed dashboard snapshot.")

    review_status = str(payload.get("status", "")).strip()
    if review_status not in {"draft", "submitted"}:
        raise ValueError("Review status must be draft or submitted.")
    decision = str(payload.get("decision", "")).strip()
    if decision not in RECORD_DECISIONS:
        raise ValueError("Select a valid record decision.")
    reason = str(payload.get("reason", "")).strip()
    if reason not in REVIEW_REASONS:
        raise ValueError("Select a valid correction reason.")
    outcome = str(payload.get("outcome", "")).strip()
    if outcome not in REVIEW_OUTCOMES:
        raise ValueError("Select a valid review outcome.")
    note = str(payload.get("note", "")).strip()
    if len(note) > MAX_REVIEW_NOTE_LENGTH:
        raise ValueError(f"Reviewer notes must contain {MAX_REVIEW_NOTE_LENGTH} characters or fewer.")
    if review_status == "submitted" and not note:
        raise ValueError("A reviewer note is required before submission.")
    try:
        proposed_imu_flight_time_ms = float(payload.get("proposedImuFlightTimeMs"))
    except (TypeError, ValueError):
        raise ValueError("Enter a valid proposed IMU flight time.") from None
    if not 0 <= proposed_imu_flight_time_ms <= 5_000:
        raise ValueError("Proposed IMU flight time must be between 0 and 5000 ms.")

    force_flight_time_ms = float(event.get("forceFlightTimeMs") or 0)
    proposed_difference_ms = abs(force_flight_time_ms - proposed_imu_flight_time_ms)
    now = datetime.now(timezone.utc).isoformat()
    reviews = load_scientist_reviews(path)
    existing = next((item for item in reviews if item.get("jumpId") == jump_id), None)
    review = {
        "reviewId": existing.get("reviewId") if existing else f"jump-{jump_id}",
        "jumpId": jump_id,
        "athleteId": event.get("athleteId"),
        "sessionId": event.get("sessionId"),
        "original": {
            "forceFlightTimeMs": event.get("forceFlightTimeMs"),
            "imuFlightTimeMs": event.get("imuFlightTimeMs"),
            "flightTimeDiffMs": event.get("flightTimeDiffMs"),
            "decision": "accepted",
        },
        "proposed": {
            "imuFlightTimeMs": proposed_imu_flight_time_ms,
            "flightTimeDiffMs": proposed_difference_ms,
            "decision": decision,
        },
        "reason": reason,
        "reasonLabel": REVIEW_REASONS[reason],
        "outcome": outcome,
        "outcomeLabel": REVIEW_OUTCOMES[outcome],
        "note": note,
        "status": review_status,
        "reviewer": {
            "email": reviewer.get("email"),
            "displayName": reviewer.get("displayName") or reviewer.get("email"),
        },
        "createdAt": existing.get("createdAt") if existing else now,
        "updatedAt": now,
    }
    updated = [item for item in reviews if item.get("jumpId") != jump_id]
    updated.append(review)
    updated.sort(key=lambda item: item.get("updatedAt", ""), reverse=True)
    document = {"version": 1, "reviews": updated}
    _atomic_write_bytes(path, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return review


def update_athlete_map(path: Path, athlete_id: str, renamed_athlete: str) -> bytes:
    """Persist one display name and return the prior bytes for rollback."""
    original, updated = render_athlete_map_update(path, athlete_id, renamed_athlete)
    _atomic_write_bytes(path, updated)
    return original


def render_athlete_map_update(path: Path, athlete_id: str, renamed_athlete: str) -> tuple[bytes, bytes]:
    """Return original and updated CSV bytes without changing the source file."""
    original = path.read_bytes()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "athlete_id" not in fieldnames:
        raise ValueError("athlete_id_map.csv is missing the athlete_id column.")
    if "renamed_athlete" not in fieldnames:
        fieldnames.append("renamed_athlete")
    found = False
    for row in rows:
        if row.get("athlete_id") == athlete_id:
            row["renamed_athlete"] = renamed_athlete
            found = True
        elif row.get("renamed_athlete") is None:
            row["renamed_athlete"] = ""
    if not found:
        raise ValueError("The supplied athlete ID is not present in athlete_id_map.csv.")

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return original, output.getvalue().encode("utf-8")


def _stage_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.stage.", dir=path.parent)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return Path(temporary)


def _commit_staged_files(staged_files: list[tuple[Path, Path]], originals: dict[Path, bytes | None]) -> None:
    """Replace all staged artifacts and restore every target if any replacement fails."""
    try:
        for staged, target in staged_files:
            os.replace(staged, target)
    except Exception:
        for target, original in originals.items():
            if original is None:
                target.unlink(missing_ok=True)
            else:
                _atomic_write_bytes(target, original)
        raise
    finally:
        for staged, _ in staged_files:
            staged.unlink(missing_ok=True)


def persist_athlete_rename(
    athlete_id: str,
    renamed_athlete: object,
    *,
    project_root: Path = DASHBOARD_DIR.parent,
    athlete_map_path: Path = ATHLETE_MAP_FILE,
    database_path: Path = DATABASE_FILE,
    snapshot_path: Path = SNAPSHOT_FILE,
) -> tuple[dict, str, str]:
    """Stage and commit the CSV, SQLite database, and JSON snapshot together."""
    label = normalize_athlete_label(renamed_athlete)
    original_map, updated_map = render_athlete_map_update(athlete_map_path, athlete_id, label)
    staged_map = _stage_bytes(athlete_map_path, updated_map)
    database_stage: Path | None = None
    snapshot_stage: Path | None = None
    try:
        tables, snapshot = build(project_root.resolve(), athlete_map_path=staged_map)
        if snapshot_path.exists():
            try:
                existing_id = json.loads(snapshot_path.read_text(encoding="utf-8")).get("id")
            except json.JSONDecodeError:
                existing_id = None
            if existing_id:
                snapshot["id"] = existing_id
        snapshot["buildStatus"] = "complete"

        database_descriptor, database_name = tempfile.mkstemp(
            prefix=f".{database_path.name}.stage.", dir=database_path.parent
        )
        os.close(database_descriptor)
        database_stage = Path(database_name)
        database_stage.unlink()
        snapshot_descriptor, snapshot_name = tempfile.mkstemp(
            prefix=f".{snapshot_path.name}.stage.", dir=snapshot_path.parent
        )
        os.close(snapshot_descriptor)
        snapshot_stage = Path(snapshot_name)
        snapshot_stage.unlink()
        write_database(database_stage, tables)
        write_snapshot(snapshot_stage, snapshot)

        originals = {
            athlete_map_path: original_map,
            database_path: database_path.read_bytes() if database_path.exists() else None,
            snapshot_path: snapshot_path.read_bytes() if snapshot_path.exists() else None,
        }
        _commit_staged_files(
            [(staged_map, athlete_map_path), (database_stage, database_path), (snapshot_stage, snapshot_path)],
            originals,
        )
    finally:
        staged_map.unlink(missing_ok=True)
        if database_stage is not None:
            database_stage.unlink(missing_ok=True)
        if snapshot_stage is not None:
            snapshot_stage.unlink(missing_ok=True)
    effective_label = label or short_label(athlete_id)
    return snapshot, label, effective_label


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """Return a versioned scrypt password hash suitable for the local user file."""
    if len(password) < 12:
        raise ValueError("Passwords must contain at least 12 characters.")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return "scrypt$16384$8$1$%s$%s" % (
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, encoded: str) -> bool:
    """Verify a password without leaking comparison timing."""
    try:
        algorithm, n, r, p, salt_text, digest_text = encoded.split("$")
        if algorithm != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


_DUMMY_PASSWORD_HASH = hash_password("KivoMotion-dummy-login-secret")


def authenticate_user(users: dict[str, dict], email: str, password: str) -> dict | None:
    """Verify known and unknown accounts through the same scrypt code path."""
    user = users.get(email)
    encoded = user["passwordHash"] if user else _DUMMY_PASSWORD_HASH
    password_matches = verify_password(password, encoded)
    return user if user is not None and password_matches else None


def load_users(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"No user file found at {path}. Create accounts with dashboard/manage_users.py."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    users: dict[str, dict] = {}
    for entry in document.get("users", []):
        email = str(entry.get("email", "")).strip().lower()
        role = entry.get("role")
        if not email or role not in {"coach", "athlete"} or not entry.get("passwordHash"):
            raise ValueError(f"Invalid user record for {email or '<missing email>'}.")
        if role == "athlete" and not entry.get("athleteId"):
            raise ValueError(f"Athlete account {email} is missing athleteId.")
        users[email] = {**entry, "email": email}
    if not users:
        raise ValueError("The user file does not contain any accounts.")
    return users


def load_build() -> tuple[str, dict]:
    index_path = AUTH_DIST_DIR / "auth-index.html"
    snapshot_path = DASHBOARD_DIR / "src" / "data.json"
    if not index_path.exists():
        raise FileNotFoundError(
            "Build the authenticated dashboard with `npx vite build --config vite.auth.config.js`."
        )
    return index_path.read_text(encoding="utf-8"), json.loads(snapshot_path.read_text(encoding="utf-8"))


def _numeric(values: list[object]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def filter_snapshot_for_athlete(snapshot: dict, athlete_id: str) -> dict:
    """Create the smallest snapshot needed for one athlete's dashboard."""
    filtered = copy.deepcopy(snapshot)
    queries = filtered.get("queries", {})
    allowed = {
        "events",
        "acceleration_events",
        "acceleration_sessions",
        "acceleration_athletes",
        "session_quality",
        "sessions",
        "athletes",
    }
    for name, query in queries.items():
        if name in allowed:
            query["rows"] = [row for row in query.get("rows", []) if row.get("athleteId") == athlete_id]
        elif name != "quality":
            query["rows"] = []

    events = queries.get("events", {}).get("rows", [])
    acceleration_events = queries.get("acceleration_events", {}).get("rows", [])
    sessions = queries.get("sessions", {}).get("rows", [])
    session_quality = queries.get("session_quality", {}).get("rows", [])
    athletes = queries.get("athletes", {}).get("rows", [])
    dates = sorted({str(row["date"]) for row in events if row.get("date")})
    sync_errors = sorted(_numeric([row.get("syncErrorMs") for row in events]))
    flight_differences = sorted(_numeric([row.get("flightTimeDiffMs") for row in events]))
    session_count = len(sessions)
    jump_ids = [row.get("jumpId") for row in events]
    duplicates = len(jump_ids) - len(set(jump_ids))
    median_sync = None
    if sync_errors:
        middle = len(sync_errors) // 2
        median_sync = sync_errors[middle] if len(sync_errors) % 2 else (sync_errors[middle - 1] + sync_errors[middle]) / 2
    median_flight_difference = None
    if flight_differences:
        middle = len(flight_differences) // 2
        median_flight_difference = (
            flight_differences[middle]
            if len(flight_differences) % 2
            else (flight_differences[middle - 1] + flight_differences[middle]) / 2
        )
    original_quality = snapshot.get("queries", {}).get("quality", {}).get("rows", [{}])[0]
    athlete_quality = {
        **original_quality,
        "matchedSessions": session_count,
        "athletes": len(athletes),
        "matchedJumps": len(events),
        "athleteIdCompletenessPct": 100.0 if events and all(row.get("athleteId") == athlete_id for row in events) else 0.0,
        "duplicateJumpIds": duplicates,
        "missingMetricCells": sum(
            row.get(field) is None
            for row in events
            for field in ("jumpHeightCm", "impulseNsKg", "syncErrorMs")
        ),
        "missingAccelerationCells": sum(
            row.get(field) is None
            for row in acceleration_events
            for field in ("accelerationMagnitudeMaxG", "accelerationMagnitudeStdG", "accelerationAreaGs")
        ),
        "medianSyncErrorMs": median_sync,
        "maxSyncErrorMs": max(sync_errors) if sync_errors else None,
        "goodSyncPct": (
            100.0 * sum(abs(value) <= 100 for value in sync_errors) / len(sync_errors) if sync_errors else 0.0
        ),
        "forceEventsDetected": sum(int(row.get("forceEventsDetected") or 0) for row in session_quality),
        "imuEventsDetected": sum(int(row.get("imuEventsDetected") or 0) for row in session_quality),
        "acceptedEvents": sum(int(row.get("acceptedEvents") or 0) for row in session_quality),
        "forceEventsRejected": sum(int(row.get("forceEventsRejected") or 0) for row in session_quality),
        "imuEventsRejected": sum(int(row.get("imuEventsRejected") or 0) for row in session_quality),
        "forceEventRetentionPct": (
            100.0 * sum(int(row.get("acceptedEvents") or 0) for row in session_quality)
            / sum(int(row.get("forceEventsDetected") or 0) for row in session_quality)
            if sum(int(row.get("forceEventsDetected") or 0) for row in session_quality) else 0.0
        ),
        "imuEventRetentionPct": (
            100.0 * sum(int(row.get("acceptedEvents") or 0) for row in session_quality)
            / sum(int(row.get("imuEventsDetected") or 0) for row in session_quality)
            if sum(int(row.get("imuEventsDetected") or 0) for row in session_quality) else 0.0
        ),
        "lowForceRetentionSessions": sum(
            float(row.get("forceEventRetentionPct") or 0) < 80 for row in session_quality
        ),
        "medianFlightTimeDiffMs": median_flight_difference,
        "flightTimeDiffOver100msPct": (
            100.0 * sum(float(row.get("flightTimeDiffMs") or 0) > 100 for row in events) / len(events)
            if events else 0.0
        ),
        "flightTimeDiffOver200msPct": (
            100.0 * sum(float(row.get("flightTimeDiffMs") or 0) > 200 for row in events) / len(events)
            if events else 0.0
        ),
        "repeatedAthletes": 1 if session_count >= 2 else 0,
        "singleSessionAthletes": 1 if session_count == 1 else 0,
        "repeatCoveragePct": 100.0 if session_count >= 2 else 0.0,
        "longitudinalCoverageStatus": "available" if session_count >= 2 else "single-session",
        "dataStartDate": dates[0] if dates else None,
        "dataEndDate": dates[-1] if dates else None,
    }
    queries.setdefault("quality", {})["rows"] = [athlete_quality]
    filtered["description"] = "Private athlete performance dashboard"
    return filtered


def _safe_json_script(value: dict) -> str:
    return json.dumps(value, separators=(",", ":")).replace("<", "\\u003c")


def _login_page(error: str | None = None) -> bytes:
    error_html = f'<p id="login-error" class="error" role="alert">{html.escape(error)}</p>' if error else ""
    error_reference = ' aria-describedby="login-error"' if error else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign in — KivoMotion</title><style>
:root{{font-family:Inter,"Avenir Next",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#f8fafc;background:#03050a;color-scheme:dark}}
*{{box-sizing:border-box}}
body{{min-height:100vh;margin:0;overflow-x:hidden;background:#03050a}}
button,input{{font:inherit}}
.login-page{{position:relative;isolation:isolate;display:grid;min-height:100vh;grid-template-columns:minmax(0,1.25fr) minmax(400px,.75fr);overflow:hidden;background:radial-gradient(circle at 20% 18%,rgba(17,63,126,.30),transparent 34%),radial-gradient(circle at 82% 82%,rgba(175,40,18,.12),transparent 30%),linear-gradient(135deg,#020308 0%,#080b12 48%,#020308 100%)}}
.login-page::before{{content:"";position:absolute;inset:-20%;z-index:-2;background:repeating-linear-gradient(112deg,transparent 0 84px,rgba(255,255,255,.018) 85px,transparent 87px);animation:grid-drift 18s linear infinite;pointer-events:none}}
.login-page::after{{content:"";position:absolute;inset:0;z-index:-1;background:linear-gradient(90deg,transparent 0 54%,rgba(2,6,23,.36) 72%,rgba(2,3,8,.84) 100%),radial-gradient(ellipse at center,transparent 35%,rgba(0,0,0,.52) 100%);pointer-events:none}}
.brand-stage{{position:relative;display:grid;min-height:100vh;grid-template-rows:auto minmax(clamp(150px,32vh,420px),1fr) auto;padding:clamp(34px,6vw,86px);overflow:hidden}}
.brand-lockup{{position:relative;z-index:2;display:flex;max-width:100%;align-items:center;gap:clamp(18px,2.5vw,38px);animation:reveal-up .85s cubic-bezier(.22,1,.36,1) both}}
.brand-mark{{flex:0 1 auto;width:clamp(96px,11vw,180px);max-width:24%;height:auto;filter:drop-shadow(0 18px 34px rgba(37,99,235,.18))}}
.brand-mark .mark-core{{transform-origin:center;animation:mark-breathe 5s ease-in-out infinite}}
.wordmark{{min-width:0;max-width:100%}}
.wordmark-line{{display:flex;max-width:100%;align-items:baseline;gap:clamp(9px,1.25vw,20px);white-space:nowrap;font-size:clamp(34px,5.15vw,80px);line-height:.9;letter-spacing:.01em}}
.wordmark-line strong{{font-weight:850;letter-spacing:-.05em}}
.wordmark-line span{{font-weight:250;letter-spacing:.015em}}
.tagline{{margin:20px 0 0;color:rgba(241,245,249,.78);font-size:clamp(10px,1.1vw,17px);font-weight:500;letter-spacing:clamp(.18em,.42vw,.38em);line-height:1.7;text-transform:uppercase}}
.motion-field{{position:relative;z-index:0;align-self:stretch;width:116%;min-height:150px;margin:clamp(12px,2.5vh,32px) -8%;overflow:hidden;opacity:.72;pointer-events:none}}
.motion-field svg{{width:100%;height:100%;overflow:visible}}
.motion-path{{fill:none;stroke:url(#motion-gradient);stroke-width:2.2;stroke-linecap:round;stroke-dasharray:8 10;filter:drop-shadow(0 0 8px rgba(59,130,246,.46));animation:path-flow 10s linear infinite}}
.motion-echo{{fill:none;stroke:rgba(255,255,255,.07);stroke-width:1}}
.motion-node{{fill:#fff;filter:drop-shadow(0 0 9px #3b82f6);animation:node-pulse 2.6s ease-in-out infinite}}
.motion-node:nth-of-type(2){{animation-delay:-.8s}}.motion-node:nth-of-type(3){{animation-delay:-1.6s}}
.performance-copy{{position:relative;z-index:2;max-width:620px;animation:reveal-up .85s .16s cubic-bezier(.22,1,.36,1) both}}
.eyebrow{{display:flex;align-items:center;gap:10px;margin:0 0 16px;color:#60a5fa;font-size:11px;font-weight:750;letter-spacing:.18em;text-transform:uppercase}}
.eyebrow::before{{content:"";width:28px;height:1px;background:#3b82f6;box-shadow:0 0 12px #3b82f6}}
.performance-copy .performance-heading{{max-width:560px;margin:0;color:#f8fafc;font-size:clamp(28px,3.3vw,52px);font-weight:620;letter-spacing:-.045em;line-height:1.02}}
.performance-copy p{{max-width:500px;margin:18px 0 0;color:rgba(203,213,225,.68);font-size:14px;line-height:1.7}}
.signal-row{{display:flex;flex-wrap:wrap;gap:9px;margin-top:26px}}
.signal-row span{{display:inline-flex;align-items:center;gap:8px;padding:8px 11px;border:1px solid rgba(148,163,184,.14);border-radius:999px;background:rgba(15,23,42,.32);color:rgba(226,232,240,.72);font-size:10px;font-weight:650;letter-spacing:.08em;text-transform:uppercase;backdrop-filter:blur(12px)}}
.signal-row i{{width:6px;height:6px;border-radius:50%;background:#3b82f6;box-shadow:0 0 10px #3b82f6}}
.auth-zone{{position:relative;z-index:3;display:grid;min-height:100vh;place-items:center;padding:clamp(24px,5vw,72px)}}
.auth-card{{width:min(100%,470px);padding:clamp(28px,4vw,48px);border:1px solid rgba(255,255,255,.14);border-radius:30px;background:linear-gradient(145deg,rgba(15,23,42,.76),rgba(3,7,18,.62));box-shadow:inset 0 1px 0 rgba(255,255,255,.12),0 40px 100px rgba(0,0,0,.48),0 0 0 1px rgba(59,130,246,.04);backdrop-filter:blur(30px) saturate(132%);animation:card-enter .95s .08s cubic-bezier(.22,1,.36,1) both}}
.auth-card::before{{content:"";display:block;width:42px;height:3px;margin-bottom:28px;border-radius:99px;background:linear-gradient(90deg,#2563eb,#60a5fa,#f97316);box-shadow:0 0 22px rgba(59,130,246,.45)}}
.auth-kicker{{margin:0 0 9px;color:#60a5fa;font-size:10px;font-weight:780;letter-spacing:.18em;text-transform:uppercase}}
h1{{margin:0;color:#fff;font-size:clamp(30px,3vw,42px);font-weight:690;letter-spacing:-.045em;line-height:1}}
.sub{{margin:12px 0 30px;color:#94a3b8;font-size:14px;line-height:1.6}}
.error{{margin:0 0 18px;padding:12px 14px;border:1px solid rgba(251,113,133,.25);border-radius:12px;background:rgba(127,29,29,.28);color:#fecdd3;font-size:13px}}
label{{display:grid;gap:8px;margin-top:18px;color:#cbd5e1;font-size:12px;font-weight:680;letter-spacing:.02em}}
input{{width:100%;height:52px;padding:0 15px;border:1px solid rgba(148,163,184,.22);border-radius:13px;background:rgba(2,6,23,.58);color:#f8fafc;caret-color:#60a5fa;outline:none;transition:border-color .2s ease,box-shadow .2s ease,background .2s ease,transform .2s ease}}
input::placeholder{{color:#94a3b8}}
input:hover{{border-color:rgba(96,165,250,.42);background:rgba(15,23,42,.72)}}
input:focus{{border-color:#60a5fa;background:rgba(15,23,42,.82);box-shadow:0 0 0 4px rgba(59,130,246,.15),0 0 24px rgba(59,130,246,.08);transform:translateY(-1px)}}
.submit{{position:relative;width:100%;height:52px;margin-top:26px;overflow:hidden;border:1px solid rgba(147,197,253,.35);border-radius:13px;background:linear-gradient(110deg,#1d4ed8,#2563eb 45%,#0ea5e9);color:#fff;font-weight:760;cursor:pointer;box-shadow:0 15px 32px rgba(29,78,216,.30);transition:transform .2s ease,box-shadow .2s ease,filter .2s ease}}
.submit::before{{content:"";position:absolute;inset:-2px;transform:translateX(-110%);background:linear-gradient(100deg,transparent,rgba(255,255,255,.22),transparent);transition:transform .55s ease}}
.submit:hover{{transform:translateY(-2px);box-shadow:0 20px 38px rgba(29,78,216,.38);filter:saturate(1.12)}}.submit:hover::before{{transform:translateX(110%)}}
.submit:active{{transform:translateY(0)}}.submit:focus-visible{{outline:3px solid rgba(96,165,250,.45);outline-offset:3px}}
.access-row{{display:flex;gap:8px;margin-top:22px}}
.access-row span{{flex:1;padding:9px;border:1px solid rgba(148,163,184,.12);border-radius:11px;background:rgba(15,23,42,.42);color:#94a3b8;font-size:9px;font-weight:700;letter-spacing:.09em;text-align:center;text-transform:uppercase}}
.privacy{{margin:20px 0 0;color:#a8b5c7;font-size:10.5px;line-height:1.55}}
.privacy strong{{color:#d6deea;font-weight:650}}
@keyframes reveal-up{{from{{opacity:0;transform:translateY(24px)}}to{{opacity:1;transform:none}}}}
@keyframes card-enter{{from{{opacity:0;transform:translateX(36px) scale(.98)}}to{{opacity:1;transform:none}}}}
@keyframes mark-breathe{{0%,100%{{transform:scale(1);opacity:1}}50%{{transform:scale(1.025);opacity:.88}}}}
@keyframes node-pulse{{0%,100%{{opacity:.45;transform:scale(.75)}}50%{{opacity:1;transform:scale(1.25)}}}}
@keyframes path-flow{{to{{stroke-dashoffset:-180}}}}
@keyframes grid-drift{{to{{transform:translate3d(120px,-44px,0)}}}}
@media(max-width:1180px){{.login-page{{grid-template-columns:1fr}}.brand-stage{{min-height:440px;grid-template-rows:auto minmax(120px,1fr) auto;padding:38px 32px}}.brand-mark{{width:clamp(96px,14vw,160px)}}.wordmark-line{{font-size:clamp(38px,7vw,72px)}}.auth-zone{{min-height:auto;padding:0 24px 40px}}.login-page::after{{background:linear-gradient(180deg,transparent 0 36%,rgba(2,3,8,.92) 62%)}}.motion-field{{width:120%;min-height:120px;margin:18px -10%}}.performance-copy p{{display:none}}}}
@media(max-width:620px){{.brand-stage{{min-height:355px;grid-template-rows:auto minmax(110px,1fr) auto;padding:28px 20px}}.brand-lockup{{align-items:flex-start;gap:clamp(10px,3vw,14px)}}.brand-mark{{width:clamp(60px,22vw,82px);max-width:24%}}.wordmark-line{{gap:clamp(5px,2vw,8px);font-size:clamp(24px,8.6vw,50px)}}.tagline{{margin-top:12px;font-size:clamp(7px,2vw,8px);letter-spacing:.15em}}.motion-field{{width:130%;min-height:110px;margin:12px -15%}}.performance-copy .performance-heading{{font-size:28px}}.signal-row{{display:none}}.auth-zone{{padding:0 14px 24px}}.auth-card{{padding:28px 22px;border-radius:24px}}}}
@media(prefers-reduced-motion:reduce){{*,*::before,*::after{{scroll-behavior:auto!important;animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}}}
</style></head><body><main class="login-page">
<section class="brand-stage" aria-label="KivoMotion performance intelligence">
  <div class="brand-lockup">
    <svg class="brand-mark" viewBox="0 0 210 120" role="img" aria-label="Kivo Motion">
      <path d="M31 14 2 60l29 46 17-20-16-26 16-26Z" fill="#fff"/>
      <path class="mark-core" d="M76 8h58l35 52-35 52H76L41 60Zm10 20L64 60l22 32h38l22-32-22-32Z" fill="#fff"/>
      <path d="m179 14 29 46-29 46-17-20 16-26-16-26Z" fill="#fff"/>
    </svg>
    <div class="wordmark"><div class="wordmark-line"><strong>KIVO</strong><span>MOTION</span></div><p class="tagline">Decoding movement. Empowering athletes.</p></div>
  </div>
  <div class="motion-field" aria-hidden="true">
    <svg viewBox="0 0 900 500" preserveAspectRatio="none"><defs><linearGradient id="motion-gradient" x1="0" x2="1"><stop stop-color="#2563eb"/><stop offset=".54" stop-color="#60a5fa"/><stop offset="1" stop-color="#f97316"/></linearGradient></defs>
      <path class="motion-echo" d="M0 365C128 365 153 178 278 205s117 142 221 84 147-206 257-129 70 195 144 201"/>
      <path class="motion-echo" d="M0 397C109 397 170 262 270 280s142 87 224 44 158-154 252-98 99 140 154 141"/>
      <path class="motion-path" d="M0 365C128 365 153 178 278 205s117 142 221 84 147-206 257-129 70 195 144 201"/>
      <circle class="motion-node" cx="278" cy="205" r="5"/><circle class="motion-node" cx="499" cy="289" r="5"/><circle class="motion-node" cx="756" cy="160" r="5"/>
    </svg>
  </div>
  <div class="performance-copy"><p class="eyebrow">Performance intelligence</p><p class="performance-heading">Turn every movement into meaningful progress.</p><p>Force and IMU insights, shaped for athletes and coaches in one secure performance workspace.</p><div class="signal-row"><span><i></i>Force</span><span><i></i>Acceleration</span><span><i></i>Impulse</span></div></div>
</section>
<section class="auth-zone" aria-label="Sign in"><div class="auth-card"><p class="auth-kicker">Secure workspace</p><h1>Welcome back</h1><p class="sub">Sign in to your KivoMotion performance intelligence dashboard.</p>{error_html}
<form action="/login" method="post"{error_reference}><label>Email address<input name="email" type="email" autocomplete="username" inputmode="email" placeholder="name@kivomotion.local" required></label><label>Password<input name="password" type="password" autocomplete="current-password" placeholder="Enter your password" required></label><button class="submit" type="submit">Enter performance workspace</button></form>
<div class="access-row" aria-label="Available account types"><span>Coach access</span><span>Athlete access</span></div><p class="privacy"><strong>Privacy by role.</strong> Athlete accounts receive only their assigned performance data; coach accounts retain the complete team workspace.</p></div></section>
</main></body></html>""".encode("utf-8")


class KivoRequestHandler(BaseHTTPRequestHandler):
    server_version = "KivoMotionAuth/1.0"

    def _headers(self, status: int, content_type: str, length: int, *, cookie: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()

    def _redirect(self, location: str, *, cookie: str | None = None) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()

    def _session(self) -> dict | None:
        raw = self.headers.get("Cookie", "")
        cookie = SimpleCookie()
        cookie.load(raw)
        token = cookie.get(SESSION_COOKIE)
        if not token:
            return None
        session = self.server.sessions.get(token.value)
        if not session or session["expiresAt"] <= time.time():
            self.server.sessions.pop(token.value, None)
            return None
        return session

    def _session_cookie(self, token: str, max_age: int = SESSION_TTL_SECONDS) -> str:
        secure = "; Secure" if self.server.secure_cookies else ""
        return f"{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={max_age}{secure}"

    def _send_json(self, value: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _send_login(self, error: str | None = None, status: int = HTTPStatus.OK) -> None:
        body = _login_page(error)
        self._headers(status, "text/html; charset=utf-8", len(body))
        self.wfile.write(body)

    def _send_dashboard(self, session: dict) -> None:
        public_session = {
            "authenticated": True,
            "role": session["role"],
            "displayName": session["displayName"],
            **({"athleteId": session["athleteId"]} if session["role"] == "athlete" else {}),
        }
        snapshot = self.server.snapshot if session["role"] == "coach" else self.server.athlete_snapshots.get(session["athleteId"])
        if snapshot is None:
            snapshot = filter_snapshot_for_athlete(self.server.snapshot, session["athleteId"])
            self.server.athlete_snapshots[session["athleteId"]] = snapshot
        injection = (
            f"<script>globalThis.__KIVO_AUTH__={_safe_json_script(public_session)};"
            f"globalThis.__KIVO_SNAPSHOT__={_safe_json_script(snapshot)};</script>"
        )
        body = self.server.index_html.replace("</head>", f"{injection}</head>", 1).encode("utf-8")
        self._headers(HTTPStatus.OK, "text/html; charset=utf-8", len(body))
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        request_path = urlsplit(self.path).path
        session = self._session()
        if request_path == "/login":
            if session:
                self._redirect("/")
            else:
                self._send_login()
            return
        if not session:
            self._redirect("/login")
            return
        if request_path in {"/", "/index.html"}:
            self._send_dashboard(session)
        elif request_path == "/api/session":
            self._send_json({key: value for key, value in session.items() if key != "expiresAt"})
        elif request_path == "/api/scientist-reviews":
            if session["role"] != "coach":
                self._send_json({"error": "Coach access is required."}, HTTPStatus.FORBIDDEN)
                return
            try:
                reviews = load_scientist_reviews(self.server.scientist_reviews_path)
            except (json.JSONDecodeError, ValueError):
                self._send_json({"error": "The scientist review ledger could not be read."}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json({"reviews": reviews})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        request_path = urlsplit(self.path).path
        if request_path == "/api/scientist-reviews":
            session = self._session()
            if not session:
                self._send_json({"error": "Authentication required."}, HTTPStatus.UNAUTHORIZED)
                return
            if session["role"] != "coach":
                self._send_json({"error": "Coach access is required."}, HTTPStatus.FORBIDDEN)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_SCIENTIST_REVIEW_BODY:
                self._send_json({"error": "Invalid scientist review request."}, HTTPStatus.BAD_REQUEST)
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                with self.server.review_lock:
                    review = persist_scientist_review(
                        payload,
                        session,
                        self.server.snapshot,
                        path=self.server.scientist_reviews_path,
                    )
            except (json.JSONDecodeError, ValueError) as exc:
                self._send_json({"error": str(exc) or "Invalid scientist review request."}, HTTPStatus.BAD_REQUEST)
                return
            except Exception:
                self._send_json({"error": "The scientist review could not be saved."}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json({"review": review})
            return
        if request_path == "/api/athletes/rename":
            session = self._session()
            if not session:
                self._send_json({"error": "Authentication required."}, HTTPStatus.UNAUTHORIZED)
                return
            if session["role"] != "coach":
                self._send_json({"error": "Coach access is required."}, HTTPStatus.FORBIDDEN)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_RENAME_BODY:
                self._send_json({"error": "Invalid rename request."}, HTTPStatus.BAD_REQUEST)
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                athlete_id = str(payload.get("athleteId", "")).strip()
                with self.server.rename_lock:
                    snapshot, renamed_athlete, athlete_label = persist_athlete_rename(
                        athlete_id,
                        payload.get("renamedAthlete", ""),
                    )
                    self.server.snapshot = snapshot
                    self.server.athlete_snapshots.clear()
            except (json.JSONDecodeError, ValueError) as exc:
                self._send_json({"error": str(exc) or "Invalid rename request."}, HTTPStatus.BAD_REQUEST)
                return
            except Exception:
                self._send_json({"error": "The athlete name could not be saved."}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json({
                "athleteId": athlete_id,
                "renamedAthlete": renamed_athlete or None,
                "athleteLabel": athlete_label,
            })
            return
        if request_path == "/logout":
            raw = self.headers.get("Cookie", "")
            cookie = SimpleCookie(); cookie.load(raw)
            if token := cookie.get(SESSION_COOKIE):
                self.server.sessions.pop(token.value, None)
            self._redirect("/login", cookie=self._session_cookie("", 0))
            return
        if request_path != "/login":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_LOGIN_BODY:
            self._send_login("Invalid sign-in request.", HTTPStatus.BAD_REQUEST)
            return
        form = parse_qs(self.rfile.read(length).decode("utf-8", errors="replace"))
        email = form.get("email", [""])[0].strip().lower()
        password = form.get("password", [""])[0]
        now = time.time()
        attempt_key = (self.client_address[0], email)
        attempts = self.server.login_attempts[attempt_key]
        while attempts and attempts[0] < now - LOGIN_WINDOW_SECONDS:
            attempts.popleft()
        if len(attempts) >= MAX_LOGIN_ATTEMPTS:
            self._send_login("Too many attempts. Please wait before trying again.", HTTPStatus.TOO_MANY_REQUESTS)
            return
        user = authenticate_user(self.server.users, email, password)
        if not user:
            attempts.append(now)
            self._send_login("Email or password is incorrect.", HTTPStatus.UNAUTHORIZED)
            return
        self.server.login_attempts.pop(attempt_key, None)
        token = secrets.token_urlsafe(32)
        self.server.sessions[token] = {
            "email": user["email"],
            "displayName": user.get("displayName") or user["email"],
            "role": user["role"],
            "athleteId": user.get("athleteId"),
            "expiresAt": now + SESSION_TTL_SECONDS,
        }
        self._redirect("/", cookie=self._session_cookie(token))

    def log_message(self, message: str, *args: object) -> None:
        print(f"{self.address_string()} - {message % args}")


def serve(host: str, port: int, users_file: Path, secure_cookies: bool) -> None:
    index_html, snapshot = load_build()
    users = load_users(users_file)
    athlete_ids = {row.get("athleteId") for row in snapshot["queries"]["athletes"]["rows"]}
    unknown = [user["athleteId"] for user in users.values() if user["role"] == "athlete" and user["athleteId"] not in athlete_ids]
    if unknown:
        raise ValueError(f"Athlete account IDs are not present in the dashboard snapshot: {unknown}")
    server = ThreadingHTTPServer((host, port), KivoRequestHandler)
    server.index_html = index_html
    server.snapshot = snapshot
    server.users = users
    server.sessions = {}
    server.athlete_snapshots = {}
    server.login_attempts = defaultdict(deque)
    server.rename_lock = threading.Lock()
    server.review_lock = threading.Lock()
    server.scientist_reviews_path = SCIENTIST_REVIEWS_FILE
    server.secure_cookies = secure_cookies
    print(f"KivoMotion authenticated dashboard: http://{host}:{port}/")
    print(f"Loaded {len(users)} account(s) from {users_file}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4180)
    parser.add_argument("--users", type=Path, default=Path(os.environ.get("KIVO_USERS_FILE", DEFAULT_USERS_FILE)))
    parser.add_argument("--secure-cookies", action="store_true", help="Require HTTPS when sending the session cookie.")
    args = parser.parse_args()
    serve(args.host, args.port, args.users, args.secure_cookies)


if __name__ == "__main__":
    main()
