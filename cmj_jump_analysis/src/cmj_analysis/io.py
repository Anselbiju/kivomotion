from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Recording:
    data: pd.DataFrame
    metadata: dict[str, str]
    source: Path


def _clean_key(value: str) -> str:
    return value.strip().lstrip("\ufeff")


def read_force_csv(path: str | Path) -> Recording:
    """Read a ForceDecks raw export, including its metadata preamble."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_idx = next(
        (n for n, line in enumerate(lines) if line.strip().startswith("Time,")), None
    )
    if header_idx is None:
        raise ValueError(f"Could not find ForceDecks data header in {path}")

    metadata: dict[str, str] = {}
    for line in lines[:header_idx]:
        if "," in line:
            key, value = line.split(",", 1)
            metadata[_clean_key(key)] = value.strip().strip('"')

    data = pd.read_csv(path, skiprows=header_idx)
    required = {"Time", "Left", "Right", "Both"}
    if not required.issubset(data.columns):
        raise ValueError(f"Force file is missing {required - set(data.columns)}")
    data = data.apply(pd.to_numeric, errors="coerce").dropna(subset=list(required))
    data = data.sort_values("Time").drop_duplicates("Time").reset_index(drop=True)
    return Recording(data=data, metadata=metadata, source=path)


def read_imu_csv(path: str | Path) -> Recording:
    """Read the IMU CSV and repair duplicate/out-of-order timestamps."""
    path = Path(path)
    metadata: dict[str, str] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            match = re.match(r"#\s*([^=]+)=(.*)", line.strip())
            if match:
                metadata[match.group(1).strip()] = match.group(2).strip()

    data = pd.read_csv(path, comment="#")
    required = {"t_ms", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"}
    if not required.issubset(data.columns):
        raise ValueError(f"IMU file is missing {required - set(data.columns)}")
    data = data.apply(pd.to_numeric, errors="coerce").dropna(subset=list(required))
    # The sample has duplicate and slightly out-of-order packets. Averaging equal
    # timestamps is safer than silently keeping whichever packet appears first.
    data = data.groupby("t_ms", as_index=False).mean(numeric_only=True)
    data = data.sort_values("t_ms").reset_index(drop=True)
    data["time_s"] = data["t_ms"] / 1000.0
    return Recording(data=data, metadata=metadata, source=path)


_ATHLETE_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def matching_athlete_id(
    force_recording: Recording,
    imu_recording: Recording,
) -> str:
    """Return the shared UUID after validating both recording metadata blocks."""
    force_id = force_recording.metadata.get("AthleteId", "").strip()
    imu_id = imu_recording.metadata.get("athleteId", "").strip()
    if not force_id:
        raise ValueError(f"Force file has no AthleteId: {force_recording.source}")
    if not imu_id:
        raise ValueError(f"IMU file has no athleteId: {imu_recording.source}")
    if not _ATHLETE_ID_PATTERN.fullmatch(force_id):
        raise ValueError(f"Force file has an invalid AthleteId: {force_id}")
    if not _ATHLETE_ID_PATTERN.fullmatch(imu_id):
        raise ValueError(f"IMU file has an invalid athleteId: {imu_id}")
    if force_id.casefold() != imu_id.casefold():
        raise ValueError(
            "Force/IMU athlete ID mismatch: "
            f"{force_recording.source.name}={force_id}, "
            f"{imu_recording.source.name}={imu_id}"
        )
    return force_id


def parse_iso_utc(value: str) -> datetime:
    value = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def nominal_clock_offset_s(
    force_recording: Recording,
    imu_recording: Recording,
    force_timezone: str = "Australia/Brisbane",
) -> float | None:
    """Map IMU elapsed seconds to force elapsed seconds from file timestamps.

    ForceDecks stores a local timestamp, while the IMU stores UTC. The force
    timestamp has only one-second precision, so this is a pairing prior rather
    than the final synchronization value.
    """
    force_value = force_recording.metadata.get("Recording Date")
    imu_value = imu_recording.metadata.get("startUtc")
    if not force_value or not imu_value:
        return None
    force_start = datetime.fromisoformat(force_value)
    if force_start.tzinfo is None:
        force_start = force_start.replace(tzinfo=ZoneInfo(force_timezone))
    imu_start = parse_iso_utc(imu_value)
    return float((imu_start.astimezone(timezone.utc) - force_start.astimezone(timezone.utc)).total_seconds())


def estimate_sampling_rate(time_s: pd.Series | np.ndarray) -> float:
    t = np.asarray(time_s, dtype=float)
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    if not len(diffs):
        raise ValueError("At least two increasing timestamps are required")
    return float(1.0 / np.median(diffs))
