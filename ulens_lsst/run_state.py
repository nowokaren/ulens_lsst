"""Operational state, restart planning, and lightweight finalization."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

import pyarrow.parquet as pq

from .work_units import work_unit_product_path


ACCEPTED_STATES = {"completed", "empty_success"}
RESTART_STATES = {"pending", "failed", "stale", "invalid"}


def _status_path(run_dir: Path, work_unit_id: str) -> Path:
    return run_dir / "status" / f"part-{work_unit_id}.json"


def _read_expected(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows = pq.read_table(
        run_dir / "work_units.parquet",
        columns=["work_unit_id", "image_ids", "n_images"],
    ).to_pylist()
    return {row["work_unit_id"]: row for row in rows}


def validate_photometry_part(path: Path) -> dict[str, Any]:
    """Validate one deterministic scientific output part without pandas."""
    required = {"event_id", "image_id"}
    metadata = pq.read_metadata(path)
    names = set(metadata.schema.names)
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Missing required photometry columns: {missing}")

    rows = 0
    seen: set[tuple[Any, Any]] = set()
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=["event_id", "image_id"]):
        for row in batch.to_pylist():
            if row["event_id"] is None or row["image_id"] is None:
                raise ValueError(f"Null scientific key in {path}: {row}")
            key = (row["event_id"], row["image_id"])
            if key in seen:
                raise ValueError(f"Duplicate scientific key in {path}: {key}")
            seen.add(key)
            rows += 1
    return {"rows": rows, "columns": sorted(names)}


def classify_work_unit(
    run_dir: str | Path,
    work_unit_id: str,
    *,
    stale_after_seconds: float = 6 * 3600,
) -> str:
    """Classify one work unit from its durable state and output part."""
    run_dir = Path(run_dir)
    status_path = _status_path(run_dir, work_unit_id)
    output_path = work_unit_product_path(run_dir, "photometry", work_unit_id)
    if not status_path.exists():
        return "pending"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        state = status.get("state")
    except Exception:
        return "invalid"

    if state == "running":
        started_at = status.get("started_at")
        try:
            started = datetime.fromisoformat(started_at).timestamp()
        except Exception:
            return "stale"
        return "stale" if time.time() - started > stale_after_seconds else "running"
    if state in ACCEPTED_STATES:
        if not output_path.exists():
            return "invalid"
        try:
            validate_photometry_part(output_path)
        except Exception:
            return "invalid"
        return state
    if state == "failed":
        return "failed"
    return "invalid"


def status_summary(
    run_dir: str | Path,
    *,
    stale_after_seconds: float = 6 * 3600,
) -> dict[str, Any]:
    """Return a compact operational summary without scanning all science rows."""
    run_dir = Path(run_dir)
    expected = _read_expected(run_dir)
    counts = {state: 0 for state in ("pending", "running", "completed", "failed", "stale", "invalid", "empty_success")}
    images_expected = 0
    images_completed = 0
    photometry_rows = 0
    elapsed_values = []
    for work_unit_id, row in expected.items():
        state = classify_work_unit(
            run_dir, work_unit_id, stale_after_seconds=stale_after_seconds
        )
        counts[state] += 1
        images_expected += int(row["n_images"])
        status_path = _status_path(run_dir, work_unit_id)
        if state in ACCEPTED_STATES:
            images_completed += int(row["n_images"])
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            photometry_rows += int(payload.get("n_photometry_rows", 0))
            if payload.get("elapsed_seconds") is not None:
                elapsed_values.append(float(payload["elapsed_seconds"]))
    return {
        "total_work_units": len(expected),
        **counts,
        "images_expected": images_expected,
        "images_completed": images_completed,
        "photometry_rows": photometry_rows,
        "mean_work_unit_seconds": (
            sum(elapsed_values) / len(elapsed_values) if elapsed_values else None
        ),
    }


def pending_work_units(
    run_dir: str | Path,
    *,
    stale_after_seconds: float = 6 * 3600,
) -> list[str]:
    """Return deterministic work units that should be submitted again."""
    expected = _read_expected(Path(run_dir))
    return [
        work_unit_id
        for work_unit_id in sorted(expected)
        if classify_work_unit(
            run_dir, work_unit_id, stale_after_seconds=stale_after_seconds
        )
        in RESTART_STATES
    ]


def finalize(run_dir: str | Path) -> dict[str, Any]:
    """Validate completion and cross-part scientific-key uniqueness."""
    run_dir = Path(run_dir)
    expected = _read_expected(run_dir)
    summary = status_summary(run_dir)
    incomplete = [
        work_unit_id
        for work_unit_id in expected
        if classify_work_unit(run_dir, work_unit_id) not in ACCEPTED_STATES
    ]
    duplicate_keys = 0
    key_db = run_dir / ".finalize-scientific-keys.sqlite"
    try:
        with sqlite3.connect(key_db) as connection:
            connection.execute(
                "CREATE TABLE keys (event_id TEXT NOT NULL, image_id TEXT NOT NULL, "
                "PRIMARY KEY (event_id, image_id))"
            )
            for work_unit_id in expected:
                if classify_work_unit(run_dir, work_unit_id) not in ACCEPTED_STATES:
                    continue
                path = work_unit_product_path(run_dir, "photometry", work_unit_id)
                for batch in pq.ParquetFile(path).iter_batches(
                    columns=["event_id", "image_id"]
                ):
                    for row in batch.to_pylist():
                        try:
                            connection.execute(
                                "INSERT INTO keys VALUES (?, ?)",
                                (str(row["event_id"]), str(row["image_id"])),
                            )
                        except sqlite3.IntegrityError:
                            duplicate_keys += 1
            connection.commit()
    finally:
        key_db.unlink(missing_ok=True)

    summary.update(
        {
            "complete": not incomplete and duplicate_keys == 0,
            "incomplete_work_units": sorted(incomplete),
            "duplicate_scientific_keys": duplicate_keys,
        }
    )
    return summary
