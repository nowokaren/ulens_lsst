"""Deterministic identities and output paths for image-processing runs.

These helpers do not execute scientific processing.  They define stable names
that are independent of worker assignment, scheduler identifiers, retries,
and execution order.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


def make_image_id(
    release_name: str,
    dataset_type: str,
    data_id: Mapping[str, object],
) -> str:
    """Create a stable identifier for one Rubin image dataset.

    The release and dataset type are included because the same visit/detector
    pair can refer to different products in different processing contexts.
    Data-ID keys are serialized in sorted order so dictionary insertion order
    cannot change the identifier.
    """
    if not release_name or not dataset_type or not data_id:
        raise ValueError("release_name, dataset_type, and data_id are required")

    canonical_data_id = ",".join(
        f"{key}={data_id[key]}" for key in sorted(data_id)
    )
    return f"{release_name}:{dataset_type}:{canonical_data_id}"


def make_injection_id(run_id: str, event_id: object, image_id: str) -> str:
    """Create a stable identifier for one event-image injection."""
    if not run_id or not image_id:
        raise ValueError("run_id and image_id are required")
    return f"{run_id}:{event_id}:{image_id}"


def make_work_unit_id(image_ids: Sequence[str]) -> str:
    """Create a stable identifier for a deterministic group of images."""
    ordered_image_ids = sorted(image_ids)
    if not ordered_image_ids or any(not image_id for image_id in ordered_image_ids):
        raise ValueError("image_ids must contain at least one non-empty ID")

    payload = json.dumps(ordered_image_ids, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"wu-{digest}"


def work_unit_product_path(
    run_dir: str | Path,
    product_name: str,
    work_unit_id: str,
) -> Path:
    """Return the independent Parquet path for one work-unit product."""
    if not product_name or not work_unit_id:
        raise ValueError("product_name and work_unit_id are required")
    if Path(work_unit_id).name != work_unit_id:
        raise ValueError("work_unit_id must be a simple filename-safe identifier")
    return Path(run_dir) / product_name / f"part-{work_unit_id}.parquet"
