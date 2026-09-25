"""Independent PROCESS execution for prepared image-processing runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import tempfile
import time
import traceback
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from .image_context import ImageContext
from .image_processors import get_image_processor
from .release_config import resolve_configured_image_processing
from .work_units import make_injection_id, work_unit_product_path


STATUS_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_path(run_dir: Path, work_unit_id: str) -> Path:
    return run_dir / "status" / f"part-{work_unit_id}.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config_file.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Prepared run is missing its configuration snapshot: {config_path}"
        )
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Run configuration must contain a mapping: {config_path}")
    return config


def _read_work_unit(run_dir: Path, work_unit_id: str) -> list[str]:
    path = run_dir / "work_units.parquet"
    table = pq.read_table(path, columns=["work_unit_id", "image_ids"])
    matches = [
        row
        for row in table.to_pylist()
        if row["work_unit_id"] == work_unit_id
    ]
    if len(matches) != 1:
        raise KeyError(f"Expected one work unit {work_unit_id!r}, found {len(matches)}")
    image_ids = json.loads(matches[0]["image_ids"])
    if not isinstance(image_ids, list) or not all(isinstance(value, str) for value in image_ids):
        raise ValueError(f"Invalid image_ids for work unit {work_unit_id!r}")
    return image_ids


def _read_image_rows(run_dir: Path, image_ids: Iterable[str]) -> list[dict[str, Any]]:
    path = run_dir / "images.parquet"
    image_ids = list(image_ids)
    table = ds.dataset(path, format="parquet").to_table(
        filter=ds.field("image_id").isin(image_ids)
    )
    rows = table.to_pylist()
    if {row["image_id"] for row in rows} != set(image_ids):
        found = {row["image_id"] for row in rows}
        missing = sorted(set(image_ids) - found)
        raise KeyError(f"Images missing from images.parquet: {missing}")
    return sorted(rows, key=lambda row: row["image_id"])


def _find_product(run_dir: Path, directory: str, pattern: str) -> Path:
    candidates = sorted((run_dir / directory).glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No {directory}/{pattern} product found in {run_dir}")
    return candidates[-1]


def _source_rows(run_dir: Path, image_row: dict[str, Any]) -> pd.DataFrame:
    """Load only source rows relevant to one image."""
    photometry_path = _find_product(run_dir, ".", "photometry_*.parquet")
    columns = ["event_id", "time", "band", "ideal_mag", "magnification"]
    source_dataset = ds.dataset(photometry_path, format="parquet")
    expression = ds.field("band") == image_row["band"]
    if image_row.get("mjd") is not None:
        tolerance = 1e-5
        expression = expression & (
            (ds.field("time") >= image_row["mjd"] - tolerance)
            & (ds.field("time") <= image_row["mjd"] + tolerance)
        )
    source_table = source_dataset.to_table(columns=columns, filter=expression)
    source = source_table.to_pandas()
    if source.empty:
        return source

    events_path = _find_product(run_dir, "events", "*.parquet") if (run_dir / "events").exists() else _find_product(run_dir, ".", "data-events_*.parquet")
    event_ids = source["event_id"].dropna().unique().tolist()
    events = ds.dataset(events_path, format="parquet").to_table(
        columns=["event_id", "ra", "dec"],
        filter=ds.field("event_id").isin(event_ids),
    ).to_pandas()
    return source.merge(events, on="event_id", how="inner")


def _context(row: dict[str, Any]) -> ImageContext:
    return ImageContext(
        image_id=row["image_id"],
        dataset_type=row["dataset_type"],
        data_id=json.loads(row["data_id"]),
        visit=row.get("visit"),
        detector=row.get("detector"),
        band=row.get("band"),
        mjd=row.get("mjd"),
        seeing=row.get("seeing"),
        limiting_depth=row.get("limiting_depth"),
    )


def _as_dataframe(product: Any) -> pd.DataFrame:
    if isinstance(product, pd.DataFrame):
        return product.copy()
    if hasattr(product, "to_pandas"):
        return product.to_pandas()
    if hasattr(product, "as_array"):
        return pd.DataFrame(product.as_array())
    raise TypeError(f"Unsupported processor product type: {type(product).__name__}")


def _create_image_data(selection: Any, config: dict[str, Any], run_dir: Path) -> Any:
    """Construct release-aware image access for one PROCESS invocation."""
    from .lsst_data import LSSTData

    return LSSTData(
        release_config=selection.release_config,
        bands=config.get("bands") or ["u", "g", "r", "i", "z", "y"],
        main_path=str(run_dir),
    )


def _scientific_rows(
    result: Any,
    sources: pd.DataFrame,
    context: ImageContext,
    run_id: str,
) -> pd.DataFrame:
    rows = _as_dataframe(result.products["photometry"])
    if rows.empty:
        return rows
    rows["image_id"] = context.image_id
    rows["visit"] = context.visit
    rows["detector"] = context.detector
    rows["band"] = context.band
    rows["eligible"] = True
    rows["attempted"] = True
    injection_flags = rows.get(
        "injection_flag", pd.Series(False, index=rows.index)
    ).fillna(False).astype(bool)
    rows["injection_status"] = np.where(injection_flags, "failed", "success")
    measurement_flags = rows.get(
        "measure_flag", pd.Series("", index=rows.index)
    ).fillna("").astype(str)
    rows["measurement_status"] = np.where(
        measurement_flags.eq(""),
        "success",
        "failed",
    )
    rows["processing_status"] = "scientific_result"
    rows["injection_id"] = [
        make_injection_id(run_id, event_id, context.image_id)
        for event_id in rows["event_id"]
    ]
    rows["injected_mag"] = rows.get("ideal_mag")
    rows["injected_flux"] = np.nan
    return rows


def _failure_rows(
    sources: pd.DataFrame,
    context: ImageContext,
    run_id: str,
    error: str,
) -> pd.DataFrame:
    rows = pd.DataFrame({"event_id": sources["event_id"].tolist()})
    rows["image_id"] = context.image_id
    rows["visit"] = context.visit
    rows["detector"] = context.detector
    rows["band"] = context.band
    rows["time"] = context.mjd
    rows["ideal_mag"] = sources.get("ideal_mag")
    rows["magnification"] = sources.get("magnification")
    rows["eligible"] = True
    rows["attempted"] = True
    rows["injection_status"] = "failed"
    rows["measurement_status"] = "not_attempted"
    rows["processing_status"] = "scientific_failure"
    rows["scientific_error"] = error
    rows["injection_id"] = [
        make_injection_id(run_id, event_id, context.image_id)
        for event_id in rows["event_id"]
    ]
    return rows


def _write_photometry_atomic(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    os.close(fd)
    try:
        pq.write_table(table, temporary_path, compression="snappy")
        check = pq.read_table(temporary_path)
        if check.num_rows:
            keys = check.select(["event_id", "image_id"]).to_pandas()
            if keys.duplicated().any():
                raise ValueError("Duplicate event_id/image_id keys in work-unit output")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def process_work_unit(
    run_dir: str | Path,
    work_unit_id: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Process one deterministic work unit and atomically persist its products."""
    run_dir = Path(run_dir)
    status_path = _status_path(run_dir, work_unit_id)
    output_path = work_unit_product_path(run_dir, "photometry", work_unit_id)
    started = time.time()
    status = {
        "status_version": STATUS_VERSION,
        "work_unit_id": work_unit_id,
        "state": "running",
        "started_at": _utc_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    _write_json_atomic(status_path, status)

    try:
        if not force and output_path.exists():
            pq.read_table(output_path, columns=["event_id", "image_id"])
            status.update({"state": "completed", "skipped": True, "ended_at": _utc_now()})
            _write_json_atomic(status_path, status)
            return status

        config = _read_config(run_dir)
        selection = resolve_configured_image_processing(config)
        if selection is None:
            raise ValueError("The prepared run has no release-aware image processor configuration")
        image_rows = _read_image_rows(run_dir, _read_work_unit(run_dir, work_unit_id))
        data = _create_image_data(selection, config, run_dir)
        processor = get_image_processor(selection.processor_name)
        run_id = str(config.get("name", run_dir.name))
        products = []
        for image_row in image_rows:
            context = _context(image_row)
            sources = _source_rows(run_dir, image_row)
            if sources.empty:
                continue
            try:
                image = data.load_image(context)
                result = processor.process(
                    image,
                    context,
                    sources,
                    config={
                        "injection_config": config.get("injection_config", {}),
                        "measurement_config": config.get("measurement_config", {}),
                    },
                )
                products.append(_scientific_rows(result, sources, context, run_id))
            except Exception:
                products.append(
                    _failure_rows(sources, context, run_id, traceback.format_exc())
                )

        result_frame = pd.concat(products, ignore_index=True) if products else pd.DataFrame()
        if result_frame.empty:
            table = pa.table(
                {
                    "event_id": pa.array([], type=pa.int64()),
                    "image_id": pa.array([], type=pa.string()),
                }
            )
        else:
            table = pa.Table.from_pandas(result_frame, preserve_index=False)
        _write_photometry_atomic(table, output_path)
        state = "empty_success" if result_frame.empty else "completed"
        status.update(
            {
                "state": state,
                "ended_at": _utc_now(),
                "elapsed_seconds": time.time() - started,
                "n_images": len(image_rows),
                "n_photometry_rows": len(result_frame),
                "output_path": str(output_path),
            }
        )
        _write_json_atomic(status_path, status)
        return status
    except Exception as exc:
        status.update(
            {
                "state": "failed",
                "ended_at": _utc_now(),
                "elapsed_seconds": time.time() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json_atomic(status_path, status)
        raise
