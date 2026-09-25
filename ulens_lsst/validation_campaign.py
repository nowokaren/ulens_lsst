"""Explicit command for bounded real-RSP scientific validation campaigns."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .density_validation import (
    DensityValidationDesign,
    density_validation_schema,
    generate_density_source_pool,
    run_density_validation,
)
from .image_context import ImageContext
from .validation_analysis import (
    plot_density_validation,
    summarize_density_validation,
)
from .work_units import make_image_id


def run_density_campaign(
    release_name: str,
    output_dir: str | Path,
    *,
    design: DensityValidationDesign | None = None,
    where: str = "",
) -> dict[str, Any]:
    """Run one bounded density experiment using a resolved release profile."""

    from data_release_config import get_data_release
    from .lsst_data import LSSTData

    design = design or DensityValidationDesign()
    design.validate()
    release = get_data_release(release_name)
    dataset_type = tuple(release.calexp_dataset_types)[0]
    data = LSSTData(release_config=release, main_path=str(output_dir))
    ref = _first_image_ref(data, dataset_type, release.butler_collections, where)
    data_id = {
        key: ref.dataId[key]
        for key in ref.datasetType.dimensions.names
    }
    image_id = make_image_id(release.name, dataset_type, data_id)
    context = ImageContext(
        image_id=image_id,
        dataset_type=dataset_type,
        data_id=data_id,
        visit=int(data_id["visit"]),
        detector=int(data_id["detector"]),
        band=str(data_id["band"]),
        mjd=0.0,
    )
    image = data.load_image(context)
    source_pool = generate_density_source_pool(image.expF, design)
    started = datetime.now(timezone.utc)
    rows = run_density_validation(
        image,
        context,
        source_pool,
        design,
        release_name=release.name,
    )
    ended = datetime.now(timezone.utc)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=density_validation_schema())
    parquet_path = output_dir / "density-validation.parquet"
    temporary = output_dir / ".density-validation.parquet.tmp"
    pq.write_table(table, temporary, compression="zstd")
    pq.read_table(temporary)
    temporary.replace(parquet_path)

    density_summary = summarize_density_validation(rows)
    summary = {
        "release": release.name,
        "image_id": image_id,
        "dataset_type": dataset_type,
        "band": context.band,
        "visit": context.visit,
        "detector": context.detector,
        **density_summary,
    }
    _write_json(output_dir / "summary.json", summary)
    plot_density_validation(rows, output_dir / "figures")

    provenance = {
        "validation_type": "global_injection_density",
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "release": release.name,
        "image_id": image_id,
        "dataset_type": dataset_type,
        "data_id": data_id,
        "butler_repo": str(release.butler_repo),
        "butler_collections": _jsonable(release.butler_collections),
        "design": {
            "source_counts": list(design.source_counts),
            "reference_source_count": design.reference_source_count,
            "magnitudes": list(design.magnitudes),
            "position_seed": design.position_seed,
            "injection_noise_seed": design.injection_noise_seed,
            "edge_margin_pixels": design.edge_margin_pixels,
            "minimum_separation_pixels": design.minimum_separation_pixels,
        },
        "ulens_lsst_git_commit": _git_value("rev-parse", "HEAD"),
        "ulens_lsst_git_dirty": bool(_git_value("status", "--porcelain")),
        "rubin_stack": _command_value("eups", "list", "lsst_distrib", "-s"),
        "python": platform.python_version(),
        "command_environment": {
            "ULENS_LSST_RSP_RELEASE": os.environ.get("ULENS_LSST_RSP_RELEASE"),
        },
    }
    _write_json(output_dir / "provenance.json", provenance)
    return summary


def _first_image_ref(data: Any, dataset_type: str, collections: Any, where: str):
    visit = next(
        iter(
            data.butler.registry.queryDataIds(
                ["visit"],
                datasets=dataset_type,
                collections=collections,
                where=where,
            )
        ),
        None,
    )
    if visit is None:
        raise RuntimeError("The resolved release returned no processable image visit.")
    visit_id = int(visit["visit"])
    data_id = next(
        iter(
            data.butler.registry.queryDataIds(
                ["visit", "detector"],
                datasets=dataset_type,
                collections=collections,
                where=f"visit={visit_id}",
            )
        ),
        None,
    )
    if data_id is None:
        raise RuntimeError("The selected visit returned no detector data ID.")
    ref = next(
        iter(
            data.butler.registry.queryDatasets(
                dataset_type,
                collections=collections,
                dataId=data_id,
                findFirst=True,
            )
        ),
        None,
    )
    if ref is None:
        raise RuntimeError("The resolved release returned no image DatasetRef.")
    return ref


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _git_value(*arguments: str) -> str:
    repository = Path(__file__).resolve().parents[1]
    return _command_value("git", "-C", str(repository), *arguments)


def _command_value(*arguments: str) -> str:
    result = subprocess.run(arguments, check=False, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded controlled injection-density validation."
    )
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--counts", default="9,50,200,500")
    parser.add_argument("--reference-count", default=9, type=int)
    parser.add_argument("--magnitudes", default="20,22,24")
    parser.add_argument("--position-seed", default=24680, type=int)
    parser.add_argument("--injection-noise-seed", default=12345, type=int)
    parser.add_argument("--where", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    design = DensityValidationDesign(
        source_counts=tuple(int(item) for item in args.counts.split(",")),
        reference_source_count=args.reference_count,
        magnitudes=tuple(float(item) for item in args.magnitudes.split(",")),
        position_seed=args.position_seed,
        injection_noise_seed=args.injection_noise_seed,
    )
    summary = run_density_campaign(
        args.release,
        args.output,
        design=design,
        where=args.where,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
