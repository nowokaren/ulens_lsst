"""Controlled artificial-source density validation on one Rubin image.

The experiment keeps a fixed reference source set while adding deterministic,
well-separated artificial sources elsewhere in the image.  It characterizes
global injection-density effects; it does not test deliberate local blending.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa

from .scientific_validation import (
    ControlledValidationSource,
    expected_inst_flux,
    image_condition_summaries,
    recovery_quantities,
    source_pixel,
)


@dataclass(frozen=True)
class DensityValidationDesign:
    """Reproducible configuration for a bounded density experiment."""

    source_counts: tuple[int, ...] = (9, 50, 200, 500)
    reference_source_count: int = 9
    magnitudes: tuple[float, ...] = (20.0, 22.0, 24.0)
    position_seed: int = 24680
    injection_noise_seed: int = 12345
    edge_margin_pixels: int = 100
    minimum_separation_pixels: float = 50.0

    def validate(self) -> None:
        if not self.source_counts:
            raise ValueError("source_counts cannot be empty")
        if tuple(sorted(set(self.source_counts))) != self.source_counts:
            raise ValueError("source_counts must be unique and increasing")
        if self.reference_source_count <= 0:
            raise ValueError("reference_source_count must be positive")
        if self.source_counts[0] < self.reference_source_count:
            raise ValueError("the lowest density must contain all reference sources")
        if not self.magnitudes:
            raise ValueError("magnitudes cannot be empty")


def approximate_image_area_arcmin2(exposure: Any) -> float:
    """Estimate image sky area from local WCS pixel scales at image center."""

    from lsst.geom import Point2D

    bbox = exposure.getBBox()
    center = bbox.getCenter()
    sky = exposure.getWcs().pixelToSky(center)
    sky_x = exposure.getWcs().pixelToSky(Point2D(center.getX() + 1.0, center.getY()))
    sky_y = exposure.getWcs().pixelToSky(Point2D(center.getX(), center.getY() + 1.0))
    scale_x = sky.separation(sky_x).asArcseconds()
    scale_y = sky.separation(sky_y).asArcseconds()
    return float(bbox.getArea() * scale_x * scale_y / 3600.0)


def generate_density_source_pool(
    exposure: Any,
    design: DensityValidationDesign,
) -> list[ControlledValidationSource]:
    """Generate one nested, deterministic, well-separated source pool."""

    design.validate()
    bbox = exposure.getBBox()
    rng = np.random.default_rng(design.position_seed)
    minimum_squared = design.minimum_separation_pixels**2
    positions: list[tuple[float, float]] = []
    maximum_attempts = design.source_counts[-1] * 500
    attempts = 0
    while len(positions) < design.source_counts[-1] and attempts < maximum_attempts:
        attempts += 1
        x = float(
            rng.uniform(
                bbox.getMinX() + design.edge_margin_pixels,
                bbox.getMaxX() - design.edge_margin_pixels,
            )
        )
        y = float(
            rng.uniform(
                bbox.getMinY() + design.edge_margin_pixels,
                bbox.getMaxY() - design.edge_margin_pixels,
            )
        )
        if any((x - px) ** 2 + (y - py) ** 2 < minimum_squared for px, py in positions):
            continue
        positions.append((x, y))
    if len(positions) < design.source_counts[-1]:
        raise RuntimeError(
            "Could not place the requested well-separated sources; reduce the "
            "source count or minimum separation."
        )

    sources: list[ControlledValidationSource] = []
    for index, (x, y) in enumerate(positions):
        sky = exposure.getWcs().pixelToSky(x, y)
        sources.append(
            ControlledValidationSource(
                source_id=f"density-source-{index:04d}",
                ra_deg=sky.getRa().asDegrees(),
                dec_deg=sky.getDec().asDegrees(),
                magnitude=design.magnitudes[index % len(design.magnitudes)],
                seed=design.position_seed + index,
            )
        )
    return sources


def density_validation_schema() -> pa.Schema:
    """Schema for one reference source measured at one injection density."""

    return pa.schema(
        [
            pa.field("density_validation_id", pa.string(), nullable=False),
            pa.field("source_id", pa.string(), nullable=False),
            pa.field("image_id", pa.string(), nullable=False),
            pa.field("release", pa.string()),
            pa.field("dataset_type", pa.string()),
            pa.field("visit", pa.int64()),
            pa.field("detector", pa.int64()),
            pa.field("band", pa.string()),
            pa.field("pixel_x", pa.float64()),
            pa.field("pixel_y", pa.float64()),
            pa.field("edge_distance_pixels", pa.float64()),
            pa.field("requested_mag", pa.float64()),
            pa.field("expected_inst_flux", pa.float64()),
            pa.field("n_injected_sources", pa.int64()),
            pa.field("reference_source_count", pa.int64()),
            pa.field("image_area_arcmin2", pa.float64()),
            pa.field("injection_density_per_arcmin2", pa.float64()),
            pa.field("minimum_separation_pixels", pa.float64()),
            pa.field("position_seed", pa.int64()),
            pa.field("injection_noise_seed", pa.int64()),
            pa.field("pre_meas_flux", pa.float64()),
            pa.field("pre_meas_flux_err", pa.float64()),
            pa.field("post_meas_flux", pa.float64()),
            pa.field("post_meas_flux_err", pa.float64()),
            pa.field("recovered_injected_flux", pa.float64()),
            pa.field("flux_residual", pa.float64()),
            pa.field("fractional_flux_residual", pa.float64()),
            pa.field("difference_flux_err", pa.float64()),
            pa.field("pull", pa.float64()),
            pa.field("recovered_flux_change_from_low_density", pa.float64()),
            pa.field("fractional_change_from_low_density", pa.float64()),
            pa.field("delta_sum_inst_flux", pa.float64()),
            pa.field("delta_peak_inst_flux", pa.float64()),
            pa.field("pre_measure_flag", pa.string()),
            pa.field("post_measure_flag", pa.string()),
            pa.field("native_flags_json", pa.string()),
            pa.field("injection_failure_count", pa.int64()),
            pa.field("measurement_failed", pa.bool_()),
            pa.field("level_runtime_seconds", pa.float64()),
            pa.field("psf_summary_json", pa.string()),
            pa.field("background_summary_json", pa.string()),
        ]
    )


def run_density_validation(
    image: Any,
    context: Any,
    source_pool: Sequence[ControlledValidationSource],
    design: DensityValidationDesign,
    *,
    release_name: str,
) -> list[dict[str, Any]]:
    """Measure fixed reference sources across nested injection-density levels."""

    from ulens_lsst.lsst_tools import LSSTTools

    design.validate()
    if len(source_pool) < design.source_counts[-1]:
        raise ValueError("source_pool is smaller than the highest density level")
    exposure = image.expF
    area = approximate_image_area_arcmin2(exposure)
    references = list(source_pool[: design.reference_source_count])
    tools = LSSTTools(
        image,
        injection_config={"noise_seed": int(design.injection_noise_seed)},
    )

    def catalog(sources: Sequence[ControlledValidationSource]):
        return tools.create_injection_catalog(
            ra_values=[source.ra_deg for source in sources],
            dec_values=[source.dec_deg for source in sources],
            mag_values=[source.magnitude for source in sources],
            magnification_values=[1.0] * len(sources),
            event_ids=list(range(len(sources))),
            expMidptMJD=context.mjd,
            visit=context.visit,
            detector=context.detector,
        )

    reference_catalog = catalog(references)
    pre = tools.measure_photometry(
        exposure, reference_catalog, include_native_diagnostics=True
    )
    bbox = exposure.getBBox()
    original = np.asarray(exposure.getMaskedImage().getImage().array)
    rows: list[dict[str, Any]] = []
    for source_count in design.source_counts:
        level_start = perf_counter()
        injection = tools.inject_sources(catalog(source_pool[:source_count]))
        post = tools.measure_photometry(
            injection["output_exposure"],
            reference_catalog,
            include_native_diagnostics=True,
        )
        injection_catalog = injection["output_catalog"]
        injection_failure_count = 0
        if "injection_flag" in injection_catalog.colnames:
            injection_failure_count = int(np.count_nonzero(injection_catalog["injection_flag"]))
        injected = np.asarray(
            injection["output_exposure"].getMaskedImage().getImage().array
        )
        delta = injected - original
        runtime = perf_counter() - level_start
        for index, source in enumerate(references):
            x, y = source_pixel(exposure, source)
            pre_row = pre[index]
            post_row = post[index]
            expected = expected_inst_flux(exposure.getPhotoCalib(), source.magnitude)
            quantities = recovery_quantities(
                expected,
                float(pre_row["meas_flux"]),
                float(post_row["meas_flux"]),
                float(pre_row["meas_flux_err"]),
                float(post_row["meas_flux_err"]),
            )
            delta_sum, delta_peak = _local_delta(
                delta, bbox, x, y, radius_pixels=12
            )
            psf_summary, background_summary = image_condition_summaries(
                exposure, x, y, radius_pixels=12
            )
            native_flags = {
                name: _json_scalar(post_row[name])
                for name in post.colnames
                if name.startswith("native_")
            }
            post_flag = str(post_row["measure_flag"])
            rows.append(
                {
                    "density_validation_id": (
                        f"{context.image_id}:{source.source_id}:n{source_count}"
                    ),
                    "source_id": source.source_id,
                    "image_id": context.image_id,
                    "release": release_name,
                    "dataset_type": context.dataset_type,
                    "visit": context.visit,
                    "detector": context.detector,
                    "band": context.band,
                    "pixel_x": x,
                    "pixel_y": y,
                    "edge_distance_pixels": float(
                        min(
                            x - bbox.getMinX(),
                            bbox.getMaxX() - x,
                            y - bbox.getMinY(),
                            bbox.getMaxY() - y,
                        )
                    ),
                    "requested_mag": source.magnitude,
                    "expected_inst_flux": expected,
                    "n_injected_sources": source_count,
                    "reference_source_count": design.reference_source_count,
                    "image_area_arcmin2": area,
                    "injection_density_per_arcmin2": source_count / area,
                    "minimum_separation_pixels": design.minimum_separation_pixels,
                    "position_seed": design.position_seed,
                    "injection_noise_seed": design.injection_noise_seed,
                    "pre_meas_flux": float(pre_row["meas_flux"]),
                    "pre_meas_flux_err": float(pre_row["meas_flux_err"]),
                    "post_meas_flux": float(post_row["meas_flux"]),
                    "post_meas_flux_err": float(post_row["meas_flux_err"]),
                    **quantities,
                    "delta_sum_inst_flux": delta_sum,
                    "delta_peak_inst_flux": delta_peak,
                    "pre_measure_flag": str(pre_row["measure_flag"]),
                    "post_measure_flag": post_flag,
                    "native_flags_json": json.dumps(native_flags, sort_keys=True),
                    "injection_failure_count": injection_failure_count,
                    "measurement_failed": bool(post_flag),
                    "level_runtime_seconds": runtime,
                    "psf_summary_json": json.dumps(psf_summary, sort_keys=True),
                    "background_summary_json": json.dumps(
                        background_summary, sort_keys=True
                    ),
                }
            )
    return add_low_density_comparisons(rows)


def add_low_density_comparisons(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add same-source flux changes relative to the lowest density level."""

    if not rows:
        return []
    lowest = min(int(row["n_injected_sources"]) for row in rows)
    reference = {
        (row["image_id"], row["source_id"]): float(row["recovered_injected_flux"])
        for row in rows
        if int(row["n_injected_sources"]) == lowest
    }
    result: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        low_flux = reference[(row["image_id"], row["source_id"])]
        change = float(row["recovered_injected_flux"]) - low_flux
        row["recovered_flux_change_from_low_density"] = change
        row["fractional_change_from_low_density"] = (
            change / low_flux if low_flux and isfinite(low_flux) else None
        )
        result.append(row)
    return result


def _local_delta(
    delta: np.ndarray,
    bbox: Any,
    x: float,
    y: float,
    *,
    radius_pixels: int,
) -> tuple[float, float]:
    ix = int(round(x)) - int(bbox.getMinX())
    iy = int(round(y)) - int(bbox.getMinY())
    y0 = max(0, iy - radius_pixels)
    y1 = min(delta.shape[0], iy + radius_pixels + 1)
    x0 = max(0, ix - radius_pixels)
    x1 = min(delta.shape[1], ix + radius_pixels + 1)
    cutout = delta[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - ix) ** 2 + (yy - iy) ** 2 <= radius_pixels**2
    values = cutout[mask]
    return float(np.sum(values, dtype=np.float64)), float(np.max(values))


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)
