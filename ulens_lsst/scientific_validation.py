"""Scientific validation helpers for controlled injected sources.

This module is deliberately separate from the production photometry product.
It records the calibration and before/after diagnostics needed to determine
whether a known injected signal is recovered sensibly.  It does not define a
microlensing detection or completeness criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from math import hypot, isfinite, sqrt
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ControlledValidationSource:
    """One deterministic source used for a validation experiment.

    ``magnitude`` is the instantaneous source magnitude passed to
    ``VisitInjectTask``.  The expected flux is calculated from the image's
    ``PhotoCalib`` in that image's instrumental-flux units.
    """

    source_id: str
    ra_deg: float
    dec_deg: float
    magnitude: float
    seed: int = 0


def expected_inst_flux(photo_calib: Any, magnitude: float) -> float:
    """Convert magnitude to Rubin instrumental flux using the image PhotoCalib."""

    value = photo_calib.magnitudeToInstFlux(float(magnitude))
    # PhotoCalib returns a scalar for a scalar input in the installed RSP.
    flux = float(getattr(value, "value", value))
    if not isfinite(flux):
        raise ValueError(f"PhotoCalib returned non-finite flux for magnitude {magnitude}")
    return flux


def source_pixel(exposure: Any, source: ControlledValidationSource) -> tuple[float, float]:
    """Return the expected pixel position from the exposure WCS."""

    point = exposure.getWcs().skyToPixel(
        _sphere_point(source.ra_deg, source.dec_deg)
    )
    return float(point.getX()), float(point.getY())


def _sphere_point(ra_deg: float, dec_deg: float) -> Any:
    # Keep the import lazy so pure calculation tests do not require RSP.
    from lsst.geom import Angle, SpherePoint, degrees

    return SpherePoint(Angle(float(ra_deg), degrees), Angle(float(dec_deg), degrees))


def image_delta_diagnostic(
    original_exposure: Any,
    injected_exposure: Any,
    x: float,
    y: float,
    *,
    radius_pixels: int = 12,
) -> dict[str, float | int | None]:
    """Summarize the local image difference around a controlled source.

    The local sum is an image-level diagnostic, not the primary flux estimate:
    PSF-weighted forced photometry is used for the scientific recovery
    residual.  The diagnostic is retained to catch sign and placement errors
    and to expose the Rubin image representation to later investigation.
    """

    original = np.asarray(original_exposure.getMaskedImage().getImage().array)
    injected = np.asarray(injected_exposure.getMaskedImage().getImage().array)
    delta = injected - original
    bbox = original_exposure.getBBox()
    ix = int(round(x)) - int(bbox.getMinX())
    iy = int(round(y)) - int(bbox.getMinY())
    yy, xx = np.ogrid[: delta.shape[0], : delta.shape[1]]
    mask = (xx - ix) ** 2 + (yy - iy) ** 2 <= radius_pixels ** 2
    if not np.any(mask):
        return {
            "delta_sum_inst_flux": None,
            "delta_peak_inst_flux": None,
            "delta_pixels": 0,
        }
    values = delta[mask]
    return {
        "delta_sum_inst_flux": float(np.sum(values, dtype=np.float64)),
        "delta_peak_inst_flux": float(np.max(values)),
        "delta_pixels": int(values.size),
    }


def image_condition_summaries(
    exposure: Any,
    x: float,
    y: float,
    *,
    radius_pixels: int = 12,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract bounded local background/noise and optional PSF summaries."""

    image = np.asarray(exposure.getMaskedImage().getImage().array)
    variance = np.asarray(exposure.getMaskedImage().getVariance().array)
    bbox = exposure.getBBox()
    ix = int(round(x)) - int(bbox.getMinX())
    iy = int(round(y)) - int(bbox.getMinY())
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = (xx - ix) ** 2 + (yy - iy) ** 2 <= radius_pixels ** 2
    values = image[mask]
    variances = variance[mask]
    background = {
        "local_image_median": float(np.nanmedian(values)),
        "local_image_std": float(np.nanstd(values)),
        "local_variance_median": float(np.nanmedian(variances)),
    }
    psf_summary: dict[str, Any] = {}
    try:
        psf = exposure.getPsf()
        position = psf.getAveragePosition()
        shape = psf.computeShape(position)
        psf_summary = {
            "average_x": float(position.getX()),
            "average_y": float(position.getY()),
            "determinant_radius": float(shape.getDeterminantRadius()),
        }
    except Exception as exc:  # pragma: no cover - RSP-version dependent
        psf_summary = {"error": type(exc).__name__}
    return psf_summary, background


def recovery_quantities(
    expected_flux: float,
    pre_flux: float,
    post_flux: float,
    pre_flux_err: float | None = None,
    post_flux_err: float | None = None,
) -> dict[str, float | None]:
    """Calculate baseline-subtracted injected-flux residual quantities."""

    recovered = float(post_flux) - float(pre_flux)
    residual = recovered - float(expected_flux)
    fractional = residual / float(expected_flux) if expected_flux else None
    difference_err = None
    if pre_flux_err is not None and post_flux_err is not None:
        if isfinite(float(pre_flux_err)) and isfinite(float(post_flux_err)):
            difference_err = hypot(float(pre_flux_err), float(post_flux_err))
    pull = residual / difference_err if difference_err and difference_err > 0 else None
    return {
        "recovered_injected_flux": recovered,
        "flux_residual": residual,
        "fractional_flux_residual": fractional,
        "difference_flux_err": difference_err,
        "pull": pull,
    }


def validation_schema() -> pa.Schema:
    """Return the stable schema for controlled validation rows."""

    return pa.schema(
        [
            pa.field("validation_id", pa.string(), nullable=False),
            pa.field("source_id", pa.string(), nullable=False),
            pa.field("image_id", pa.string(), nullable=False),
            pa.field("release", pa.string()),
            pa.field("visit", pa.int64()),
            pa.field("detector", pa.int64()),
            pa.field("band", pa.string()),
            pa.field("mjd", pa.float64()),
            pa.field("ra_deg", pa.float64()),
            pa.field("dec_deg", pa.float64()),
            pa.field("pixel_x", pa.float64()),
            pa.field("pixel_y", pa.float64()),
            pa.field("edge_distance_pixels", pa.float64()),
            pa.field("requested_mag", pa.float64()),
            pa.field("injection_noise_seed", pa.int64()),
            pa.field("expected_inst_flux", pa.float64()),
            pa.field("pre_meas_flux", pa.float64()),
            pa.field("pre_meas_flux_err", pa.float64()),
            pa.field("post_meas_flux", pa.float64()),
            pa.field("post_meas_flux_err", pa.float64()),
            pa.field("recovered_injected_flux", pa.float64()),
            pa.field("flux_residual", pa.float64()),
            pa.field("fractional_flux_residual", pa.float64()),
            pa.field("difference_flux_err", pa.float64()),
            pa.field("pull", pa.float64()),
            pa.field("delta_sum_inst_flux", pa.float64()),
            pa.field("delta_peak_inst_flux", pa.float64()),
            pa.field("delta_pixels", pa.int64()),
            pa.field("pre_measure_flag", pa.string()),
            pa.field("post_measure_flag", pa.string()),
            pa.field("injection_status", pa.string()),
            pa.field("measurement_status", pa.string()),
            pa.field("native_flags_json", pa.string()),
            pa.field("psf_summary_json", pa.string()),
            pa.field("background_summary_json", pa.string()),
        ]
    )


def write_validation_rows(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write controlled validation rows as one independently readable Parquet file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=validation_schema())
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    pq.read_table(temporary, columns=["validation_id", "source_id", "image_id"])
    temporary.replace(path)


def summarize_validation(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize residuals without imposing scientific acceptance thresholds."""

    rows = list(rows)
    pulls = np.asarray([r["pull"] for r in rows if r.get("pull") is not None], dtype=float)
    residuals = np.asarray(
        [r["fractional_flux_residual"] for r in rows if r.get("fractional_flux_residual") is not None],
        dtype=float,
    )
    summary: dict[str, Any] = {"n_rows": len(rows), "n_finite_pulls": int(pulls.size)}
    if pulls.size:
        summary.update(
            pull_mean=float(np.mean(pulls)),
            pull_median=float(np.median(pulls)),
            pull_std=float(np.std(pulls, ddof=1)) if pulls.size > 1 else None,
            pull_mad=float(np.median(np.abs(pulls - np.median(pulls)))),
        )
    if residuals.size:
        summary.update(
            fractional_residual_mean=float(np.mean(residuals)),
            fractional_residual_median=float(np.median(residuals)),
            fractional_residual_std=(
                float(np.std(residuals, ddof=1)) if residuals.size > 1 else None
            ),
        )
    return summary


def run_controlled_validation(
    image: Any,
    context: Any,
    sources: Sequence[ControlledValidationSource],
    *,
    release_name: str | None = None,
    image_id: str | None = None,
    local_radius_pixels: int = 12,
    injection_noise_seed: int = 12345,
) -> list[dict[str, Any]]:
    """Run before/after forced photometry for controlled sources on one image.

    This is an RSP-dependent validation operation.  The original exposure is
    never modified: Rubin injection receives its own clone, and the injected
    exposure is discarded after measurement.
    """

    from ulens_lsst.lsst_tools import LSSTTools

    if not sources:
        return []
    exposure = image.expF
    photo_calib = exposure.getPhotoCalib()
    context_value = lambda name, default=None: (
        context.get(name, default)
        if isinstance(context, Mapping)
        else getattr(context, name, default)
    )
    mjd = context_value("mjd", context_value("expMidptMJD"))
    visit = context_value("visit")
    detector = context_value("detector")
    tools = LSSTTools(image, injection_config={"noise_seed": int(injection_noise_seed)})
    source_catalog = tools.create_injection_catalog(
        ra_values=[source.ra_deg for source in sources],
        dec_values=[source.dec_deg for source in sources],
        mag_values=[source.magnitude for source in sources],
        magnification_values=[1.0 for _ in sources],
        event_ids=list(range(len(sources))),
        expMidptMJD=mjd,
        visit=visit,
        detector=detector,
    )
    pre = tools.measure_photometry(
        exposure, source_catalog, include_native_diagnostics=True
    )
    injection = tools.inject_sources(source_catalog)
    post = tools.measure_photometry(
        injection["output_exposure"],
        injection["output_catalog"],
        include_native_diagnostics=True,
    )

    bbox = exposure.getBBox()
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        x, y = source_pixel(exposure, source)
        expected = expected_inst_flux(photo_calib, source.magnitude)
        pre_row = pre[index]
        post_row = post[index]
        pre_flux = float(pre_row["meas_flux"])
        post_flux = float(post_row["meas_flux"])
        pre_err = float(pre_row["meas_flux_err"])
        post_err = float(post_row["meas_flux_err"])
        quantities = recovery_quantities(expected, pre_flux, post_flux, pre_err, post_err)
        delta = image_delta_diagnostic(
            exposure,
            injection["output_exposure"],
            x,
            y,
            radius_pixels=local_radius_pixels,
        )
        psf_summary, background_summary = image_condition_summaries(
            exposure, x, y, radius_pixels=local_radius_pixels
        )
        edge_distance = min(
            x - bbox.getMinX(),
            bbox.getMaxX() - x,
            y - bbox.getMinY(),
            bbox.getMaxY() - y,
        )
        native_flags = {
            key: _json_scalar(post_row[key])
            for key in post.colnames
            if key.startswith("native_")
        }
        rows.append(
            {
                "validation_id": f"{image_id or context_value('image_id', 'image')}:{source.source_id}",
                "source_id": source.source_id,
                "image_id": image_id or context_value("image_id", ""),
                "release": release_name,
                "visit": visit,
                "detector": detector,
                "band": context_value("band"),
                "mjd": mjd,
                "ra_deg": source.ra_deg,
                "dec_deg": source.dec_deg,
                "pixel_x": x,
                "pixel_y": y,
                "edge_distance_pixels": float(edge_distance),
                "requested_mag": source.magnitude,
                "injection_noise_seed": int(injection_noise_seed),
                "expected_inst_flux": expected,
                "pre_meas_flux": pre_flux,
                "pre_meas_flux_err": pre_err,
                "post_meas_flux": post_flux,
                "post_meas_flux_err": post_err,
                **quantities,
                **delta,
                "pre_measure_flag": str(pre_row["measure_flag"]),
                "post_measure_flag": str(post_row["measure_flag"]),
                "injection_status": "success",
                "measurement_status": "success",
                "native_flags_json": json.dumps(native_flags, sort_keys=True),
                "psf_summary_json": json.dumps(psf_summary, sort_keys=True),
                "background_summary_json": json.dumps(
                    background_summary, sort_keys=True
                ),
            }
        )
    return rows


def _json_scalar(value: Any) -> Any:
    """Convert common NumPy/Astropy scalar values for JSON diagnostics."""

    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)
