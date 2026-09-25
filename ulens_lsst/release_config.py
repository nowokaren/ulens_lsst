"""Resolve Rubin data-release configuration through the RSP abstraction.

This module deliberately contains no release-name mapping.  The installed
``data_release_config`` package is the single source of truth for release-
specific Butler and TAP configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProcessorCompatibility:
    """Structural compatibility result for an image processor and release."""

    status: str
    message: str


@dataclass(frozen=True)
class ResolvedImageProcessing:
    """Resolved release and processor selection for the new image path."""

    release_config: Any
    processor_name: str
    compatibility: ProcessorCompatibility


def resolve_data_release(data_release: Any = None) -> Any:
    """Return a resolved RSP ``DataReleaseConfig``.

    Parameters
    ----------
    data_release : str or DataReleaseConfig, optional
        Release name accepted by the installed RSP configuration package, or
        an already-resolved configuration object.

    Returns
    -------
    DataReleaseConfig
        The resolved release configuration supplied by RSP.

    Raises
    ------
    RuntimeError
        If the RSP configuration package is unavailable.
    TypeError
        If ``data_release`` is neither a release name nor a resolved
        ``DataReleaseConfig`` instance.
    """
    try:
        from data_release_config import DataReleaseConfig, get_data_release
    except ImportError as exc:  # pragma: no cover - depends on the RSP env
        raise RuntimeError(
            "Resolving a Rubin data release requires the RSP "
            "'data_release_config' package."
        ) from exc

    if data_release is None:
        return get_data_release()

    if not isinstance(data_release, (str, DataReleaseConfig)):
        raise TypeError(
            "data_release must be a release name or a DataReleaseConfig "
            f"instance, not {type(data_release).__name__}."
        )

    return get_data_release(data_release)


def check_processor_compatibility(
    processor_name: str,
    release_config: Any,
) -> ProcessorCompatibility:
    """Check release information needed to prepare an image-processing run.

    This is deliberately a structural check. It does not claim that Rubin
    injection or measurement tasks work for a deployment; that requires an
    RSP validation run.
    """
    if processor_name != "injection_forced_photometry":
        return ProcessorCompatibility(
            status="unsupported",
            message=(
                f"No release requirements are defined for image processor "
                f"{processor_name!r}."
            ),
        )

    required_attributes = (
        "butler_repo",
        "butler_collections",
        "calexp_dataset_types",
        "calexp_data_id",
        "photometry_method",
    )
    missing = [
        attribute
        for attribute in required_attributes
        if not hasattr(release_config, attribute)
    ]
    if missing:
        return ProcessorCompatibility(
            status="requires_rsp_validation",
            message=(
                "The resolved release does not expose all structural fields "
                f"needed for {processor_name!r}: {missing}."
            ),
        )

    if release_config.photometry_method != "calexp_forced":
        return ProcessorCompatibility(
            status="currently_unavailable",
            message=(
                f"Image processor {processor_name!r} currently requires "
                "individual calibrated-image forced photometry, but the "
                f"resolved release declares photometry_method="
                f"{release_config.photometry_method!r}. The release may "
                "become available when its RSP profile exposes the required "
                "individual-image product."
            ),
        )

    return ProcessorCompatibility(
        status="requires_rsp_validation",
        message=(
            "The resolved release declares individual calibrated-image "
            "forced photometry. Butler dataset availability and Rubin "
            "injection/measurement task compatibility still require an RSP "
            "validation run."
        ),
    )


def require_prepare_compatibility(
    processor_name: str,
    release_config: Any,
) -> ProcessorCompatibility:
    """Reject structurally unsupported processor/release combinations."""
    result = check_processor_compatibility(processor_name, release_config)
    if result.status in {"unsupported", "currently_unavailable"}:
        release_name = getattr(release_config, "name", "<unknown release>")
        raise ValueError(
            f"Selected image processor {processor_name!r} is not compatible "
            f"with resolved release {release_name!r}: {result.message}"
        )
    return result


def resolve_configured_image_processing(
    config: Mapping[str, Any],
) -> ResolvedImageProcessing | None:
    """Resolve the configured release for the new image PREPARE path.

    A missing ``data_release`` deliberately returns ``None`` so existing
    configurations continue through the explicitly temporary ``data_preview``
    compatibility path. Once ``data_release`` is supplied, release resolution
    and processor validation are mandatory.
    """
    if config.get("sim_type") != "lsst_images":
        return None

    data_release = config.get("data_release")
    if data_release is None:
        return None

    processor_name = config.get(
        "image_processor", "injection_forced_photometry"
    )
    release_config = resolve_data_release(data_release)
    compatibility = require_prepare_compatibility(
        processor_name, release_config
    )
    return ResolvedImageProcessing(
        release_config=release_config,
        processor_name=processor_name,
        compatibility=compatibility,
    )
