"""Image-level scientific processors.

Processors operate on one already-loaded Rubin image and return their own
scientific products.  They do not decide how products are persisted or how
images are scheduled.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class ImageProcessResult:
    """Result returned by one image processor.

    ``products`` is intentionally processor-specific.  A future processor may
    return different tables or a different number of products.
    """

    products: Mapping[str, Any]
    status: str = "success"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class ImageProcessor:
    """Minimal interface for an operation applied to one Rubin image."""

    name = "image_processor"
    version = "1"

    def process(
        self,
        image: Any,
        context: Any,
        sources: Any,
        config: Optional[Mapping[str, Any]] = None,
    ) -> ImageProcessResult:
        raise NotImplementedError


class InjectionForcedPhotometryProcessor(ImageProcessor):
    """Inject simulated sources in memory and run forced photometry."""

    name = "injection_forced_photometry"
    version = "1"

    def process(
        self,
        image: Any,
        context: Any,
        sources: Any,
        config: Optional[Mapping[str, Any]] = None,
    ) -> ImageProcessResult:
        # Keep Rubin imports out of module import time so offline tests can
        # inspect the processor registry without an RSP environment.
        from ulens_lsst.lsst_tools import LSSTTools

        config = dict(config or {})
        tools = LSSTTools(
            image,
            injection_config=config.get("injection_config"),
            measurement_config=config.get("measurement_config"),
        )

        def column(name: str, default: Any = None):
            if hasattr(sources, "colnames") and name in sources.colnames:
                return sources[name]
            if hasattr(sources, "columns") and name in sources.columns:
                return sources[name].to_numpy()
            if isinstance(sources, Mapping) and name in sources:
                return sources[name]
            return default

        def context_value(name: str, default: Any = None):
            if isinstance(context, Mapping):
                return context.get(name, default)
            return getattr(context, name, default)

        magnification = column("magnification")
        if magnification is None:
            magnification = [1.0] * len(sources)
        event_ids = column("event_id")
        if event_ids is None:
            event_ids = [None] * len(sources)

        exposure_time = context_value("expMidptMJD", context_value("mjd"))
        if exposure_time is None:
            exposure_time = column("time")

        injection_catalog = tools.create_injection_catalog(
            ra_values=column("ra"),
            dec_values=column("dec"),
            mag_values=column("ideal_mag"),
            magnification_values=magnification,
            event_ids=event_ids,
            expMidptMJD=exposure_time,
            visit=context_value("visit"),
            detector=context_value("detector"),
        )
        injection_output = tools.inject_sources(injection_catalog)
        extraction = tools.measure_photometry(
            injection_output["output_exposure"],
            injection_output["output_catalog"],
        )
        return ImageProcessResult(
            products={"photometry": extraction},
            diagnostics={"n_sources": len(injection_catalog)},
        )


IMAGE_PROCESSORS = {
    InjectionForcedPhotometryProcessor.name: InjectionForcedPhotometryProcessor,
}


def get_image_processor(name: str) -> ImageProcessor:
    """Construct the statically registered processor named by the run config."""

    try:
        return IMAGE_PROCESSORS[name]()
    except KeyError as exc:
        available = ", ".join(sorted(IMAGE_PROCESSORS))
        raise ValueError(f"Unknown image processor {name!r}; available: {available}") from exc
