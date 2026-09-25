"""Generic context for one Rubin image processed by ulens_lsst."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .work_units import make_image_id


@dataclass(frozen=True)
class ImageContext:
    """Describe one processable image without loading its pixel data."""

    image_id: str
    dataset_type: str
    data_id: Mapping[str, Any]
    visit: Optional[int] = None
    detector: Optional[int] = None
    band: Optional[str] = None
    mjd: Optional[float] = None
    seeing: Optional[float] = None
    limiting_depth: Optional[float] = None

    @classmethod
    def from_release(
        cls,
        release_config: Any,
        visit: int,
        detector: int,
        *,
        band: Optional[str] = None,
        mjd: Optional[float] = None,
        seeing: Optional[float] = None,
        limiting_depth: Optional[float] = None,
    ) -> "ImageContext":
        """Build an image context using the resolved RSP configuration."""
        dataset_types = tuple(release_config.calexp_dataset_types)
        if not dataset_types:
            raise ValueError("The resolved release has no image dataset type.")

        dataset_type = dataset_types[0]
        data_id = dict(release_config.calexp_data_id(visit, detector))
        image_id = make_image_id(release_config.name, dataset_type, data_id)
        return cls(
            image_id=image_id,
            dataset_type=dataset_type,
            data_id=data_id,
            visit=int(visit),
            detector=int(detector),
            band=band,
            mjd=mjd,
            seeing=seeing,
            limiting_depth=limiting_depth,
        )
