"""Image discovery and loading through Rubin Butler or TAP/DataLink."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
import json
import math
import os
import re
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urljoin

import numpy as np

DEFAULT_TAP_URL = "https://data.lsst.cloud/api/tap"
DEFAULT_DATALINK_URL = "https://data.lsst.cloud/api/datalink"
DEFAULT_TOKEN_PATH = Path.home() / ".rsp-tap.token"
# DP1 visit-image detectors are about 0.23 deg across; the margin bounds the
# TAP center preselection before exact client-side footprint/circle intersection.
DP1_MAX_IMAGE_CENTER_OFFSET_DEG = 0.25

DATA_PREVIEW_IMAGE_SETTINGS = {
    "dp0": {
        "collection": "2.2i/runs/DP0.2",
        "butler_config": "dp02",
        "dataset_type": "calexp",
    },
    "dp1": {
        "collection": "LSSTComCam/DP1",
        "butler_config": "dp1",
        "dataset_type": "visit_image",
    },
}


@dataclass
class ImageProduct:
    """Serializable identity and available metadata for one image product."""

    backend: str
    stable_id: str
    collection: str
    product_type: str
    product_subtype: Optional[str] = None
    obs_id: Optional[str] = None
    obs_publisher_did: Optional[str] = None
    visit: Optional[int] = None
    detector: Optional[int] = None
    band: Optional[str] = None
    expMidptMJD: Optional[float] = None
    t_min: Optional[float] = None
    t_max: Optional[float] = None
    obscore_midpoint_mjd: Optional[float] = None
    seeing: Optional[float] = None
    psfSigma: Optional[float] = None
    magLim: Optional[float] = None
    datalink_url: Optional[str] = None
    dataset_type: Optional[str] = None
    data_id: Optional[dict[str, Any]] = None
    local_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return plain Python values suitable for JSON, CSV, or worker configs."""
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=_json_default)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | str) -> "ImageProduct":
        if isinstance(value, str):
            value = json.loads(value)
        fields = cls.__dataclass_fields__
        values = {key: val for key, val in value.items() if key in fields}
        for key in ("data_id", "metadata"):
            item = values.get(key)
            if isinstance(item, str):
                values[key] = json.loads(item)
        for key, item in values.items():
            if isinstance(item, float) and math.isnan(item):
                values[key] = None
        for key in ("visit", "detector"):
            if values.get(key) is not None:
                values[key] = int(values[key])
        return cls(**values)

    def to_record(self) -> dict[str, Any]:
        """Return a flat CSV-safe record for catalogs and worker inputs."""
        record = self.to_dict()
        record["data_id"] = json.dumps(record["data_id"], sort_keys=True)
        record["metadata"] = json.dumps(record["metadata"], sort_keys=True, default=_json_default)
        return record


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} in an ImageProduct")


def _plain_value(value: Any) -> Any:
    """Convert an Astropy/NumPy cell into a JSON-friendly Python scalar."""
    try:
        import numpy as np

        if np.ma.is_masked(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _text(value: Any) -> str:
    value = _plain_value(value)
    if value is None:
        raise ValueError("A required text value is missing")
    return str(value)


def _http_failure(exc: Exception, operation: str) -> RuntimeError:
    """Describe an HTTP failure without copying a possibly signed URL to logs."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        match = re.search(r"\b([45]\d\d)\b", str(exc))
        status = match.group(1) if match else None
    message = f"{operation} failed" + (f" (HTTP {status})" if status else "")
    error = RuntimeError(message)
    error.response = response
    return error


def _adql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _validate_limit(limit: Optional[int]) -> None:
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer or None")


def _optional_number(value: Any) -> Optional[float]:
    value = _plain_value(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    value = _plain_value(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def data_id_to_dict(data_id: Any) -> dict[str, Any]:
    """Convert Butler DataCoordinate mappings or plain mappings to serializable IDs."""
    mapping = getattr(data_id, "mapping", data_id)
    return {str(key): _plain_value(value) for key, value in mapping.items()}


def _footprint_overlaps_region(
    footprint: Any, region: Sequence[float]
) -> bool:
    """Test a DP1 ObsCore ICRS polygon against a circular sky region."""
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from shapely.geometry import Point, Polygon

    ra, dec, radius = _validate_region(region)
    parts = str(footprint).split()
    if len(parts) < 8 or parts[:2] != ["POLYGON", "ICRS"]:
        raise ValueError(f"Unsupported or missing DP1 ObsCore footprint: {footprint!r}")
    coordinates = np.asarray(parts[2:], dtype=float).reshape(-1, 2)
    center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    vertices = SkyCoord(ra=coordinates[:, 0] * u.deg, dec=coordinates[:, 1] * u.deg, frame="icrs")
    east, north = center.spherical_offsets_to(vertices)
    polygon = Polygon(np.column_stack((east.to_value(u.deg), north.to_value(u.deg))))
    return polygon.intersects(Point(0, 0).buffer(radius, quad_segs=256))


def image_product_from_record(
    record: Mapping[str, Any], *, data_preview: str = "dp0"
) -> ImageProduct:
    """Rebuild an image descriptor from a CSV row, including v1.0.2 rows."""
    backend = record.get("backend")
    if backend is None or (isinstance(backend, float) and math.isnan(backend)):
        settings = DATA_PREVIEW_IMAGE_SETTINGS[data_preview]
        data_id = {
            key: _optional_int(record.get(key))
            for key in ("visit", "detector")
            if record.get(key) is not None
        }
        return ImageProduct(
            backend="butler",
            stable_id=f"legacy:{data_preview}:{data_id.get('visit')}:{data_id.get('detector')}",
            collection=settings["collection"],
            product_type=settings["dataset_type"],
            visit=data_id.get("visit"),
            detector=data_id.get("detector"),
            band=record.get("band"),
            expMidptMJD=_optional_number(record.get("expMidptMJD")),
            psfSigma=_optional_number(record.get("psfSigma")),
            seeing=_optional_number(record.get("seeing")),
            magLim=_optional_number(record.get("magLim")),
            dataset_type=settings["dataset_type"],
            data_id=data_id,
        )
    return ImageProduct.from_dict(record)


def _validate_region(region: Optional[Sequence[float]]) -> Optional[tuple[float, float, float]]:
    if region is None:
        return None
    if len(region) != 3:
        raise ValueError("region must be (ra_deg, dec_deg, radius_deg)")
    ra, dec, radius = map(float, region)
    if not all(map(lambda x: x == x and abs(x) != float("inf"), (ra, dec, radius))):
        raise ValueError("region coordinates and radius must be finite")
    if not 0 <= ra < 360 or not -90 <= dec <= 90 or radius <= 0:
        raise ValueError("region requires RA in [0, 360), Dec in [-90, 90], and radius > 0")
    return ra, dec, radius


def make_tap_service(
    token_path: str | Path = DEFAULT_TOKEN_PATH,
    tap_url: str = DEFAULT_TAP_URL,
    *,
    auth: str = "rsp",
    session: Any = None,
) -> tuple[Any, Any]:
    """Create a TAP service using RSP authentication or an external token file.

    Pass an existing service/session when available. Token contents are read only
    for ``auth='token'`` and are never logged or written by this function.
    """
    if auth not in {"rsp", "token"}:
        raise ValueError("image_access.auth must be 'rsp' or 'token'")
    if session is None and auth == "rsp":
        from lsst.rsp import get_tap_service

        service = get_tap_service("tap")
        session = getattr(service, "_session", None)
        if session is None:
            raise RuntimeError("Rubin TAP service did not expose an authenticated HTTP session")
        return service, session

    from pyvo.dal import TAPService
    from pyvo.auth import AuthSession, securitymethods

    if session is None:
        token_path = Path(token_path).expanduser()
        try:
            token = token_path.read_text().strip()
        except OSError as exc:
            raise FileNotFoundError(f"TAP token file is unavailable: {token_path}") from exc
        if not token:
            raise ValueError(f"TAP token file is empty: {token_path}")
        session = AuthSession()
        session.credentials.set_password("x-oauth-basic", token)

    session.add_security_method_for_url(DEFAULT_DATALINK_URL, securitymethods.BASIC)
    return TAPService(tap_url, session=session), session


def find_tap_images(
    service: Any,
    *,
    obs_collection: str = "LSST.DP1",
    product_subtype: str = "lsst.visit_image",
    obs_id: Optional[str] = None,
    visit: Optional[int] = None,
    detector: Optional[int] = None,
    region: Optional[Sequence[float]] = None,
    bands: Optional[Sequence[str]] = None,
    limit: Optional[int] = 100,
) -> list[dict[str, Any]]:
    """Search ObsCore and return plain dictionaries, never live TAP rows."""
    _validate_limit(limit)
    region = _validate_region(region)
    conditions = [
        f"obs_collection = {_adql_string(obs_collection)}",
        "dataproduct_type = 'image'",
        f"dataproduct_subtype = {_adql_string(product_subtype)}",
    ]
    if obs_id is not None:
        conditions.append(f"obs_id = {_adql_string(obs_id)}")
    if visit is not None:
        conditions.append(f"lsst_visit = {int(visit)}")
    if detector is not None:
        conditions.append(f"lsst_detector = {int(detector)}")
    if region is not None:
        ra, dec, radius = region
        conditions.append(
            f"1 = CONTAINS(POINT('ICRS', s_ra, s_dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius}))"
        )
    if bands:
        conditions.append(
            "lsst_band IN (" + ", ".join(_adql_string(band) for band in bands) + ")"
        )

    columns = ", ".join(
        (
            "obs_collection", "obs_id", "obs_publisher_did", "dataproduct_type",
            "dataproduct_subtype", "calib_level", "access_url", "access_format",
            "s_ra", "s_dec", "s_region", "t_min", "t_max", "t_exptime",
            "lsst_band", "lsst_visit", "lsst_detector", "lsst_filter",
        )
    )
    top = f"TOP {limit} " if limit is not None else ""
    query = (
        f"SELECT {top}{columns} FROM ivoa.ObsCore "
        f"WHERE {' AND '.join(conditions)}"
    )
    table = service.search(query).to_table()
    return [
        {column: _plain_value(row[column]) for column in table.colnames}
        for row in table
    ]


def resolve_datalink_product(
    session: Any,
    datalink_url: str,
    *,
    semantics: str = "#this",
) -> dict[str, Any]:
    """Resolve one DataLink product, returning its temporary access URL."""
    from astropy.io.votable import parse_single_table

    response = session.get(str(datalink_url))
    try:
        response.raise_for_status()
    except Exception as exc:
        raise _http_failure(exc, "DataLink request") from None
    content_type = response.headers.get("Content-Type", "").lower()
    if content_type and "votable" not in content_type and "xml" not in content_type:
        raise ValueError(f"DataLink response is not a VOTable: {content_type}")

    try:
        table = parse_single_table(BytesIO(response.content)).to_table()
    except Exception as exc:
        raise ValueError("Could not parse the DataLink VOTable response") from exc

    if "error_message" in table.colnames:
        errors = [_plain_value(value) for value in table["error_message"]]
        errors = [str(value) for value in errors if value not in (None, "")]
        if errors:
            raise RuntimeError(f"DataLink reported an error: {'; '.join(errors)}")

    if "semantics" not in table.colnames or "access_url" not in table.colnames:
        raise ValueError("DataLink VOTable is missing semantics or access_url")
    matches = [
        index
        for index, value in enumerate(table["semantics"])
        if _plain_value(value) == semantics
    ]
    if not matches:
        raise LookupError(f"DataLink has no product with semantics={semantics!r}")
    if len(matches) > 1:
        raise RuntimeError(f"DataLink has multiple products with semantics={semantics!r}")

    row = table[matches[0]]
    product = {column: _plain_value(row[column]) for column in table.colnames}
    product_type = product.get("content_type")
    if product_type and "fits" not in str(product_type).lower():
        raise ValueError(
            f"DataLink #this product is not FITS: content_type={product_type!r}"
        )
    product_url = _text(product["access_url"])
    product["access_url"] = urljoin(response.url or str(datalink_url), product_url)
    if hasattr(response, "close"):
        response.close()
    return product


@contextmanager
def _product_lock(output_path: Path):
    """Serialize downloads targeting one cache path; flock releases on process death."""
    import fcntl

    lock_dir = output_path.parent / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_name = sha256(str(output_path.resolve()).encode()).hexdigest() + ".lock"
    with (lock_dir / lock_name).open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def download_tap_image(
    session: Any,
    product: Mapping[str, Any] | str,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    chunk_size: int = 2 * 1024**2,
) -> Path:
    """Download a DataLink product atomically with a per-product file lock."""
    url = product.get("access_url") if isinstance(product, Mapping) else product
    if not url:
        raise ValueError("product must provide a non-empty access_url")
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _product_lock(output_path):
        if output_path.exists() and not overwrite:
            if output_path.stat().st_size:
                return output_path
            output_path.unlink()

        response = None
        try:
            response = session.get(str(url), stream=True)
            try:
                response.raise_for_status()
            except Exception as exc:
                raise _http_failure(exc, "DataLink image download") from None
            response_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in response_type:
                raise ValueError("DataLink download returned HTML instead of a FITS product")

            expected_size = response.headers.get("Content-Length")
            expected_size = int(expected_size) if expected_size else None
            fd, temporary_name = tempfile.mkstemp(
                dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".part"
            )
            temporary_path = Path(temporary_name)
            size = 0
            with os.fdopen(fd, "wb") as stream:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        stream.write(chunk)
                        size += len(chunk)
            if size == 0:
                raise IOError("Downloaded image product is empty")
            if expected_size is not None and size != expected_size:
                raise IOError(
                    f"Downloaded image size {size} differs from Content-Length {expected_size}"
                )
            temporary_path.replace(output_path)
        except Exception:
            if "temporary_path" in locals():
                temporary_path.unlink(missing_ok=True)
            raise
        finally:
            if response is not None and hasattr(response, "close"):
                response.close()
    return output_path


def load_fits_exposure(path: str | Path) -> Any:
    """Load a local FITS image as an LSST ExposureF without constructing Butler."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"FITS file is empty: {path}")
    try:
        import lsst.afw.image as afwImage
    except ImportError as exc:
        raise RuntimeError("Loading ExposureF requires the LSST Science Pipelines") from exc
    exposure = afwImage.ExposureF(str(path))
    if exposure is None:
        raise RuntimeError(f"Could not load ExposureF from {path}")
    return exposure


def find_butler_images(
    butler: Any,
    *,
    dataset_type: str,
    where: str = "",
    bind: Optional[Mapping[str, Any]] = None,
    limit: Optional[int] = None,
) -> list[Any]:
    """Query Butler and return DatasetRefs for conversion to serializable products."""
    _validate_limit(limit)
    options: dict[str, Any] = {"limit": limit}
    if where:
        options["where"] = where
    if bind:
        options["bind"] = dict(bind)
    return list(butler.query_datasets(dataset_type, **options))


def load_butler_exposure(
    butler: Any,
    product: Any,
    *,
    dataset_type: Optional[str] = None,
) -> Any:
    """Load a DatasetRef or serializable data ID and validate the ExposureF result."""
    from collections.abc import Mapping as MappingABC
    from lsst.afw.image import ExposureF
    from lsst.daf.butler import DatasetRef

    if isinstance(product, DatasetRef):
        exposure = butler.get(product)
    elif isinstance(product, MappingABC):
        if dataset_type is None:
            dataset_type = product.get("dataset_type")
        data_id = product.get("data_id", product)
        if not dataset_type:
            raise ValueError("dataset_type is required when loading a Butler data ID")
        exposure = butler.get(dataset_type, **dict(data_id))
    else:
        raise TypeError("product must be a DatasetRef or a mapping containing a data ID")

    if not isinstance(exposure, ExposureF):
        raise TypeError(f"Butler returned {type(exposure).__name__}, not ExposureF")
    return exposure


class TapImageAccess:
    """Stateful TAP/DataLink access; construct one locally in each worker.

    The pipeline route validated to date uses collection ``LSST.DP1`` and
    product subtype ``lsst.visit_image``. Set ``cache_dir`` to use a managed
    local FITS cache; otherwise pass ``output_path`` to ``load_exposure``.
    ObsCore's original DataLink URL is kept as provenance. The resolved signed
    ``#this`` URL is temporary and is never stored as the product's stable ID.
    """

    def __init__(
        self,
        *,
        collection: str = "LSST.DP1",
        product_subtype: str = "lsst.visit_image",
        token_path: str | Path = DEFAULT_TOKEN_PATH,
        tap_url: str = DEFAULT_TAP_URL,
        auth: str = "rsp",
        service: Any = None,
        session: Any = None,
        cache_dir: Optional[str | Path] = None,
    ):
        if not collection:
            raise ValueError("image_access.collection is required for TAP")
        if not product_subtype:
            raise ValueError("image_access.product_subtype is required for TAP")
        if auth not in {"rsp", "token"}:
            raise ValueError("image_access.auth must be 'rsp' or 'token'")
        self.collection = collection
        self.product_subtype = product_subtype
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        if service is None:
            service, session = make_tap_service(
                token_path, tap_url, auth=auth, session=session
            )
        elif session is None:
            session = getattr(service, "_session", None)
        self.service = service
        self.session = session

    def find_exposures(
        self,
        *,
        region: Optional[Sequence[float]] = None,
        bands: Optional[Sequence[str]] = None,
        obs_id: Optional[str] = None,
        visit: Optional[int] = None,
        detector: Optional[int] = None,
        limit: Optional[int] = 100,
    ) -> list[ImageProduct]:
        """Find DP1 image products and return serializable descriptors.

        Results preserve the original DataLink URL and stable product ID; the
        signed URL for the FITS product is resolved later by ``load_exposure``.
        The currently validated pipeline configuration is ``LSST.DP1`` with
        subtype ``lsst.visit_image``.
        """
        _validate_limit(limit)
        query_region = region
        if region is not None:
            ra, dec, radius = _validate_region(region)
            query_region = (
                ra,
                dec,
                radius + DP1_MAX_IMAGE_CENTER_OFFSET_DEG,
            )
        rows = find_tap_images(
            self.service,
            obs_collection=self.collection,
            product_subtype=self.product_subtype,
            obs_id=obs_id,
            visit=visit,
            detector=detector,
            region=query_region,
            bands=bands,
            limit=None if region is not None else limit,
        )
        if region is not None:
            rows = [
                row for row in rows
                if _footprint_overlaps_region(row.get("s_region"), region)
            ]
            if limit is not None:
                rows = rows[:limit]
        visit_ids = sorted({int(row["lsst_visit"]) for row in rows if row.get("lsst_visit") is not None})
        ccd_visit = {}
        if visit_ids:
            visit_values = ", ".join(map(str, visit_ids))
            ccd_rows = self.service.search(
                "SELECT visitId, detector, expMidptMJD, psfSigma, seeing, magLim, band "
                f"FROM dp1.CcdVisit WHERE visitId IN ({visit_values})"
            ).to_table()
            ccd_visit = {
                (int(row["visitId"]), int(row["detector"])): {
                    key: _plain_value(row[key]) for key in ccd_rows.colnames
                }
                for row in ccd_rows
            }
        products = []
        for row in rows:
            collection = _text(row["obs_collection"])
            obs_id_value = _text(row["obs_id"])
            subtype = _text(row["dataproduct_subtype"])
            datalink_url = _text(row["access_url"])
            publisher_id = row.get("obs_publisher_did")
            stable_id = (
                _text(publisher_id)
                if publisher_id
                else f"{collection}:{obs_id_value}:{subtype}:"
                f"{row.get('lsst_visit')}:{row.get('lsst_detector')}"
            )
            visit = _optional_int(row.get("lsst_visit"))
            detector = _optional_int(row.get("lsst_detector"))
            ccd = ccd_visit.get((visit, detector))
            if ccd is None:
                raise LookupError(
                    f"No dp1.CcdVisit row for visit={visit}, detector={detector} "
                    f"(ObsCore product {stable_id!r})"
                )
            t_min = _optional_number(row.get("t_min"))
            t_max = _optional_number(row.get("t_max"))
            obs_midpoint = (
                (t_min + t_max) / 2 if t_min is not None and t_max is not None else None
            )
            products.append(
                ImageProduct(
                    backend="tap",
                    stable_id=stable_id,
                    collection=collection,
                    product_type=_text(row.get("dataproduct_type", "image")),
                    product_subtype=subtype,
                    obs_id=obs_id_value,
                    obs_publisher_did=(
                        _text(publisher_id) if publisher_id is not None else None
                    ),
                    visit=visit,
                    detector=detector,
                    band=row.get("lsst_band"),
                    expMidptMJD=_optional_number(ccd.get("expMidptMJD")),
                    t_min=t_min,
                    t_max=t_max,
                    obscore_midpoint_mjd=obs_midpoint,
                    seeing=_optional_number(ccd.get("seeing")),
                    psfSigma=_optional_number(ccd.get("psfSigma")),
                    magLim=_optional_number(ccd.get("magLim")),
                    datalink_url=datalink_url,
                    local_path=None,
                    metadata={
                        key: value
                        for key, value in row.items()
                        if key not in {
                            "obs_collection", "obs_id", "obs_publisher_did",
                            "dataproduct_type", "dataproduct_subtype", "access_url",
                            "lsst_band", "lsst_visit", "lsst_detector", "lsst_filter",
                            "em_filter", "visit", "detector", "exp_midpoint_mjd",
                            "expMidptMJD", "t_min", "t_max", "psfSigma", "seeing",
                            "magLim",
                        }
                    },
                )
            )
        return products

    def load_exposure(
        self,
        product: ImageProduct | Mapping[str, Any] | str,
        *,
        output_path: Optional[str | Path] = None,
        overwrite: bool = False,
    ) -> Any:
        """Resolve and load a TAP product as an ``ExposureF``.

        With ``cache_dir`` set on this access object, the destination is a FITS
        file named from the product's stable ID. Otherwise ``output_path`` is
        required. An existing readable FITS is reused unless ``overwrite`` is
        true. DataLink's potentially signed final ``#this`` URL is resolved
        here, used for download, and not retained in the descriptor.
        """
        product = ImageProduct.from_dict(product) if not isinstance(product, ImageProduct) else product
        if product.backend != "tap":
            raise ValueError(f"TapImageAccess cannot load backend={product.backend!r}")
        if output_path is None:
            if self.cache_dir is None:
                raise ValueError("output_path or cache_dir is required for TAP image download")
            name = sha256(product.stable_id.encode()).hexdigest() + ".fits"
            output_path = self.cache_dir / name

        output_path = Path(output_path).expanduser()
        if not overwrite:
            cached_path = Path(product.local_path).expanduser() if product.local_path else output_path
            if cached_path.is_file():
                try:
                    product.local_path = str(cached_path)
                    return load_fits_exposure(cached_path)
                except Exception as exc:
                    raise RuntimeError(
                        f"Cached product {product.stable_id!r} is not a readable ExposureF"
                    ) from exc
        if self.session is None:
            raise RuntimeError("A DataLink HTTP session is required to download this product")

        resolved = resolve_datalink_product(self.session, product.datalink_url)
        try:
            path = download_tap_image(
                self.session, resolved, output_path, overwrite=overwrite
            )
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status not in (401, 403):
                raise
            # DataLink access URLs can expire; resolve a new URL once after auth failure.
            resolved = resolve_datalink_product(self.session, product.datalink_url)
            path = download_tap_image(
                self.session, resolved, output_path, overwrite=overwrite
            )
        product.local_path = str(path)
        try:
            return load_fits_exposure(path)
        except Exception as exc:
            raise RuntimeError(
                f"Downloaded product {product.stable_id!r} could not be opened as ExposureF"
            ) from exc


class ButlerImageAccess:
    """Butler image discovery and loading with lazy Butler construction."""

    def __init__(
        self,
        *,
        data_preview: str = "dp0",
        collection: Optional[str] = None,
        butler_config: Optional[str] = None,
        dataset_type: Optional[str] = None,
        butler: Any = None,
    ):
        if data_preview not in DATA_PREVIEW_IMAGE_SETTINGS:
            raise ValueError(f"Unsupported data_preview: {data_preview!r}")
        settings = DATA_PREVIEW_IMAGE_SETTINGS[data_preview]
        self.collection = collection or settings["collection"]
        self.butler_config = butler_config or settings["butler_config"]
        self.dataset_type = dataset_type or settings["dataset_type"]
        self._butler = butler

    @property
    def butler(self) -> Any:
        if self._butler is None:
            from lsst.daf.butler import Butler

            self._butler = Butler(self.butler_config, collections=self.collection)
        return self._butler

    def find_exposures(
        self,
        *,
        region: Sequence[float],
        bands: Sequence[str],
        limit: Optional[int] = None,
    ) -> list[ImageProduct]:
        from lsst.geom import Angle, SpherePoint, degrees
        from lsst.sphgeom import Region

        ra, dec, radius = _validate_region(region)
        bands = tuple(bands)
        if not bands:
            raise ValueError("bands cannot be empty")
        bands_sql = "(" + ", ".join(_adql_string(band) for band in bands) + ")"
        circle = Region.from_ivoa_pos(f"CIRCLE {ra} {dec} {radius}")
        point = SpherePoint(Angle(ra, degrees), Angle(dec, degrees))
        where = (
            "visit_detector_region.region OVERLAPS my_region "
            f"AND band IN {bands_sql}"
        )
        refs = find_butler_images(
            self.butler,
            dataset_type=self.dataset_type,
            where=where,
            bind={
                "ra": point.getLongitude().asDegrees(),
                "dec": point.getLatitude().asDegrees(),
                "my_region": circle,
            },
            limit=limit,
        )
        products = []
        for ref in refs:
            data_id = data_id_to_dict(ref.dataId)
            stable_id = str(ref.id)
            products.append(
                ImageProduct(
                    backend="butler",
                    stable_id=stable_id,
                    collection=self.collection,
                    product_type=self.dataset_type,
                    visit=data_id.get("visit"),
                    detector=data_id.get("detector"),
                    band=data_id.get("band"),
                    dataset_type=self.dataset_type,
                    data_id=data_id,
                    metadata={"run": getattr(ref, "run", None)},
                )
            )
        return products

    def load_exposure(self, product: ImageProduct | Mapping[str, Any] | Any) -> Any:
        if isinstance(product, ImageProduct):
            product = product.to_dict()
        if isinstance(product, Mapping):
            product = dict(product)
            product.setdefault("dataset_type", self.dataset_type)
        return load_butler_exposure(self.butler, product)


def make_image_access(
    config: Optional[Mapping[str, Any]] = None,
    *,
    data_preview: str = "dp0",
    butler: Any = None,
    service: Any = None,
    session: Any = None,
    cache_dir: Optional[str | Path] = None,
) -> ButlerImageAccess | TapImageAccess:
    """Construct the configured image backend; Butler remains the default.

    TAP is currently validated for DP1 image access using collection
    ``LSST.DP1`` and subtype ``lsst.visit_image``. ``cache_dir`` is forwarded to
    ``TapImageAccess`` to choose a managed FITS cache. If it is omitted, callers
    must provide ``output_path`` when loading a TAP product. The signed DataLink
    product URL is resolved only during loading and is not a stable identifier.
    """
    config = dict(config or {})
    validate_image_access_config(config, data_preview=data_preview)
    backend = config.get("backend", "butler")
    if backend == "butler":
        if data_preview not in DATA_PREVIEW_IMAGE_SETTINGS:
            raise ValueError(f"Unsupported data_preview: {data_preview!r}")
        return ButlerImageAccess(
            data_preview=data_preview,
            collection=config.get("collection"),
            butler_config=config.get("butler_config"),
            dataset_type=config.get("dataset_type"),
            butler=butler,
        )
    if backend == "tap":
        if data_preview != "dp1":
            raise ValueError("TAP image access is currently validated only for DP1")
        collection = config.get("collection")
        product_subtype = config.get("product_subtype")
        auth = config.get("auth", "rsp")
        return TapImageAccess(
            collection=collection,
            product_subtype=product_subtype,
            token_path=config.get("token_path", DEFAULT_TOKEN_PATH),
            tap_url=config.get("tap_url", DEFAULT_TAP_URL),
            auth=auth,
            service=service,
            session=session,
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unknown image_access.backend {backend!r}; choose 'butler' or 'tap'")


def validate_image_access_config(
    config: Optional[Mapping[str, Any]], *, data_preview: str
) -> None:
    """Validate serializable backend settings without opening a service or Butler."""
    config = dict(config or {})
    backend = config.get("backend", "butler")
    if backend == "butler":
        if data_preview not in DATA_PREVIEW_IMAGE_SETTINGS:
            raise ValueError(f"Unsupported data_preview: {data_preview!r}")
        return
    if backend != "tap":
        raise ValueError(f"Unknown image_access.backend {backend!r}; choose 'butler' or 'tap'")
    if data_preview != "dp1":
        raise ValueError("TAP image access is currently validated only for DP1")
    if not config.get("collection"):
        raise ValueError("image_access.collection is required for TAP")
    if not config.get("product_subtype"):
        raise ValueError("image_access.product_subtype is required for TAP")
    if config.get("auth", "rsp") not in {"rsp", "token"}:
        raise ValueError("image_access.auth must be 'rsp' or 'token'")
    if config.get("auth") == "token" and not config.get("token_path", DEFAULT_TOKEN_PATH):
        raise ValueError("image_access.token_path is required when auth='token'")
