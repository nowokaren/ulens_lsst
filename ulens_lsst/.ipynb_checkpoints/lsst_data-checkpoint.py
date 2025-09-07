"""
LSST data management module for the LSST pipeline project.

This module provides the `LSSTData` and `Calexp` classes for loading, processing, and visualizing
LSST data products, including calibrated exposures (calexps) and catalog data from TAP services.
It is designed to support scientific experiments such as microlensing light curve recoverability,
blending effect studies, and comparisons between DP0 simulations and DP1 real data samples.
The module optimizes memory usage by loading data selectively and processing in batches where
possible, and prioritizes time efficiency with streamlined queries and parallelizable operations.

Classes
-------
LSSTData : Manages loading, querying, and storing LSST catalog and calexp data.
Calexp : Handles individual LSST calibrated exposures, including WCS operations, cutouts, and visualization.
Functions
-------
submit_with_retries : Submits TAP queries with retry logic for handling rate limits.
"""

# Standard library imports
import gc
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from matplotlib.patches import Circle, Polygon as pltPolygon
from pyvo.dal import DALServiceError
from regions import CircleSkyRegion
from shapely.geometry import Point, Polygon

# LSST imports
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
from lsst.daf.butler import Butler
from lsst.geom import Angle, SpherePoint, degrees, Point2D, Extent2I, Point2I, Box2I
from lsst.meas.algorithms.detection import SourceDetectionTask
from lsst.meas.base import SingleFrameMeasurementTask
from lsst.rsp import get_tap_service
from lsst.sphgeom import Region
from lsst.afw.geom import makeSkyWcs


def submit_with_retries(
    service, query: str, max_retries: int = 50, base_delay: float = 1
) -> "pyvo.dal.TAPJob":
    """
    Submit a TAP query with retry logic for handling rate limit errors (HTTP 429).

    Parameters
    ----------
    service : pyvo.dal.TAPService
        TAP service instance.
    query : str
        ADQL query to submit.
    max_retries : int, optional
        Maximum number of retry attempts (default: 50).
    base_delay : float, optional
        Base delay for exponential backoff in seconds (default: 1).

    Returns
    -------
    pyvo.dal.TAPJob
        Submitted TAP job.

    Raises
    ------
    RuntimeError
        If max retries are reached or non-429 errors occur.
    """
    for attempt in range(max_retries):
        try:
            return service.submit_job(query)
        except DALServiceError as e:
            if "429" in str(e):
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                time.sleep(delay)
            else:
                raise RuntimeError(f"Query failed: {str(e)}")
    raise RuntimeError(f"Max retries ({max_retries}) reached due to 429 errors.")


class LSSTData:
    """
    Class to load, process, and manage LSST data products with flexible catalog and calexp handling.

    Supports querying TAP catalogs and retrieving calibrated exposures via Butler, with methods
    optimized for memory efficiency through selective column loading and batch processing.

    Attributes
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    radius : float
        Search radius in degrees.
    bands : List[str]
        List of photometric bands.
    main_path : str
        Directory for storing output files.
    catalogs : Dict[str, pd.DataFrame]
        Loaded catalog data.
    calexps_data : pd.DataFrame
        Metadata for calibrated exposures.
    butler : lsst.daf.butler.Butler
        Butler instance for accessing data.
    region : regions.CircleSkyRegion
        Sky region for queries.
    """

    AVAILABLE_SCHEMAS = {
        "tap_schemas": {
            "dp02_dc2_catalogs": [
                "CcdVisit",
                "CoaddPatches",
                "DiaObject",
                "DiaSource",
                "ForcedSource",
                "ForcedSourceOnDiaObject",
                "MatchesTruth",
                "Object",
                "ObsCore",
                "Source",
                "TruthSummary",
                "Visit",
            ],
            "dp1": [
                "CcdVisit",
                "CoaddPatches",
                "DiaObject",
                "DiaSource",
                "ForcedSource",
                "ForcedSourceOnDiaObject",
                "MPCORB",
                "Object",
                "Source",
                "SSObject",
                "Visit",
            ],
            "ivoa": ["ObsCore"],
            "tap_schema": ["columns", "key_columns", "keys", "schemas", "tables"],
        },
        "butler_collections": [
            "2.2i/runs/DP0.2",
            "2.2i/defaults/DP0.2",
            "2.2i/raw/DP0.2",
            "2.2i/runs/DP0.2/v23_0_1/PREOPS-905/step3",
            "2.2i/runs/DP0.2/v23_0_1/PREOPS-905/step4",
            "2.2i/runs/DP0.2/v23_0_2/PREOPS-905/step5",
            "2.2i/runs/DP0.2/v23_0_2/PREOPS-905/step6",
        ],
    }

    DATA_PREVIEW_SETTINGS = {
        "dp1": {
            "collection": "LSSTComCam/DP1",
            "butler_config": "dp1",
            "tap_schema": "dp1",
        },
        "dp0": {
            "collection": "2.2i/runs/DP0.2",
            "butler_config": "dp02",
            "tap_schema": "dp02_dc2_catalogs",
        },
    }

    DEFAULT_CATALOG_COLS_DP1 = {
        "Source": [
            "coord_ra",
            "coord_dec",
            "visit",
            "detector",
            "band",
            "blendedness_abs",
            "deblend_nChild",
            "psfFlux",
            "psfFluxErr",
            "psfFlux_flag",
            "localBackground_instFlux",
        ],
        "ForcedSource": [
            "coord_ra",
            "coord_dec",
            "band",
            "visit",
            "detector",
            "patch",
            "psfFlux",
            "psfFluxErr",
            "psfFlux_flag",
        ],
        "Object": ["coord_ra", "coord_dec", "ObjectId"],
        "CoaddPatches": ["s_ra", "s_dec", "lsst_tract", "lsst_patch", "s_region"],
        "Visit": [
            "ra",
            "dec",
            "airmass",
            "band",
            "expTime",
            "obsStartMJD",
            "expMidpt",
            "skyRotation",
            "visit",
        ],
        "CcdVisit": [
            "ra",
            "dec",
            "expMidptMJD",
            "detector",
            "visitId",
            "ccdVisitId",
            "magLim",
            "pixelScale",
            "psfSigma",
            "zeroPoint",
        ],
    }

    DEFAULT_CATALOG_COLS_DP02 = {
        "Source": [
            "coord_ra",
            "coord_dec",
            "visit",
            "detector",
            "band",
            "blendedness_abs",
            "deblend_nChild",
            "psfFlux",
            "psfFluxErr",
            "psfFlux_flag",
            "localBackground_instFlux",
        ],
        "ForcedSource": [
            "coord_ra",
            "coord_dec",
            "band",
            "ccdVisitId",
            "tract",
            "patch",
            "psfFlux",
            "psfFluxErr",
            "psfFlux_flag",
        ],
        "Object": ["coord_ra", "coord_dec", "ObjectId"],
        "CoaddPatches": ["s_ra", "s_dec", "lsst_tract", "lsst_patch", "s_region"],
        "Visit": [
            "ra",
            "decl",
            "airmass",
            "band",
            "expTime",
            "obsStartMJD",
            "expMidpt",
            "skyRotation",
            "visit",
        ],
        "CcdVisit": ["ra", "decl", "expMidptMJD", "detector", "visitId", "psfSigma", "zeroPoint"],
    }

    def __init__(
        self,
        ra: float = 62.0,
        dec: float = -34.0,
        radius: float = 0.01,
        data_preview: str = "dp1",
        bands: List[str] = ["u", "g", "r", "i", "z", "y"],
        name: Optional[str] = None,
        main_path: str = "./runs",
    ):
        """
        Initialize the LSSTData class.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (default: 62.0).
        dec : float, optional
            Declination in degrees (default: -34.0).
        radius : float, optional
            Search radius in degrees (default: 0.01).
        data_preview : str, optional
            Data product type ('dp0' or 'dp1') (default: 'dp1').
        bands : List[str], optional
            List of photometric bands (default: ['u', 'g', 'r', 'i', 'z', 'y']).
        name : str, optional
            Name of the run; defaults to timestamp if None.
        main_path : str, optional
            Base directory for storing run data (default: './runs').

        Raises
        ------
        ValueError
            If tap_schema or butler_config is invalid.
        """
        if data_preview not in self.DATA_PREVIEW_SETTINGS:
            raise ValueError(
                f"Invalid data_preview: {data_preview}. Choose from {list(self.DATA_PREVIEW_SETTINGS.keys())}"
            )

        self.collection = self.DATA_PREVIEW_SETTINGS[data_preview]["collection"]
        self.butler_config = self.DATA_PREVIEW_SETTINGS[data_preview]["butler_config"]
        self.tap_schema = self.DATA_PREVIEW_SETTINGS[data_preview]["tap_schema"]

        if self.tap_schema not in self.AVAILABLE_SCHEMAS["tap_schemas"]:
            raise ValueError(
                f"Invalid tap_schema: {self.tap_schema}. Choose from {list(self.AVAILABLE_SCHEMAS['tap_schemas'].keys())}"
            )
        if self.butler_config not in ["dp02", "dp02-remote", "dp1"]:
            print(
                f"Warning: butler_config {self.butler_config} not in known aliases (dp02, dp02-remote, dp1)."
            )

        self.name = name or time.strftime("%Y%m%d_%H%M%S")
        self.main_path = os.path.join(main_path, self.name)
        os.makedirs(self.main_path, exist_ok=True)

        self.ra = float(ra)
        self.dec = float(dec)
        self.radius = float(radius)
        self.bands = bands
        self.data_preview = data_preview
        self.area = np.pi * self.radius**2
        self.catalogs = {}
        self.calexps_data = None
        self.datasetRefs = None
        self.epochs = None
        self.survey_dates = None
        self.butler = Butler(self.butler_config, collections=self.collection)
        self._setup_region()

    def _setup_region(self) -> None:
        """Set up the sky region for source and calexp queries."""
        center = SkyCoord(ra=self.ra, dec=self.dec, unit="deg", frame="icrs")
        self.region = CircleSkyRegion(center=center, radius=self.radius * u.deg)

    def load_catalog(
        self,
        catalog: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List[str]] = None,
        region: Optional[CircleSkyRegion] = None,
        ra_values: Optional[np.ndarray] = None,
        dec_values: Optional[np.ndarray] = None,
        multisearch_radius: Optional[float] = None,
        drop_na: bool = False,
        save: bool = True,
        delta_radius: float = 0.0,
        print_on: bool = True,
    ) -> pd.DataFrame:
        """
        Load data from a specified catalog using TAP service.

        Parameters
        ----------
        catalog : str
            Catalog name (e.g., 'Source', 'ForcedSource', 'Object').
        columns : List[str], optional
            Columns to retrieve; defaults to predefined columns for the catalog.
        filters : List[str], optional
            Additional ADQL filter conditions (e.g., ['extendedness = 0']).
        region : CircleSkyRegion, optional
            Sky region for spatial query; defaults to self.region.
        ra_values : np.ndarray, optional
            Array of RA values for multi-point search.
        dec_values : np.ndarray, optional
            Array of Dec values for multi-point search.
        multisearch_radius : float, optional
            Search radius for multi-point search (degrees).
        drop_na : bool, optional
            Drop rows with NaN values (default: False).
        save : bool, optional
            Save catalog to CSV file (default: True).
        delta_radius : float, optional
            Additional radius for spatial query (degrees) (default: 0.0).
        print_on : bool, optional
            Print progress messages (default: True).

        Returns
        -------
        pd.DataFrame
            Retrieved catalog data.

        Raises
        ------
        ValueError
            If catalog is unsupported.
        RuntimeError
            If no records are found.
        """
        default_columns = (
            self.DEFAULT_CATALOG_COLS_DP1
            if "dp0" not in self.tap_schema
            else self.DEFAULT_CATALOG_COLS_DP02
        )
        if catalog not in default_columns:
            raise ValueError(
                f"Unsupported catalog: {catalog}. Choose from {list(default_columns.keys())}"
            )

        columns = default_columns[catalog] + columns if columns else default_columns[catalog]
        full_catalog = f"{self.tap_schema}.{catalog}"
        region = region or self.region
        center = region.center
        radius = region.radius.to(u.deg).value

        if print_on:
            print(f"Loading data from {full_catalog}...")

        query = f"SELECT {', '.join(columns)} FROM {full_catalog} "
        spatial_clauses = []
        non_spatial_clauses = []

        # Spatial constraint
        if catalog in ["Source", "ForcedSource", "Object", "CoaddPatches", "Visit", "CcdVisit"]:
            if ra_values is None:
                spatial_clauses.append(
                    f"CONTAINS(POINT('ICRS', {columns[0]}, {columns[1]}), "
                    f"CIRCLE('ICRS', {center.ra.deg}, {center.dec.deg}, {radius + delta_radius})) = 1"
                )
            elif isinstance(ra_values, np.ndarray) and multisearch_radius is not None:
                for ra, dec in zip(ra_values, dec_values):
                    spatial_clauses.append(
                        f"CONTAINS(POINT('ICRS', {columns[0]}, {columns[1]}), "
                        f"CIRCLE('ICRS', {ra}, {dec}, {multisearch_radius + delta_radius})) = 1"
                    )

        # Special condition for dp0
        if "dp0" in self.data_preview and catalog in ["Source", "ForcedSource"]:
            non_spatial_clauses.append("detect_isPrimary = 1")

        # Custom filters
        if filters:
            non_spatial_clauses.extend(filters)

        # Build WHERE clause
        where_clauses = [f"({' OR '.join(spatial_clauses)})" if spatial_clauses else ""] + non_spatial_clauses
        where_clauses = [c for c in where_clauses if c]
        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses)

        # Execute query
        service = get_tap_service("tap")
        job = submit_with_retries(service, query)
        job.run()
        job.wait(phases=["COMPLETED", "ERROR"])
        if job.phase == "ERROR":
            job.raise_if_error()
        results = job.fetch_result().to_table().to_pandas()

        if print_on:
            print(f"Records found: {len(results)}")
        if len(results) == 0:
            raise RuntimeError("No records found. Consider increasing radius or checking filters.")

        if drop_na:
            results = results.dropna()

        if save:
            results.to_csv(
                os.path.join(self.main_path, f"{catalog.lower()}_catalog.csv"), index=False
            )
            self.catalogs[catalog] = results

        return results

    def butler_collections(self, query: str = "*") -> None:
        """
        Print available Butler collections matching the query.

        Parameters
        ----------
        query : str, optional
            Query pattern for collections (e.g., '*calib*') (default: '*').
        """
        for c in sorted(self.butler.collections.query(query)):
            print(c)

    def butler_collection_info(self, collection: str) -> None:
        """
        Print information about a specific Butler collection.

        Parameters
        ----------
        collection : str
            Name of the collection to query.
        """
        print(self.butler.collections.query_info(collection))

    def butler_dataset_types(self, query: str = "*") -> None:
        """
        Print available dataset types matching the query.

        Parameters
        ----------
        query : str, optional
            Query pattern for dataset types (e.g., '*_tract') (default: '*').
        """
        for dt in sorted(self.butler.registry.queryDatasetTypes(query)):
            print(dt)

    def get_available_tap_schemas(self) -> List[str]:
        """
        List available TAP schemas.

        Returns
        -------
        List[str]
            Available TAP schemas.

        Raises
        ------
        RuntimeError
            If query fails.
        """
        try:
            service = get_tap_service("tap")
            query = "SELECT schema_name FROM tap_schema.schemas"
            job = submit_with_retries(service, query)
            job.run()
            job.wait(phases=["COMPLETED", "ERROR"])
            if job.phase == "ERROR":
                job.raise_if_error()
            return job.fetch_result().to_table().to_pandas()["schema_name"].tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to query TAP schemas: {str(e)}")

    def load_calexp(
        self,
        visit: Optional[int] = None,
        detector: Optional[int] = None,
        dataset_ref: Optional["lsst.daf.butler.DatasetRef"] = None,
    ) -> "Calexp":
        """
        Load a calibrated exposure (calexp) for a specific visit and detector or from a DatasetRef.

        Parameters
        ----------
        visit : int, optional
            Visit ID.
        detector : int, optional
            Detector ID.
        dataset_ref : lsst.daf.butler.DatasetRef, optional
            Butler dataset reference.

        Returns
        -------
        Calexp
            Calibrated exposure instance.

        Raises
        ------
        ValueError
            If neither visit/detector nor dataset_ref is provided.
        """
        if dataset_ref is not None:
            data_id = {"visit": dataset_ref.dataId["visit"], "detector": dataset_ref.dataId["detector"]}
        elif visit is not None and detector is not None:
            data_id = {"visit": visit, "detector": detector}
        else:
            raise ValueError("Must provide either visit and detector or dataset_ref.")

        print(f"Loading calexp for visit={data_id['visit']}, detector={data_id['detector']}...")
        return Calexp.from_lsstdata(self, data_id)

    def calexp_catalog(
        self, bands: Optional[List[str]] = None, n_max: Union[int, str] = None, load_mjd: bool = True
    ) -> None:
        """
        Load catalog of calibrated exposures overlapping the specified region.

        Stores results in `self.calexps_data` and computes survey epochs.

        Parameters
        ----------
        bands : List[str], optional
            Bands to filter; defaults to self.bands.
        n_max : Union[int, str], optional
            Maximum number of calexps to load ('all' or integer) (default: 100000000).
        load_mjd : bool, optional
            Load MJD from CcdVisit catalog (default: True).
        """
        bands = bands or self.bands
        bands_str = f"({', '.join(map(repr, bands))})"
        print(f"Collecting calexps for bands {bands_str}...")

        target_point = SpherePoint(Angle(self.ra, degrees), Angle(self.dec, degrees))
        RA = target_point.getLongitude().asDegrees()
        DEC = target_point.getLatitude().asDegrees()
        circle = Region.from_ivoa_pos(f"CIRCLE {RA} {DEC} {self.radius}")
        dataset_type = "visit_image" if "dp1" in self.butler_config else "calexp"

        self.datasetRefs = list(
            self.butler.query_datasets(
                dataset_type,
                where=f"visit_detector_region.region OVERLAPS my_region AND band IN {bands_str}",
                bind={"ra": RA, "dec": DEC, "my_region": circle},
                limit=None if n_max is None else n_max,
            )
        )
        print(f"Found {len(self.datasetRefs)} calexps.")

        calexps_data = []
        if load_mjd:
            ccd_visit = self.load_catalog("CcdVisit", delta_radius=0.4, save=False)
            for ref in self.datasetRefs:
                detector = ref.dataId["detector"]
                visit = ref.dataId["visit"]
                mask = (ccd_visit["detector"] == detector) & (ccd_visit["visitId"] == visit)
                if mask.any():
                    row = ccd_visit[mask].iloc[0]
                    data = {
                        "detector": detector,
                        "visit": visit,
                        "band": ref.dataId["band"],
                        "expMidptMJD": row["expMidptMJD"],
                        "psfSigma": row["psfSigma"],
                    }
                    if "dp1" in self.butler_config:
                        data["magLim"] = row["magLim"]
                    calexps_data.append(data)
        else:
            for ref in self.datasetRefs:
                calexps_data.append(
                    {
                        "detector": ref.dataId["detector"],
                        "visit": ref.dataId["visit"],
                        "band": ref.dataId["band"],
                    }
                )

        self.calexps_data = pd.DataFrame(calexps_data).sort_values("expMidptMJD")
        self.epochs = {
            band: sorted(set(self.calexps_data[self.calexps_data["band"] == band]["expMidptMJD"].values))
            for band in self.bands
        }
        self.epochs = {k: v for k, v in self.epochs.items() if v}
        if self.epochs:
            self.survey_dates = (
                np.min([np.min(arr) for arr in self.epochs.values()]),
                np.max([np.max(arr) for arr in self.epochs.values()]),
            )

    def _spherical_circle_edge(self, ra: float, dec: float, radius: float, n_points: int = 150) -> Tuple[List[float], List[float]]:
        """
        Generate points for a circular region on the sky.

        Parameters
        ----------
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.
        radius : float
            Radius in degrees.
        n_points : int, optional
            Number of points to generate (default: 150).

        Returns
        -------
        Tuple[List[float], List[float]]
            RA and Dec coordinates of the circle edge.
        """
        center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        theta = np.linspace(0, 2 * np.pi, n_points)
        ra_edge, dec_edge = [], []
        for t in theta:
            point = center.directional_offset_by(t * u.rad, radius * u.deg)
            ra_edge.append(point.ra.deg)
            dec_edge.append(point.dec.deg)
        return ra_edge, dec_edge

    def save_data(self) -> None:
        """
        Save loaded catalog and calexp data to CSV files.
        """
        for catalog_name, catalog_data in self.catalogs.items():
            catalog_data.to_csv(
                os.path.join(self.main_path, f"{catalog_name.lower()}_catalog.csv"),
                index=False,
            )
        if self.calexps_data is not None:
            self.calexps_data.to_csv(os.path.join(self.main_path, "calexps.csv"), index=False)


class Calexp:
    """
    Class to handle LSST calibrated exposures, including WCS operations, cutouts, and visualization.

    Attributes
    ----------
    expF : lsst.afw.image.ExposureF
        The calibrated exposure.
    wcs : lsst.afw.geom.SkyWcs
        World Coordinate System of the exposure.
    band : str
        Photometric band of the exposure.
    data_id : Dict[str, Any]
        Identifier for the exposure (visit, detector).
    butler : lsst.daf.butler.Butler
        Butler instance for data access.
    """

    DATA_PREVIEW_SETTINGS = {
        "dp1": {
            "collection": "LSSTComCam/DP1",
            "butler_config": "dp1",
            "tap_schema": "dp1",
        },
        "dp0": {
            "collection": "2.2i/runs/DP0.2",
            "butler_config": "dp02",
            "tap_schema": "dp02_dc2_catalogs",
        },
    }

    def __init__(
        self,
        data_id: Union[Dict[str, Any], "lsst.daf.butler.DatasetRef", "lsst.afw.image.ExposureF", str],
        butler: Optional["lsst.daf.butler.Butler"] = None,
        butler_config: Optional[str] = None,
        data_preview: Optional[str] = None,
        name: Optional[str] = None,
        ra: float = 0.0,
        dec: float = 0.0,
        radius: float = 0.0,
        bands: Optional[List[str]] = None,
        main_path: str = ".",
    ):
        """
        Initialize a Calexp instance.

        Parameters
        ----------
        data_id : Union[dict, DatasetRef, ExposureF, str]
            Identifier for the exposure (dict with visit/detector, DatasetRef, ExposureF, or file path).
        butler : lsst.daf.butler.Butler, optional
            Butler instance; created if None.
        butler_config : str, optional
            Butler configuration identifier.
        data_preview : str, optional
            Data product type ('dp0', 'dp1').
        name : str, optional
            Name of the dataset; defaults to 'default_calexp' if None.
        ra : float, optional
            Right Ascension in degrees (default: 0.0).
        dec : float, optional
            Declination in degrees (default: 0.0).
        radius : float, optional
            Search radius in degrees (default: 0.0).
        bands : List[str], optional
            List of filter bands; defaults to ['u', 'g', 'r', 'i', 'z', 'y'].
        main_path : str, optional
            Base path for file storage (default: '.').

        Raises
        ------
        ValueError
            If data_id is invalid.
        RuntimeError
            If exposure cannot be loaded.
        """
        data_preview = data_preview or "dp0"
        if data_preview not in self.DATA_PREVIEW_SETTINGS:
            data_preview = "dp0"

        self.collection = self.DATA_PREVIEW_SETTINGS[data_preview]["collection"]
        self.butler_config = butler_config or self.DATA_PREVIEW_SETTINGS[data_preview]["butler_config"]
        self.tap_schema = self.DATA_PREVIEW_SETTINGS[data_preview]["tap_schema"]
        self.name = name or "default_calexp"
        self.ra = ra
        self.dec = dec
        self.radius = radius
        self.bands = bands or ["u", "g", "r", "i", "z", "y"]
        self.main_path = main_path

        self.butler = butler or Butler(self.butler_config, collections=self.collection)

        if isinstance(data_id, afwImage.ExposureF):
            self.expF = data_id
            self.data_id = {}
        elif isinstance(data_id, dict):
            self.data_id = data_id
            dataset_type = "visit_image" if "dp1" in self.butler_config else "calexp"
            self.expF = self.butler.get(dataset_type, **data_id)
        elif isinstance(data_id, str):
            self.expF = afwImage.ExposureF(data_id)
            self.data_id = {}
        else:
            raise ValueError("data_id must be a dict, DatasetRef, ExposureF, or str")

        if self.expF is None:
            raise RuntimeError("Failed to load exposure.")

        self.wcs = self.expF.getWcs()
        self.band = self.expF.getFilter().bandLabel

    @classmethod
    def from_lsstdata(cls, lsstdata_obj: "LSSTData", data_id: Union[Dict, "lsst.daf.butler.DatasetRef"]) -> "Calexp":
        """
        Create a Calexp instance from an LSSTData object.

        Parameters
        ----------
        lsstdata_obj : LSSTData
            LSSTData instance containing butler and configuration.
        data_id : Union[dict, DatasetRef]
            Identifier for the exposure.

        Returns
        -------
        Calexp
            New Calexp instance.
        """
        return cls(
            data_id=data_id,
            butler=lsstdata_obj.butler,
            butler_config=lsstdata_obj.butler_config,
            data_preview=lsstdata_obj.data_preview,
            name=lsstdata_obj.name,
            ra=lsstdata_obj.ra,
            dec=lsstdata_obj.dec,
            radius=lsstdata_obj.radius,
            bands=lsstdata_obj.bands,
            main_path=lsstdata_obj.main_path,
        )

    @property
    def center(self) -> Tuple[float, float]:
        """
        Return the RA, Dec of the calexp center.

        Returns
        -------
        Tuple[float, float]
            RA and Dec in degrees.
        """
        ra, dec = self.wcs.getSkyOrigin()
        return ra.asDegrees(), dec.asDegrees()

    def pix_to_sky(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to RA and Dec.

        Parameters
        ----------
        x : float
            Pixel x-coordinate.
        y : float
            Pixel y-coordinate.

        Returns
        -------
        Tuple[float, float]
            RA and Dec in degrees.
        """
        sphere_point = self.wcs.pixelToSky(Point2D(x, y))
        return sphere_point.getRa().asDegrees(), sphere_point.getDec().asDegrees()

    def sky_to_pix(self, ra: float, dec: float) -> Tuple[int, int]:
        """
        Convert RA and Dec to pixel coordinates.

        Parameters
        ----------
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.

        Returns
        -------
        Tuple[int, int]
            Pixel x and y coordinates.
        """
        xy = self.wcs.skyToPixel(SpherePoint(ra * degrees, dec * degrees))
        return int(np.round(xy.getX())), int(np.round(xy.getY()))

    def get_corners(self, coord: str = "sky") -> Tuple[List[float], List[float]]:
        """
        Return the corners of the calexp in sky or pixel coordinates.

        Parameters
        ----------
        coord : str, optional
            Coordinate system ('sky' or 'pixel') (default: 'sky').

        Returns
        -------
        Tuple[List[float], List[float]]
            x (or RA) and y (or Dec) coordinates of the corners.
        """
        x0 = float(self.expF.getX0())
        y0 = float(self.expF.getY0())
        width = self.expF.getWidth()
        height = self.expF.getHeight()
        xcorners = [x0, x0 + width, x0 + width, x0]
        ycorners = [y0, y0, y0 + height, y0 + height]

        if coord == "sky":
            ra_corners, dec_corners = [], []
            for x, y in zip(xcorners, ycorners):
                ra, dec = self.pix_to_sky(x, y)
                ra_corners.append(ra)
                dec_corners.append(dec)
            return ra_corners, dec_corners
        return xcorners, ycorners

    def contains(self, ra: Union[float, np.ndarray], dec: Union[float, np.ndarray]) -> Union[bool, List[bool]]:
        """
        Check if a point or points are within the calexp boundaries.

        Parameters
        ----------
        ra : Union[float, np.ndarray]
            Right Ascension(s) in degrees.
        dec : Union[float, np.ndarray]
            Declination(s) in degrees.

        Returns
        -------
        Union[bool, List[bool]]
            Whether the point(s) are within the calexp boundaries.
        """
        ra_corners, dec_corners = self.get_corners()
        polygon = Polygon(zip(ra_corners, dec_corners))
        if isinstance(ra, (float, np.floating, int)):
            return polygon.contains(Point(ra, dec))
        return [polygon.contains(Point(r, d)) for r, d in zip(ra, dec)]

    def cutout(self, roi: Tuple[Tuple[float, float], Union[int, Extent2I]]) -> "Calexp":
        """
        Create a cutout of the image centered at (ra, dec) with a given size in pixels.

        Parameters
        ----------
        roi : Tuple[Tuple[float, float], Union[int, Extent2I]]
            (ra, dec) in degrees and size in pixels (int or Extent2I).

        Returns
        -------
        Calexp
            New Calexp instance representing the cutout.

        Raises
        ------
        ValueError
            If coordinates are out of image field.
        RuntimeError
            If cutout creation fails.
        """
        ra, dec = roi[0]
        cutout_side_length = roi[1]
        x, y = self.sky_to_pix(ra, dec)
        width, height = self.expF.image.getDimensions()

        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Coordinates out of image field.")

        size = (
            Extent2I(cutout_side_length, cutout_side_length)
            if isinstance(cutout_side_length, int)
            else cutout_side_length
        )

        try:
            exp = self.expF.getCutout(Point2D(x, y), size)
        except Exception as e:
            raise RuntimeError(f"Could not create cutout: {str(e)}")

        cutout = Calexp(data_id=exp, butler=self.butler, butler_config=self.butler_config)
        cutout.wcs = makeSkyWcs(
            crpix=Point2D(size.getX() / 2.0, size.getY() / 2.0),
            crval=SpherePoint(ra, dec, degrees),
            cdMatrix=self.wcs.getCdMatrix(),
        )
        return cutout

    def plot(
        self,
        title: Optional[str] = None,
        sources: Optional[Union[pd.DataFrame, afwTable.SourceCatalog]] = None,
        show: bool = False,
        close: bool = False,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (6, 6),
        save_path: Optional[str] = None,
        ra_space: float = 10,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> Optional[plt.Axes]:
        """
        Plot the calibrated exposure using matplotlib.

        Parameters
        ----------
        title : str, optional
            Plot title.
        sources : Union[pd.DataFrame, afwTable.SourceCatalog], optional
            Sources to overlay as circles.
        show : bool, optional
            Display the plot (default: False).
        close : bool, optional
            Close the figure after saving (default: False).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on; creates new if None.
        figsize : Tuple[float, float], optional
            Figure size (default: (6, 6)).
        save_path : str, optional
            Path to save the plot.
        ra_space : float, optional
            Tick spacing in degrees (default: 10).
        vmin : float, optional
            Minimum value for image scaling.
        vmax : float, optional
            Maximum value for image scaling.

        Returns
        -------
        matplotlib.axes.Axes, optional
            Plot axes if not closed.
        """
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = plt.subplot(projection=WCS(self.wcs.getFitsMetadata()))

        if vmin is None or vmax is None:
            data = self.expF.image.array
            vmin = vmin or np.percentile(data, 1)
            vmax = vmax or np.percentile(data, 99)

        ax.imshow(self.expF.image.array, cmap="gray", vmin=vmin, vmax=vmax)
        ax.coords["ra"].set_axislabel("RA (degrees)")
        ax.coords["dec"].set_axislabel("Dec (degrees)")
        ax.coords["ra"].set_major_formatter("dd:mm:ss")
        ax.coords["dec"].set_major_formatter("dd:mm:ss")

        ra_corners, _ = self.get_corners()
        space = abs(ra_corners[1] - ra_corners[0]) / ra_space
        ax.coords["ra"].set_ticks(spacing=space * u.deg)
        ax.coords["dec"].set_ticks(spacing=space * u.deg)
        ax.coords["ra"].set_ticklabel(rotation=30, fontsize=8)
        ax.coords["dec"].set_ticklabel(rotation=30, fontsize=8, pad=15)
        ax.grid(color="white", ls="--", lw=0.2)

        if sources is not None:
            if isinstance(sources, afwTable.SourceCatalog):
                sources = sources.asAstropy().to_pandas()
                sources["coord_ra"] = sources["coord_ra"] * 180 / np.pi
                sources["coord_dec"] = sources["coord_dec"] * 180 / np.pi
            for _, src in sources.iterrows():
                x, y = self.sky_to_pix(src["coord_ra"], src["coord_dec"])
                circle = Circle(
                    (x, y),
                    radius=5,
                    edgecolor="orange",
                    facecolor="none",
                    transform=ax.get_transform("pixel"),
                )
                ax.add_patch(circle)

        if title is None:
            title = (
                f"Calexp (Visit: {self.data_id['visit']}, Detector: {self.data_id['detector']}, Band: {self.band})"
                if isinstance(self.data_id, dict)
                else self.data_id
            )
        ax.set_title(title)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        elif close:
            plt.close(ax.figure)
            gc.collect()
        else:
            return ax

    def add_point(
        self, ax: plt.Axes, ra: float, dec: float, r: float = 5, c: str = "r", label: Optional[str] = None
    ) -> None:
        """
        Add a point (circle) to the plot at the given RA, Dec coordinates.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to add the point to.
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.
        r : float, optional
            Radius of the circle in pixels (default: 5).
        c : str, optional
            Color of the circle (default: 'r').
        label : str, optional
            Label for the circle.
        """
        x, y = self.sky_to_pix(ra, dec)
        circle = Circle((x, y), radius=r, edgecolor=c, facecolor="none", label=label)
        ax.add_patch(circle)

    def add_region(
        self, ax: plt.Axes, region: CircleSkyRegion, c: str = "r", label: Optional[str] = None, alpha: float = 0.5
    ) -> None:
        """
        Add a region to the plot as a polygon.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to add the region to.
        region : regions.CircleSkyRegion
            Sky region to plot.
        c : str, optional
            Color of the region boundary (default: 'r').
        label : str, optional
            Label for the region.
        alpha : float, optional
            Transparency of the region (default: 0.5).
        """
        ra_coords, dec_coords = region.boundary.xy
        pixel_vertices = [self.sky_to_pix(ra, dec) for ra, dec in zip(ra_coords, dec_coords)]
        polygon = pltPolygon(pixel_vertices, edgecolor=c, facecolor="none", alpha=alpha, label=label)
        ax.add_patch(polygon)

    def save_plot(self, ax: plt.Axes, image_path: str, show: bool = False) -> None:
        """
        Save the plot to a file.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to save.
        image_path : str
            Path to save the plot.
        show : bool, optional
            Display the plot before saving (default: False).
        """
        ax.figure.savefig(image_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(ax.figure)
        gc.collect()

    def get_sources(self, stdev: float = 5) -> afwTable.SourceCatalog:
        """
        Extract sources from the calexp using source detection.

        Parameters
        ----------
        stdev : float, optional
            Detection threshold in standard deviations (default: 5).

        Returns
        -------
        lsst.afw.table.SourceCatalog
            Detected sources.
        """
        schema = afwTable.SourceTable.makeMinimalSchema()
        schema.addField("coord_raErr", type="F", doc="RA error")
        schema.addField("coord_decErr", type="F", doc="Dec error")
        algMetadata = dafBase.PropertyList()
        config = SourceDetectionTask.ConfigClass()
        config.thresholdValue = stdev
        config.thresholdType = "stdev"
        detection_task = SourceDetectionTask(schema=schema, config=config)
        config = SingleFrameMeasurementTask.ConfigClass()
        measurement_task = SingleFrameMeasurementTask(
            schema=schema, config=config, algMetadata=algMetadata
        )
        tab = afwTable.SourceTable.make(schema)
        result = detection_task.run(tab, self.expF)
        sources = result.sources
        measurement_task.run(measCat=sources, exposure=self.expF)
        return sources

    def save_fits(self, path: str) -> None:
        """
        Save the calexp as a FITS file.

        Parameters
        ----------
        path : str
            Path to save the FITS file.
        """
        self.expF.writeFits(path)

    def flux_to_mag(self, flux: float, flux_err: float) -> Tuple[float, float]:
        """
        Convert flux to magnitude.

        Parameters
        ----------
        flux : float
            Flux value.
        flux_err : float
            Flux error.

        Returns
        -------
        Tuple[float, float]
            Magnitude and magnitude error.
        """
        photoCalib = self.expF.getPhotoCalib()
        measure = photoCalib.instFluxToMagnitude(flux, flux_err)
        return measure.value, measure.error

    def get_data(self, catalog: str = "CcdVisit") -> pd.DataFrame:
        """
        Retrieve metadata for the calexp from a catalog.

        Parameters
        ----------
        catalog : str, optional
            Catalog to query (default: 'CcdVisit').

        Returns
        -------
        pd.DataFrame
            Metadata for the calexp.

        Raises
        ------
        RuntimeError
            If query fails.
        """
        service = get_tap_service("tap")
        full_catalog = f"{self.tap_schema}.{catalog}"
        visit = self.data_id["visit"]
        detector = self.data_id["detector"]

        query = (
            f"SELECT visitId, detector, expMidptMJD, seeing, band "
            f"FROM {full_catalog} "
            f"WHERE visitId = {visit} AND detector = {detector} "
            "ORDER BY expMidptMJD ASC"
        )

        job = submit_with_retries(service, query)
        job.run()
        job.wait(phases=["COMPLETED", "ERROR"])
        if job.phase == "ERROR":
            job.raise_if_error()
        return job.fetch_result().to_table().to_pandas()