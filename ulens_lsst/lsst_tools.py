"""
LSST Tools module for the LSST pipeline project.

This module provides tools for handling source injection and photometry measurement
in LSST calibrated exposures. It is designed to support experiments in the LSST
data processing pipeline, including source injection into simulations (DP0) or
real data samples (DP1), and photometric measurements. The primary class,
LSSTTools, encapsulates the functionality for setting up tasks, creating injection
catalogs, injecting sources, measuring photometry, and processing calibrated
exposures.

The module aims to be versatile for scientific experiments such as studying
microlensing light curve recoverability, blending effects, and comparisons between
simulations and real data. It is optimized for efficiency in time and memory usage,
with clear and concise code.

Classes
-------
LSSTTools : Handles source injection and photometry measurement for LSST calibrated exposures.

Functions
---------
None (all functionality is encapsulated in the LSSTTools class).
"""

# Standard library imports
from typing import List, Dict, Union, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from astropy.table import Table

# LSST imports
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import lsst.geom as geom
from lsst.source.injection import VisitInjectConfig, VisitInjectTask
from lsst.meas.base import ForcedMeasurementTask

# Local imports
from ulens_lsst.lsst_data import Calexp


class LSSTTools:
    """
    A class to handle source injection and photometry measurement for LSST calibrated exposures.

    This class provides methods to set up schemas and tasks for injection and measurement,
    create injection catalogs, inject sources into exposures, measure photometry, and process
    calibrated exposures. It is designed to be memory-efficient by avoiding unnecessary copies
    and using direct array operations where possible.
    """

    def __init__(
        self,
        calexp: Calexp,
        injection_config: Optional[Dict] = None,
        measurement_config: Optional[Dict] = None,
    ):
        """
        Initialize the LSSTTools class with a calibrated exposure and optional task configurations.

        Parameters
        ----------
        calexp : Calexp
            A Calexp object containing the exposure to process.
        injection_config : dict, optional
            Configuration for VisitInjectTask. If None, uses defaults.
        measurement_config : dict, optional
            Configuration for ForcedMeasurementTask. If None, uses defaults.
        """
        self.calexp = calexp
        self.exposure = calexp.expF
        self.wcs = calexp.wcs
        self.photo_calib = self.exposure.getPhotoCalib()
        self.psf = self.exposure.getPsf()
        self.schema = None
        self.injection_task = None
        self.forced_measurement_task = None
        self._setup_schema()
        self._setup_injection_task(injection_config)
        self._setup_forced_measurement_task(measurement_config)

    def _setup_schema(self) -> None:
        """Set up the schema for forced measurement."""
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        alias = self.schema.getAliasMap()
        self.schema.addField("centroid_x", type="D", doc="Centroid x-coordinate in pixels")
        self.schema.addField("centroid_y", type="D", doc="Centroid y-coordinate in pixels")
        alias.set("slot_Centroid", "centroid")
        self.schema.addField("shape_xx", type="D", doc="Shape xx moment")
        self.schema.addField("shape_yy", type="D", doc="Shape yy moment")
        self.schema.addField("shape_xy", type="D", doc="Shape xy moment")
        alias.set("slot_Shape", "shape")
        self.schema.addField("type_flag", type="F", doc="Source type flag")
        self.schema.addField("psfFlux", type="D", doc="PSF flux")
        self.schema.addField("psfFluxErr", type="D", doc="Error in PSF flux")
        self.schema.addField("psfFlux_flag", type="Flag", doc="Flag for PSF flux measurement")

    def _setup_injection_task(self, config: Optional[Dict] = None) -> None:
        """Configure the injection task with optional custom settings."""
        injection_config = VisitInjectConfig()
        if config:
            for key, value in config.items():
                setattr(injection_config, key, value)
        self.injection_task = VisitInjectTask(config=injection_config)

    def _setup_forced_measurement_task(self, config: Optional[Dict] = None) -> None:
        """Configure the forced measurement task with optional custom settings."""
        forced_config = ForcedMeasurementTask.ConfigClass()
        forced_config.copyColumns = {}
        forced_config.plugins.names = [
            "base_TransformedCentroid",
            "base_PsfFlux",
            "base_TransformedShape",
        ]
        forced_config.doReplaceWithNoise = False
        if config:
            for key, value in config.items():
                setattr(forced_config, key, value)
        self.forced_measurement_task = ForcedMeasurementTask(
            self.schema, config=forced_config
        )

    def create_injection_catalog(
        self,
        ra_values: List[float],
        dec_values: List[float],
        mag_values: List[float],
        magnification_values: Optional[List[float]] = None,
        event_ids: Optional[List[int]] = None,
        expMidptMJD: Optional[float] = None,
        visit: Optional[int] = None,
        detector: Optional[int] = None,
        source_type: str = "Star",
    ) -> Table:
        """
        Create a pandas DataFrame for source injection with RA, Dec, and magnitude values.

        This function constructs a DataFrame with the provided RA, Dec, and magnitude values,
        assigning a unique injection ID to each source. Optional columns for magnification,
        event_ids, expMidptMJD, visit, and detector are included only if provided. All sources
        are assigned the specified source_type. Uses numpy for efficient array creation to
        optimize memory and time.

        Parameters
        ----------
        ra_values : List[float]
            List of Right Ascension values in degrees.
        dec_values : List[float]
            List of Declination values in degrees.
        mag_values : List[float]
            List of magnitude values for the sources.
        magnification_values : List[float], optional
            List of magnification values. If provided, adds a column.
        event_ids : List[int], optional
            List of event IDs. If provided, adds a column.
        expMidptMJD : float, optional
            MJD of the exposure midpoint. If provided, adds a column.
        visit : int, optional
            Visit ID for the exposure. If provided, adds a column.
        detector : int, optional
            Detector ID for the exposure. If provided, adds a column.
        source_type : str, optional
            Type of source (default: "Star").

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['injection_id', 'ra', 'dec', 'mag', 'source_type']
            and optional columns ['magnification', 'event_id', 'expMidptMJD', 'visit', 'detector'].

        Raises
        ------
        ValueError
            If input lists have unequal lengths or are empty.
        """
        # Validate input lengths
        lengths = [len(ra_values), len(dec_values), len(mag_values)]
        if magnification_values is not None:
            lengths.append(len(magnification_values))
        if event_ids is not None:
            lengths.append(len(event_ids))
        if len(set(lengths)) > 1:
            raise ValueError("All input lists must have equal lengths.")
        if lengths[0] == 0:
            raise ValueError("Input lists cannot be empty.")

        n_sources = lengths[0]
        injection_ids = np.arange(1, n_sources + 1)

        # Initialize data dictionary with required columns using numpy arrays for efficiency
        data = {
            "injection_id": injection_ids,
            "ra": np.array(ra_values, dtype=float),
            "dec": np.array(dec_values, dtype=float),
            "mag": np.array(mag_values, dtype=float),
            "source_type": np.full(n_sources, source_type, dtype="object"),
        }

        # Add optional columns if provided
        if magnification_values is not None:
            data["magnification"] = np.array(magnification_values, dtype=float)
        if event_ids is not None:
            data["event_id"] = np.array(event_ids, dtype=int)
        if expMidptMJD is not None:
            data["expMidptMJD"] = np.full(n_sources, expMidptMJD, dtype=float)
        if visit is not None:
            data["visit"] = np.full(n_sources, visit, dtype=int)
        if detector is not None:
            data["detector"] = np.full(n_sources, detector, dtype=int)

        return Table(data)

    def inject_sources(
        self, injection_catalog: pd.DataFrame
    ) -> Dict[str, Union[afwImage.ExposureF, afwTable.SourceCatalog]]:
        """
        Inject multiple point sources into the exposure.

        Clones the exposure only once per injection to minimize memory usage.

        Parameters
        ----------
        injection_catalog : pd.DataFrame
            DataFrame with columns ['injection_id', 'visit', 'detector', 'ra', 'dec', 'source_type', 'expMidptMJD', 'mag'].

        Returns
        -------
        dict
            Dictionary with 'output_exposure' (injected exposure) and 'output_catalog' (injected source catalog).

        Raises
        ------
        ValueError
            If injection_catalog is missing required columns.
        RuntimeError
            If injection fails.
        """
        required_columns = [
            "injection_id",
            "visit",
            "detector",
            "ra",
            "dec",
            "source_type",
            "expMidptMJD",
            "mag",
        ]
        missing_cols = set(required_columns) - set(injection_catalog.columns)
        if missing_cols:
            raise ValueError(f"injection_catalog missing columns: {missing_cols}")

        try:
            injected_output = self.injection_task.run(
                injection_catalogs=[injection_catalog],
                input_exposure=self.exposure.clone(),
                psf=self.psf,
                photo_calib=self.photo_calib,
                wcs=self.wcs,
            )
            return {
                "output_exposure": injected_output.output_exposure,
                "output_catalog": injected_output.output_catalog,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to inject sources: {str(e)}")

    def measure_photometry(
        self,
        injected_exposure: afwImage.ExposureF,
        injected_catalog: afwTable.SourceCatalog,
    ) -> pd.DataFrame:
        """
        Measure photometry for injected sources in the exposure.

        Optimizes by using vectorized operations for flux to magnitude conversions and flag processing.

        Parameters
        ----------
        injected_exposure : afwImage.ExposureF
            Exposure with injected sources.
        injected_catalog : afwTable.SourceCatalog
            Catalog of injected sources.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['event_id', 'time', 'band', 'ideal_mag', 'meas_mag', 'meas_mag_err',
            'meas_flux', 'meas_flux_err', 'magnification', 'injection_flag', 'measure_flag'].
            Returns empty DataFrame with None values if measurement fails or catalog is empty.
        """
        if len(injected_catalog) == 0:
            return pd.DataFrame(
                columns=[
                    "event_id",
                    "time",
                    "band",
                    "ideal_mag",
                    "meas_mag",
                    "meas_mag_err",
                    "meas_flux",
                    "meas_flux_err",
                    "magnification",
                    "injection_flag",
                    "measure_flag",
                ]
            )

        # Create source catalog for forced measurement efficiently
        forced_source = afwTable.SourceCatalog(self.schema)
        ra_array = injected_catalog["ra"]
        dec_array = injected_catalog["dec"]
        for i, source in enumerate(injected_catalog):
            source_rec = forced_source.addNew()
            coord = geom.SpherePoint(
                geom.Angle(ra_array[i], geom.degrees),
                geom.Angle(dec_array[i], geom.degrees),
            )
            source_rec.setCoord(coord)
            x, y = self.calexp.sky_to_pix(ra_array[i], dec_array[i])
            source_rec["centroid_x"] = x
            source_rec["centroid_y"] = y
            source_rec["type_flag"] = 0  # Star type

        # Run forced measurement
        new_calexp = Calexp(injected_exposure)
        forced_meas_cat = self.forced_measurement_task.generateMeasCat(
            new_calexp.expF, forced_source, new_calexp.wcs
        )
        self.forced_measurement_task.run(
            measCat=forced_meas_cat,
            exposure=new_calexp.expF,
            refCat=forced_source,
            refWcs=new_calexp.wcs,
        )

        # Convert results to Astropy Table and add columns
        table = forced_meas_cat.asAstropy()
        table["coord_ra"] = ra_array
        table["coord_dec"] = dec_array

        # Add optional columns if present
        if "event_id" in injected_catalog.colnames:
            table["event_id"] = injected_catalog["event_id"]
        else:
            table["event_id"] = np.full(len(table), np.nan)
        if "injection_id" in injected_catalog.colnames:
            table["injection_id"] = injected_catalog["injection_id"]
        else:
            table["injection_id"] = np.full(len(table), np.nan)
        if "injection_flag" in injected_catalog.colnames:
            table["injection_flag"] = injected_catalog["injection_flag"]
        else:
            table["injection_flag"] = np.full(len(table), False)
        if "magnification" in injected_catalog.colnames:
            table["magnification"] = injected_catalog["magnification"]
        else:
            table["magnification"] = np.full(len(table), np.nan)

        # Process flags vectorized
        measure_flag = []
        flag_cols = [col for col in table.columns if "flag" in col]
        for row in table:
            flags = [col for col in flag_cols if row[col]]
            measure_flag.append("-".join(flags))
        table["measure_flag"] = measure_flag

        # Extract photometry vectorized
        flux = table["base_PsfFlux_instFlux"]
        flux_err = table["base_PsfFlux_instFluxErr"]
        mags_and_errs = np.array(
            [
                new_calexp.flux_to_mag(f, fe)
                for f, fe in zip(flux, flux_err)
            ],
            dtype=object,
        )
        meas_mag = [m[0] for m in mags_and_errs]
        meas_mag_err = [m[1] for m in mags_and_errs]

        # # Create extraction DataFrame
        # extraction_data = {
        #     "event_id": table["event_id"],
        #     "time": injected_catalog["expMidptMJD"],
        #     "band": np.full(len(injected_catalog), self.calexp.band),
        #     "ideal_mag": injected_catalog["mag"],
        #     "meas_mag": meas_mag,
        #     "meas_mag_err": meas_mag_err,
        #     "meas_flux": flux,
        #     "meas_flux_err": flux_err,
        #     "injection_flag": table["injection_flag"],
        #     "measure_flag": table["measure_flag"],
        # }
        # if "magnification" in injected_catalog.schema.getNames():
        #     extraction_data["magnification"] = injected_catalog["magnification"]
        # else:
        #     extraction_data["magnification"] = np.full(len(table), np.nan)

        extraction_table = Table()
        extraction_table["event_id"] = table["event_id"]
        extraction_table["time"] = np.full(len(table), injected_catalog["expMidptMJD"])
        extraction_table["band"] = np.full(len(table), self.calexp.band)
        extraction_table["ideal_mag"] = injected_catalog["mag"]
        extraction_table["meas_mag"] = meas_mag
        extraction_table["meas_mag_err"] = meas_mag_err
        extraction_table["meas_flux"] = flux
        extraction_table["meas_flux_err"] = flux_err
        extraction_table["injection_flag"] = table["injection_flag"]
        extraction_table["measure_flag"] = table["measure_flag"]
        extraction_table["magnification"] = table["magnification"]
    
        return extraction_table


    def process_calexp(
        self, injection_catalog: pd.DataFrame
    ) -> Dict[str, List[Dict]]:
        """
        Process a calexp by injecting sources and measuring photometry.

        Handles exceptions at injection and measurement stages, returning structured results
        for Parquet storage. Optimizes by reusing data structures and avoiding redundant computations.

        Parameters
        ----------
        injection_catalog : pd.DataFrame
            DataFrame with columns ['event_id', 'ra', 'dec', 'ideal_mag', 'band', 'time', 'source_type'].

        Returns
        -------
        dict
            Dictionary with 'event_results' and 'photometry_results' lists.
        """
        event_results = []
        photometry_results = []

        # Inject sources
        try:
            injection_output = self.inject_sources(injection_catalog)
            injected_exposure = injection_output["output_exposure"]
            injected_catalog = injection_output["output_catalog"]
        except Exception as e:
            # Efficiently create failed results using numpy
            n = len(injection_catalog)
            event_ids = injection_catalog["event_id"].to_numpy()
            times = injection_catalog["time"].to_numpy()
            bands = injection_catalog["band"].to_numpy()
            ideal_mags = injection_catalog["ideal_mag"].to_numpy()
            for i in range(n):
                event_results.append(
                    {
                        "event_id": event_ids[i],
                        "status": "failed",
                        "error": f"Injection failed: {str(e)}",
                    }
                )
                photometry_results.append(
                    {
                        "event_id": event_ids[i],
                        "time": times[i],
                        "band": bands[i],
                        "ideal_mag": ideal_mags[i],
                        "meas_mag": None,
                        "meas_mag_err": None,
                        "meas_flux": None,
                        "meas_flux_err": None,
                        "flag": "InjectionError",
                    }
                )
            return {"event_results": event_results, "photometry_results": photometry_results}

        # Measure photometry
        extraction = self.measure_photometry(injected_exposure, injected_catalog)

        # Create event and photometry results efficiently
        for _, row in extraction.iterrows():
            flag = row["measure_flag"]
            status = "failed" if flag else "success"
            error = "Measurement flagged" if flag else ""
            event_results.append(
                {"event_id": row["event_id"], "status": status, "error": error}
            )
            photometry_results.append(
                {
                    "event_id": row["event_id"],
                    "time": row["time"],
                    "band": row["band"],
                    "ideal_mag": row["ideal_mag"],
                    "meas_mag": row["meas_mag"],
                    "meas_mag_err": row["meas_mag_err"],
                    "meas_flux": row["meas_flux"],
                    "meas_flux_err": row["meas_flux_err"],
                    "flag": "MeasurementError" if flag else "",
                }
            )

        return {"event_results": event_results, "photometry_results": photometry_results}