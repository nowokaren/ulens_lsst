"""
Utilities module for the LSST pipeline project.

This module contains utility functions supporting the versatile pipeline for experiments
in the context of LSST data processing, simulations, and analysis. It includes tools for
memory management, logging configuration, astronomical coordinate matching, catalog querying,
and handling Rubin Observatory simulation data.

The functions are designed to be reusable across the pipeline, facilitating experiments
such as microlensing light curve recoverability, blending effects, and comparisons between
simulations and real data.

Functions
---------
check_memory : Check current memory usage and raise an error if above a threshold.
setup_logger : Set up a consistent logger across processes.
get_nearby_objects : Find nearest coordinates in a catalog for given targets.
sky_catalog_query : Generate a query for astronomical catalogs based on regions and filters.
baseline_name : Get the name of the baseline OpSim database.
get_dataSlice : Retrieve observation data slice for a given sky position.
get_lsst_noise_for_lc : Compute LSST-like photometric errors for light curves.
get_lsst_mjds_per_band : Get observation MJDs per LSST band for a sky location.
"""

# Standard library imports
import logging
import sys

# Third-party imports
import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from psutil import Process
from typing import Dict, Any, List, Tuple

from rubin_sim.phot_utils.bandpass import Bandpass
from rubin_sim.phot_utils import calc_mag_error_m5
from rubin_sim.phot_utils.photometric_parameters import PhotometricParameters
from rubin_sim.data import get_baseline
# from bandpass_dict import BandpassDict
import rubin_sim.maf as maf


def check_memory(threshold_mb: float = 2000) -> None:
    """
    Check current memory usage and raise an error if above threshold.

    Parameters
    ----------
    threshold_mb : float, optional
        Memory usage threshold in MB (default: 2000).

    Raises
    ------
    MemoryError
        If memory usage exceeds the threshold.
    """
    memory_usage = Process().memory_info().rss / 1024**2  # Convert to MB
    if memory_usage > threshold_mb:
        raise MemoryError(
            f"Memory usage ({memory_usage:.2f} MB) exceeds threshold ({threshold_mb} MB)"
        )
    print(f"Current memory usage: {memory_usage:.2f} MB")


def setup_logger(
    log_file: str, log_to_console: bool = True, logger_name: str = "pipeline"
) -> logging.Logger:
    """
    Set up a consistent logger across processes, capturing warnings and external logs.

    Parameters
    ----------
    log_file : str
        Path to the log file.
    log_to_console : bool, optional
        Whether to log to console (default: True).
    logger_name : str, optional
        Name of the logger (default: "pipeline").

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Avoid duplicated handlers
    logger.propagate = False  # Prevent messages from being sent to the root logger

    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] [%(name)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (optional)
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Capture warnings
    logging.captureWarnings(True)

    # Clean up root logger to avoid unwanted prints
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.handlers.clear()  # Remove all existing handlers
    root_logger.addHandler(file_handler)  # Log warnings to file only

    # External module loggers
    lsst_logger = logging.getLogger("lsst")
    lsst_logger.setLevel(logging.INFO)
    lsst_logger.handlers.clear()  # Avoid double handlers
    lsst_logger.addHandler(file_handler)
    lsst_logger.propagate = False  # Prevent propagation to root

    return logger


def get_nearby_objects(
    coords_array_1: np.ndarray, coords_array_2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each coordinate in coords_array_1, find the nearest coordinate in coords_array_2.

    Parameters
    ----------
    coords_array_1 : ndarray
        Array Nx2 of RA, Dec in degrees (targets).
    coords_array_2 : ndarray
        Array Mx2 of RA, Dec in degrees (catalog to match against).

    Returns
    -------
    indices : ndarray
        Array of indices of the nearest neighbor in coords_array_2 for each coord in coords_array_1.
    distances_arcsec : ndarray
        Array of angular distances in arcseconds.
    nearest_coords : ndarray
        Array Nx2 with RA, Dec of the nearest sources in coords_array_2.
    """
    targets = SkyCoord(ra=coords_array_1[:, 0] * u.deg, dec=coords_array_1[:, 1] * u.deg)
    catalog = SkyCoord(ra=coords_array_2[:, 0] * u.deg, dec=coords_array_2[:, 1] * u.deg)
    indices, d2d, _ = match_coordinates_sky(targets, catalog)
    nearest_coords = np.column_stack((catalog[indices].ra.deg, catalog[indices].dec.deg))
    distances_arcsec = d2d.to(u.arcsec).value
    return indices, distances_arcsec, nearest_coords


def sky_catalog_query(
    catalog: str,
    columns: List[str],
    regions: List[Tuple[float, float, float]],
    filters: List[str],
    language: str = "sql",
) -> str:
    """
    Generate a query for astronomical catalogs based on spatial regions and filters.

    Parameters
    ----------
    catalog : str
        Name of the catalog table.
    columns : list of str
        List of column names to select.
    regions : list of tuples
        List of (ra, dec, radius) in degrees for spatial queries.
    filters : list of str
        List of additional filter conditions (e.g., 'mag < 20').
    language : str, optional
        Query language: 'sql' (uses q3c_radial_query) or 'adql' (uses CONTAINS + CIRCLE) (default: 'sql').

    Returns
    -------
    str
        The generated query string.

    Raises
    ------
    ValueError
        If language is not 'sql' or 'adql'.
    """
    if language == "sql":
        # SQL version with q3c_radial_query
        spatial_clauses = [
            f"q3c_radial_query(ra, dec, {ra}, {dec}, {radius})"
            for ra, dec, radius in regions
        ]
    elif language == "adql":
        # ADQL version with CONTAINS + CIRCLE
        spatial_clauses = [
            f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius}))=1"
            for ra, dec, radius in regions
        ]
    else:
        raise ValueError("language must be 'sql' or 'adql'")

    spatial_part = " OR ".join(spatial_clauses)

    # Additional filters
    filter_part = " AND ".join(filters)
    if filters and spatial_part:
        where_statement = f"({spatial_part}) AND ({filter_part})"
    elif spatial_part:
        where_statement = spatial_part
    elif filters:
        where_statement = filter_part
    else:
        where_statement = "1=1"

    query = f"""
    SELECT {', '.join(columns)}
    FROM {catalog}
    WHERE {where_statement}
    """
    return query


def baseline_name() -> str:
    """
    Get the name of the baseline OpSim database.

    Returns
    -------
    str
        Baseline filename without extension (e.g., 'baseline_v2.0_10yrs').
    """
    return get_baseline().split("/")[-1].replace(".db", "")


def get_dataSlice(ra: float, dec: float, opsim: Any = None) -> Dict[str, Any]:
    """
    Retrieve observation data slice for a given sky position from OpSim.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    opsim : Any, optional
        OpSim database path or object (default: baseline).

    Returns
    -------
    dict
        Dictionary containing observation data.
    """
    if opsim is None:
        opsim = get_baseline()

    # Ensure RA and Dec are scalars
    ra = float(ra)
    dec = float(dec)

    metric = maf.metrics.PassMetric(cols=["filter", "observationStartMJD", "fiveSigmaDepth"])
    slicer = maf.slicers.UserPointsSlicer(ra=[ra], dec=[dec])
    sql_constraint = ""
    bundle = maf.MetricBundle(metric=metric, slicer=slicer, constraint=sql_constraint)
    bg = maf.MetricBundleGroup([bundle], opsim, out_dir="temp")
    bg.run_all()
    return bundle.metric_values[0]


def get_lsst_noise_for_lc(
    lc: Dict[str, Dict[str, List[float]]],
    ra: float,
    dec: float,
    opsim: str = "baseline",
    mjd_range: Tuple[float, float] = (60849, 60849 + 3 * 365.25),
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute LSST-like photometric errors for given light curves.

    Parameters
    ----------
    lc : dict
        Dictionary with keys like 'u', 'g', 'r', 'i', 'z', 'y',
        each containing a dict with keys 'time' and 'mag'.
    ra : float
        Right Ascension of the object (degrees).
    dec : float
        Declination of the object (degrees).
    opsim : str or Any, optional
        Rubin Sim opsim database or "baseline" (default: "baseline").
    mjd_range : tuple of float, optional
        MJD range to consider (default: (60849, 60849 + 3*365.25)).

    Returns
    -------
    dict
        Dictionary with keys for each band. Each value is a dict with keys 'mag' and 'mag_err' containing lists.
    """
    if opsim == "baseline":
        opsim = get_baseline()

    # Load photometric info
    photParams = PhotometricParameters(exptime=30, nexp=1, readnoise=None)
    # LSST_BandPass = BandpassDict.load_total_bandpasses_from_files(
    #     bandpass_dir="../roman_rubin/troughputs"
    # )
    dataSlice = get_dataSlice(ra, dec, opsim)

    result = {}
    for band in lc:
        mag_list = lc[band]["mag"]
        time_list = lc[band]["time"]
        time_array = np.array(time_list)
        mag_array = np.array(mag_list)

        # Match MJDs in opsim to those in input (approximate match)
        band_mask = (
            (dataSlice["filter"] == band.lower())
            & (dataSlice["observationStartMJD"] >= mjd_range[0])
            & (dataSlice["observationStartMJD"] <= mjd_range[1])
        )
        mjd_opsim = dataSlice["observationStartMJD"][band_mask]
        m5_opsim = dataSlice["fiveSigmaDepth"][band_mask]

        # Interpolate m5 to match input times
        if len(mjd_opsim) < 2:
            print(
                f"[Warning] Not enough observations in band {band} between MJDs {mjd_range}. Skipping."
            )
            continue

        m5_interp = np.interp(time_array, mjd_opsim, m5_opsim)
        mag_err = [
            calc_mag_error_m5(mag, LSST_BandPass[band.lower()], m5, photParams)[0]
            for mag, m5 in zip(mag_array, m5_interp)
        ]

        # Add Gaussian noise
        mag_noisy = [np.random.normal(m, e) for m, e in zip(mag_array, mag_err)]

        result[band] = {"mag": mag_noisy, "mag_err": mag_err}

    return result


def get_lsst_mjds_per_band(
    ra: float,
    dec: float,
    bands: Any = "all",
    mjd_range: Tuple[float, float] = (60849, 60849 + 3 * 365.25),
    opsim: str = "baseline",
) -> Dict[str, np.ndarray]:
    """
    Get observation MJDs per LSST band for a given sky location.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    bands : str or list, optional
        'all' or a list of bands to include (e.g. ['r', 'i']) (default: 'all').
    mjd_range : tuple of float, optional
        MJD range to include (default: (60849, 60849 + 3*365.25)).
    opsim : str or Any, optional
        Rubin Sim opsim database or "baseline" (default: "baseline").

    Returns
    -------
    dict
        Dictionary with band names as keys and arrays of MJDs as values.
    """
    if opsim == "baseline":
        opsim = get_baseline()

    if bands == "all":
        bands = ["u", "g", "r", "i", "z", "y"]

    # Get observation data
    slicer = maf.slicers.UserPointsSlicer(ra=[ra], dec=[dec])
    metric = maf.metrics.PassMetric(cols=["filter", "observationStartMJD"])
    bundle = maf.MetricBundle(metric, slicer, "")
    group = maf.MetricBundleGroup([bundle], opsim, out_dir="temp")
    group.run_all(plot_now=False)
    dataSlice = bundle.metric_values[0]

    result = {}
    for band in bands:
        mask = (
            (dataSlice["filter"] == band.lower())
            & (dataSlice["observationStartMJD"] >= mjd_range[0])
            & (dataSlice["observationStartMJD"] <= mjd_range[1])
        )
        result[band] = np.array(sorted(dataSlice["observationStartMJD"][mask]))

    return result