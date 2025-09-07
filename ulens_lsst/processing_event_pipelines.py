"""
Event processing functions for the LSST pipeline project.

This module provides functions to simulate and process microlensing and non-microlensing events,
designed for integration with :class:`simulation_pipeline.SimPipeline` and
:class:`parallel_utils.ParallelProcessor`. It supports generating light curves for constant
(cte) and microlensing (ulens, SNANA-specific) events, performing photometry measurements,
and computing chi-squared statistics. The module is optimized for memory efficiency by using
selective data loading and pyarrow for Parquet operations, and for time efficiency through
streamlined processing and robust error handling.

Functions
---------
process_cte_event : Processes a constant light curve event from TRILEGAL data.
process_ulens_event : Processes a microlensing event without additional filtering.
process_SNANA_ulens_event : Processes a microlensing event with SNANA-specific filters.
LSST_synthetic_photometry : Processes a calexp by injecting sources and measuring photometry.
compute_chi2 : Computes chi-squared statistics for an event and saves results.
"""

# Standard library imports
import logging
import os
import traceback
from typing import Dict, Any, Tuple
from copy import copy
import gc

# Third-party imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
from astropy import units as u
from regions import CircleSkyRegion

# Local imports
from ulens_lsst.catalogs_utils import Catalog
from ulens_lsst.light_curves import Event
from ulens_lsst.lsst_data import LSSTData, Calexp
from ulens_lsst.lsst_tools import LSSTTools
from ulens_lsst.utils import get_nearby_objects


def process_cte_event(row: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a constant light curve event based on TRILEGAL data.

    Generates a constant light curve using baseline magnitudes from a TRILEGAL chunk
    and saves results to temporary Parquet files.

    Parameters
    ----------
    row : pd.Series
        Input row with columns 'event_id', 'ra', 'dec'.
    config : Dict[str, Any]
        Configuration with keys:
        - temp_dir: Directory for temporary files.
        - peak_range: Tuple of (start, end) MJD for simulation.
        - bands: List of filter bands.
        - mag_sat: Dictionary of saturation magnitudes per band.
        - m_5sigma: Dictionary of 5-sigma depth magnitudes per band.
        - log_name: Logger name.

    Returns
    -------
    Dict[str, Any]
        Result dictionary with 'event_id', 'status', and optional 'error'.

    Raises
    ------
    Exception
        Captures and logs any processing errors, returning them in the result.
    """
    logger = logging.getLogger(config.get("log_name", "pipeline"))
    try:
        event_id = row["event_id"]
        ra = row["ra"]
        dec = row["dec"]
        bands = config["bands"]
        chunk_number = event_id // 10000 + 1
        row_index = event_id % 10000
        chunk_path = f"../roman_rubin/chunks_TRILEGAL_GENULENS/TRILEGAL_chunk_{chunk_number}.csv"

        if not os.path.exists(chunk_path):
            return {"event_id": event_id, "status": "failed", "error": f"Chunk file not found: {chunk_path}"}

        trilegal_bands = [band if band != "y" else "Y" for band in bands]
        chunk_df = pd.read_csv(chunk_path, skiprows=range(1, row_index + 1), nrows=1, usecols=trilegal_bands)
        chunk_df.rename(columns={"Y": "y"}, inplace=True)
        baseline = chunk_df.iloc[0].to_dict()

        valid_bands = [band for band in bands if config["mag_sat"][band] < baseline[band] < config["m_5sigma"][band] + 1]
        if not valid_bands:
            return {"event_id": event_id, "status": "skipped", "error": "No valid band"}

        times = np.linspace(config["peak_range"][0], config["peak_range"][0] + 10, 10)
        photometry_rows = [
            {
                "event_id": event_id,
                "time": t,
                "band": band,
                "ideal_mag": baseline[band],
                "meas_mag": np.nan,
                "meas_mag_err": np.nan,
                "meas_flux": np.nan,
                "meas_flux_err": np.nan,
                "magnification": np.nan,
                "injection_flag": None,
                "measure_flag": None,
            }
            for band in bands
            for t in times
        ]
        photometry_df = pd.DataFrame(photometry_rows)
        event_df = pd.DataFrame([{
            "event_id": event_id,
            "ra": ra,
            "dec": dec,
            "model": f"trilegal_chunk{chunk_number}_row{row_index}",
            "system_type": "none",
            **{f"param_{band}": baseline[band] for band in baseline},
        }])

        photometry_path = os.path.join(config["temp_dir"], f"temp_photometry_{event_id}.parquet")
        event_path = os.path.join(config["temp_dir"], f"temp_data-events_{event_id}.parquet")
        pq.write_table(pa.Table.from_pandas(photometry_df), photometry_path, compression="snappy")
        pq.write_table(pa.Table.from_pandas(event_df), event_path, compression="snappy")
        return {"event_id": event_id, "status": "success", "error": ""}
    except Exception as e:
        logger.error(f"Event {row['event_id']}: Failed with error: {str(e)}")
        return {"event_id": row["event_id"], "status": "failed", "error": traceback.format_exc()}


def process_ulens_event(row: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a microlensing event without additional filtering.

    Simulates microlensing parameters and light curves, saving results to temporary Parquet files.

    Parameters
    ----------
    row : pd.Series
        Input row with columns 'event_id', 'ra', 'dec', 'bands', 'model', 'system_type'.
    config : Dict[str, Any]
        Configuration with keys:
        - temp_dir: Directory for temporary files.
        - peak_range: Tuple of (start, end) MJD for simulation.
        - epochs: Dictionary of observation times per band.
        - cadence_noise: Source of cadence and noise ('dp0', 'dp1', 'ideal').
        - pylima_blend: Blending parameter for pyLIMA.
        - photometry_schema: Schema for photometry Parquet files.
        - events_schema: Schema for events Parquet files.
        - sources_catalog: Source catalog ('TRILEGAL' or CSV path).
        - log_name: Logger name.

    Returns
    -------
    Dict[str, Any]
        Result dictionary with 'event_id', 'status', and optional 'error'.
    """
    logger = logging.getLogger(config.get("log_name", "pipeline"))
    try:
        logger.info(f"Event {row['event_id']}: Starting")
        event_id = row["event_id"]
        bands = config["bands"]
        epochs = config["epochs"]
        cadence_noise = config["cadence_noise"]

        event = Event(
            event_id=event_id,
            ra=row["ra"],
            dec=row["dec"],
            bands=bands,
            model=row["model"],
            system_type=row["system_type"],
            parallax=True,
            cadence_noise=cadence_noise,
            photometry_schema=config["photometry_schema"],
            events_schema=config["events_schema"],
            sources_catalog=config["sources_catalog"],
        )

        logger.info(f"Event {row['event_id']}: Simulating parameters")
        event.simulate_ulens_parameters(peak_range=config["peak_range"], blend=config["pylima_blend"])

        logger.info(f"Event {row['event_id']}: Simulating light curve")
        event.simulate_lc(epochs)
        if event.photometry.empty:
            logger.info(f"Event {row['event_id']}: Failed (empty photometry)")
            return {"event_id": event_id, "status": "failed", "error": "Empty photometry data"}

        logger.info(f"Event {row['event_id']}: Saving event")
        event.to_parquet(
            os.path.join(config["temp_dir"], f"temp_photometry_{event_id}.parquet"),
            os.path.join(config["temp_dir"], f"temp_data-events_{event_id}.parquet"),
        )
                # calexp_photometry_file = os.path.join(config["temp_dir"], f"temp_calexps-photometry_{calexp_id}.parquet")
        del event
        gc.collect()
        logger.info(f"Event {row['event_id']}: Success")
        return {"event_id": event_id, "status": "success", "error": ""}
    except Exception as e:
        logger.error(f"Event {row['event_id']}: Failed with error: {str(e)}")
        return {"event_id": row["event_id"], "status": "failed", "error": traceback.format_exc()}


def process_SNANA_ulens_event(row: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a microlensing event with SNANA-specific filters.

    Applies filters to ensure light curves meet SNANA requirements (e.g., sufficient significant
    points, magnitude constraints) before saving to temporary Parquet files.

    Parameters
    ----------
    row : pd.Series
        Input row with columns 'event_id', 'ra', 'dec', 'bands', 'model', 'system_type'.
    config : Dict[str, Any]
        Configuration with keys:
        - temp_dir: Directory for temporary files.
        - survey_dates: Tuple of (start, end) MJD for survey.
        - peak_range: Tuple of (start, end) MJD for peak simulation.
        - epochs: Dictionary of observation times per band.
        - cadence_noise: Source of cadence and noise.
        - m_sat: Dictionary of saturation magnitudes per band.
        - m_5sigma: Dictionary of 5-sigma depth magnitudes per band.
        - pylima_blend: Blending parameter for pyLIMA.
        - photometry_schema: Schema for photometry Parquet files.
        - events_schema: Schema for events Parquet files.
        - sources_catalog: Source catalog ('TRILEGAL' or CSV path).
        - log_name: Logger name.

    Returns
    -------
    Dict[str, Any]
        Result dictionary with 'event_id', 'status', and optional 'error'.
    """
    logger = logging.getLogger(config.get("log_name", "pipeline"))
    try:
        logger.info(f"Event {row['event_id']}: Starting")
        event_id = row["event_id"]
        bands = config["bands"]
        epochs = config["epochs"]
        cadence_noise = config["cadence_noise"]
        m_sat = config["m_sat"]
        m_5sigma = config["m_5sigma"]
        f = config.get("f", 0.0001)

        event = Event(
            event_id=event_id,
            ra=row["ra"],
            dec=row["dec"],
            bands=bands,
            model=row["model"],
            system_type=row["system_type"],
            parallax=True,
            cadence_noise=cadence_noise,
            photometry_schema=config["photometry_schema"],
            events_schema=config["events_schema"],
            sources_catalog=config["sources_catalog"],
        )

        logger.info(f"Event {row['event_id']}: Simulating parameters")
        event.simulate_ulens_parameters(peak_range=config["peak_range"], blend=config["pylima_blend"])
        baseline_mags = event.source_data
        all_faint = all(baseline_mags[band] >= m_5sigma[band] + 1 for band in bands)
        if all_faint:
            logger.info(f"Event {row['event_id']}: Failed (all baseline magnitudes too faint)")
            return {"event_id": event_id, "status": "failed", "error": "All baseline magnitudes too faint"}

        logger.info(f"Event {row['event_id']}: Simulating initial light curve")
        limit_epoch = {band: np.array(config["survey_dates"]) for band in bands}
        event.simulate_lc(limit_epoch)

        for band in bands:
            band_data = event.photometry[event.photometry["band"] == band]
            if band_data.empty:
                logger.info(f"Event {row['event_id']}: Failed (no data for band {band})")
                return {"event_id": event_id, "status": "failed", "error": f"No data for band {band}"}
            first_epoch = band_data["time"].min()
            last_epoch = band_data["time"].max()
            first_mag = band_data.loc[band_data["time"] == first_epoch, "ideal_mag"].iloc[0]
            last_mag = band_data.loc[band_data["time"] == last_epoch, "ideal_mag"].iloc[0]
            template_mag = baseline_mags[band]
            if abs(first_mag - template_mag) / template_mag > f * 1.5:
                logger.info(f"Event {row['event_id']}: Failed (first magnitude too different)")
                return {"event_id": event_id, "status": "failed", "error": "First magnitude too different"}
            if abs(last_mag - template_mag) / template_mag > f * 1.5:
                logger.info(f"Event {row['event_id']}: Failed (last magnitude too different)")
                return {"event_id": event_id, "status": "failed", "error": "Last magnitude too different"}

        logger.info(f"Event {row['event_id']}: Simulating full light curve")
        event.simulate_lc(epochs)
        event.photometry["ideal_mag"] = event.photometry["ideal_mag"].round(3)

        bad_bands = sum(
            1 for band in bands if (event.photometry[event.photometry["band"] == band]["ideal_mag"] > m_sat[band]).sum() <= 10
        )
        if bad_bands == len(bands):
            logger.info(f"Event {row['event_id']}: Failed (fewer than 10 epochs after saturation)")
            return {"event_id": event_id, "status": "failed", "error": "Fewer than 10 epochs after saturation"}

        sig_times = []
        bad_bands = 0
        for band in bands:
            band_data = event.photometry[event.photometry["band"] == band]
            baseline = baseline_mags[band]
            mask = abs(band_data["ideal_mag"] - baseline) / baseline >= f
            true_times = band_data.loc[mask, "time"].values
            if len(true_times) < 10:
                bad_bands += 1
            else:
                sig_times.extend(true_times)
        if bad_bands == len(bands):
            logger.info(f"Event {row['event_id']}: Failed (less than 10 significant points)")
            return {"event_id": event_id, "status": "failed", "error": "Less than 10 significant points"}

        min_time, max_time = min(sig_times), max(sig_times)
        event.photometry = event.photometry[
            (event.photometry["time"] >= min_time) & (event.photometry["time"] <= max_time)
        ].reset_index(drop=True)

        for band in bands:
            band_data = event.photometry[event.photometry["band"] == band]
            if len(band_data) < 10:
                logger.info(f"Event {row['event_id']}: Failed (band {band} has fewer than 10 points)")
                return {"event_id": event_id, "status": "failed", "error": f"Band {band} has fewer than 10 points"}
            if band_data.empty:
                continue
            first_epoch = band_data["time"].min()
            last_epoch = band_data["time"].max()
            first_mag = round(band_data.loc[band_data["time"] == first_epoch, "ideal_mag"].iloc[0], 3)
            last_mag = round(band_data.loc[band_data["time"] == last_epoch, "ideal_mag"].iloc[0], 3)
            template_mag = round(event.parameters[band], 3)
            if first_mag != last_mag:
                if abs(first_mag - template_mag) / template_mag > f * 2:
                    logger.info(f"Event {row['event_id']}: Failed (first magnitude too different)")
                    return {"event_id": event_id, "status": "failed", "error": "First mag too different"}
                if abs(last_mag - template_mag) / template_mag > f * 2:
                    logger.info(f"Event {row['event_id']}: Failed (last magnitude too different)")
                    return {"event_id": event_id, "status": "failed", "error": "Last mag too different"}
                dist_first = abs(first_mag - template_mag)
                dist_last = abs(last_mag - template_mag)
                if dist_first > dist_last:
                    new_row = pd.DataFrame({
                        "event_id": [event_id],
                        "time": [first_epoch - 0.5],
                        "band": [band],
                        "ideal_mag": [last_mag],
                        "meas_mag": [np.nan],
                        "meas_mag_err": [np.nan],
                        "meas_flux": [np.nan],
                        "meas_flux_err": [np.nan],
                        "magnification": [np.nan],
                        "injection_flag": ["none"],
                        "measure_flag": ["none"],
                    })
                    event.photometry = pd.concat([new_row, event.photometry], ignore_index=True)
                elif dist_last > dist_first:
                    new_row = pd.DataFrame({
                        "event_id": [event_id],
                        "time": [last_epoch + 0.5],
                        "band": [band],
                        "ideal_mag": [first_mag],
                        "meas_mag": [np.nan],
                        "meas_mag_err": [np.nan],
                        "meas_flux": [np.nan],
                        "meas_flux_err": [np.nan],
                        "magnification": [np.nan],
                        "injection_flag": ["none"],
                        "measure_flag": ["none"],
                    })
                    event.photometry = pd.concat([event.photometry, new_row], ignore_index=True)
            band_data_sorted = event.photometry[event.photometry["band"] == band].sort_values("time")
            band_data_rounded = band_data_sorted.copy()
            band_data_rounded["ideal_mag_rounded"] = band_data_rounded["ideal_mag"].round(3)
            mask = (band_data_rounded["ideal_mag_rounded"] == first_mag).values
            false_index = np.where(~mask)[0]
            if len(false_index) > 0:
                start_id = false_index[0] - 1 if false_index[0] != 0 else 0
                end_id = false_index[-1] + 2 if false_index[-1] != len(mask) - 1 else None
                band_data_rounded = (
                    band_data_rounded.iloc[start_id:end_id]
                    if end_id is not None
                    else band_data_rounded.iloc[start_id:]
                )
            else:
                logger.info(f"Event {row['event_id']}: Failed (constant light curve)")
                return {"event_id": event_id, "status": "failed", "error": "Constant light curve"}
            other_bands = event.photometry[event.photometry["band"] != band]
            event.photometry = pd.concat(
                [other_bands, band_data_rounded.drop(columns=["ideal_mag_rounded"])],
                ignore_index=True,
            )

        logger.info(f"Event {row['event_id']}: Saving event")
        event.to_parquet(
            os.path.join(config["temp_dir"], f"temp_photometry_{event_id}.parquet"),
            os.path.join(config["temp_dir"], f"temp_data-events_{event_id}.parquet"),
        )
        del event
        gc.collect()
        logger.info(f"Event {row['event_id']}: Success")
        return {"event_id": event_id, "status": "success", "error": ""}
    except Exception as e:
        logger.error(f"Event {row['event_id']}: Failed with error: {str(e)}")
        return {"event_id": row["event_id"], "status": "failed", "error": traceback.format_exc()}


def process_ulens_event_load_nearby_object(row: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a microlensing event and load nearby object data.

    Simulates microlensing parameters and light curves, queries nearby objects from LSST catalog,
    and saves results to temporary Parquet files. Note: Parallel queries may cause job failures
    due to rate limits.

    Parameters
    ----------
    row : pd.Series
        Input row with columns 'event_id', 'ra', 'dec', 'bands', 'model', 'system_type'.
    config : Dict[str, Any]
        Configuration with keys:
        - temp_dir: Directory for temporary files.
        - peak_range: Tuple of (start, end) MJD for simulation.
        - epochs: Dictionary of observation times per band.
        - cadence_noise: Source of cadence and noise ('dp0', 'dp1', 'ideal').
        - blend_distance: Radius for nearby object search (degrees).
        - m_sat: Dictionary of saturation magnitudes per band.
        - m_5sigma: Dictionary of 5-sigma depth magnitudes per band.
        - pylima_blend: Blending parameter for pyLIMA.
        - photometry_schema: Schema for photometry Parquet files.
        - events_schema: Schema for events Parquet files.
        - sources_catalog: Source catalog ('TRILEGAL' or CSV path).
        - log_name: Logger name.

    Returns
    -------
    Dict[str, Any]
        Result dictionary with 'event_id', 'status', and optional 'error'.
    """
    logger = logging.getLogger(config.get("log_name", "pipeline"))
    try:
        logger.info(f"Event {row['event_id']}: Starting")
        event_id = row["event_id"]
        bands = config["bands"]
        epochs = config["epochs"]
        cadence_noise = config["cadence_noise"]

        event = Event(
            event_id=event_id,
            ra=row["ra"],
            dec=row["dec"],
            bands=bands,
            model=row["model"],
            system_type=row["system_type"],
            parallax=True,
            cadence_noise=cadence_noise,
            photometry_schema=config["photometry_schema"],
            events_schema=config["events_schema"],
            sources_catalog=config["sources_catalog"],
        )

        logger.info(f"Event {row['event_id']}: Simulating parameters")
        event.simulate_ulens_parameters(peak_range=config["peak_range"], blend=config["pylima_blend"])
        event.parameters["u0"] = np.random.uniform(0.01, 0.1)
        baseline_mags = event.source_data
        m_5sigma = config["m_5sigma"]
        all_faint = all(baseline_mags[band] >= m_5sigma[band] + 1 for band in bands)
        if all_faint:
            logger.info(f"Event {row['event_id']}: Failed (all baseline magnitudes too faint)")
            return {"event_id": event_id, "status": "failed", "error": "All baseline magnitudes too faint"}

        logger.info(f"Event {row['event_id']}: Simulating light curve")
        event.simulate_lc(epochs)
        if event.photometry.empty:
            logger.info(f"Event {row['event_id']}: Failed (empty photometry)")
            return {"event_id": event_id, "status": "failed", "error": "Empty photometry data"}

        if cadence_noise.startswith("dp"):
            logger.info(f"Event {row['event_id']}: Loading nearby object")
            lsst_data = LSSTData(
                ra=event.ra,
                dec=event.dec,
                radius=config["blend_distance"],
                data_preview=cadence_noise,
                bands=bands,
                name="closest_source",
            )
            columns = []
            for band in bands:
                if cadence_noise == "dp0":
                    columns.extend([
                        f"scisql_nanojanskyToAbMag({band}_cModelFlux) AS mag_{band}",
                        f"scisql_nanojanskyToAbMagSigma({band}_cModelFlux, {band}_cModelFluxErr) AS mag_err_{band}",
                        f"{band}_fwhm AS fwhm_{band}",
                    ])
                elif cadence_noise == "dp1":
                    columns.extend([
                        f"{band}_cModelMag AS mag_{band}",
                        f"{band}_cModelMagErr AS err_{band}",
                        f"{band}_fwhm AS fwhm_{band}",
                    ])

            center = SkyCoord(ra=lsst_data.ra, dec=lsst_data.dec, unit="deg", frame="icrs")
            region = CircleSkyRegion(center=center, radius=lsst_data.radius * u.deg)

            try:
                sources_data = lsst_data.load_catalog("Object", columns=columns, region=region, print_on=False)
                coords_array = sources_data[["coord_ra", "coord_dec"]].to_numpy()
                idx, min_dist_arcsec, nearest_coords = get_nearby_objects(
                    np.array([[event.ra, event.dec]]), coords_array
                )
                source_data = sources_data.iloc[idx[0]].to_dict()
                event.nearby_object = {
                    "objectId": source_data["ObjectId"],
                    "ra": nearest_coords[0, 0],
                    "dec": nearest_coords[0, 1],
                    "distance": min_dist_arcsec[0],
                    **{f"mag_{band}": source_data.get(f"mag_{band}", np.nan) for band in bands},
                    **{f"fwhm_{band}": source_data.get(f"fwhm_{band}", np.nan) for band in bands},
                }
            except Exception as e:
                if "No records found" in str(e):
                    event.nearby_object = {
                        "objectId": None,
                        "ra": None,
                        "dec": None,
                        "distance": -1,
                        **{f"mag_{band}": np.nan for band in bands},
                        **{f"fwhm_{band}": np.nan for band in bands},
                    }
                else:
                    logger.error(f"Event {row['event_id']}: Failed loading nearby object: {str(e)}")
                    return {"event_id": event_id, "status": "failed", "error": traceback.format_exc()}

        logger.info(f"Event {row['event_id']}: Saving event")
        event.to_parquet(
            os.path.join(config["temp_dir"], f"temp_photometry_{event_id}.parquet"),
            os.path.join(config["temp_dir"], f"temp_data-events_{event_id}.parquet"),
        )
        del event
        gc.collect()
        logger.info(f"Event {row['event_id']}: Success")
        return {"event_id": event_id, "status": "success", "error": ""}
    except Exception as e:
        logger.error(f"Event {row['event_id']}: Failed with error: {str(e)}")
        return {"event_id": row["event_id"], "status": "failed", "error": traceback.format_exc()}


def LSST_synthetic_photometry(row: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single calexp by injecting sources and measuring photometry.

    Saves results to temporary Parquet files for later combination.

    Parameters
    ----------
    row : pd.Series
        Series with columns ['visit', 'detector', 'band', 'expMidptMJD', 'magLim'].
    config : Dict[str, Any]
        Configuration with keys:
        - output_dir: Directory for output files.
        - temp_dir: Directory for temporary files.
        - photometry_file: Path to photometry Parquet file.
        - events_data: DataFrame with event data (event_id, ra, dec).
        - calexps_photometry_schema: Schema for calexps photometry Parquet files.
        - data_preview: Data preview type ('dp0', 'dp1').
        - log_name: Logger name.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'calexp_id', 'status', and optional 'error'.
    """
    logger = logging.getLogger(config.get("log_name", "pipeline"))
    calexp_id = row.name
    try:
        logger.info(f"Calexp {calexp_id}: Loading calexp")
        data_id = {"visit": row["visit"], "detector": row["detector"]}
        calexp = Calexp(data_id, data_preview=config["data_preview"])
    except Exception as e:
        logger.error(f"Calexp {calexp_id}: Failed loading calexp. {traceback.format_exc()}")
        return {
            "calexp_id": calexp_id,
            "status": "failed",
            "error": f"Failed to initialize calexp ({data_id=}): {str(e)}\n{traceback.format_exc()}",
        }

    logger.info(f"Calexp {calexp_id}: Loading events for injection")
    try:
        tol = 1e-5
        filters = [
            ("band", "=", row["band"]),
            ("time", ">=", row["expMidptMJD"] - tol),
            ("time", "<=", row["expMidptMJD"] + tol),
        ]
        table = pq.read_table(config["photometry_file"], filters=filters)
        photo = table.to_pandas()
        injection_data = pd.merge(config["events_data"], photo, on="event_id", how="inner")
    except Exception as e:
        logger.error(f"Calexp {calexp_id}: Failed obtaining injection data")
        return {
            "calexp_id": calexp_id,
            "status": "failed",
            "error": f"Failed to obtain injection_data: {str(e)}\n{traceback.format_exc()}",
        }

    logger.info(f"Calexp {calexp_id}: Initializing LSSTTools")
    try:
        tools = LSSTTools(calexp)
    except Exception as e:
        logger.error(f"Calexp {calexp_id}: Failed initializing LSSTTools")
        return {
            "calexp_id": calexp_id,
            "status": "failed",
            "error": f"Failed to initialize LSSTTools: {str(e)}\n{traceback.format_exc()}",
        }

    logger.info(f"Calexp {calexp_id}: Creating injection catalog")
    try:
        injection_catalog = tools.create_injection_catalog(
            ra_values=injection_data["ra"],
            dec_values=injection_data["dec"],
            mag_values=injection_data["ideal_mag"],
            magnification_values=injection_data["magnification"],
            expMidptMJD=row["expMidptMJD"],
            visit=row["visit"],
            detector=row["detector"],
            event_ids=injection_data["event_id"],
        )
        
        logger.info(f"Calexp {calexp_id}: Injecting {len(injection_catalog)} sources")
        injection_output = tools.inject_sources(injection_catalog)
        injected_exposure = injection_output["output_exposure"]
        injected_catalog = injection_output["output_catalog"]
        
    except Exception as e:
        err_str = str(e)
        if "No sources to be injected for this DatasetRef" in err_str:
            logger.info(f"Calexp {calexp_id}: No sources to inject")
            return {"calexp_id": calexp_id, "status": "failed", "error": "No sources to inject: skipping this calexp"}
        logger.error(f"Calexp {calexp_id}: Failed injecting sources. {traceback.format_exc()}")
        return {
            "calexp_id": calexp_id,
            "status": "failed",
            "error": f"Failed to process calexp: {err_str}\n{traceback.format_exc()}",
        }

    logger.info(f"Calexp {calexp_id}: Measuring photometry for {len(injected_catalog)} sources")
    try:
        extraction = tools.measure_photometry(injected_exposure, injected_catalog)
    except Exception as e:
        logger.error(f"Calexp {calexp_id}: Failed measuring photometry{traceback.format_exc()}")
        return {
            "calexp_id": calexp_id,
            "status": "failed",
            "error": f"Failed to measure photometry: {str(e)}\n{traceback.format_exc()}",
        }
    n_meas_sources = len(extraction[extraction["measure_flag"]==""])
    logger.info(f"Calexp {calexp_id}: Saving {n_meas_sources} photometry results")
    try:
        calexp_photometry_file = os.path.join(
            config["temp_dir"], f"temp_calexps-photometry_{calexp_id}.parquet"
        )
    
        # Agregar columnas extra a Astropy Table
        extraction["calexp_id"] = [calexp_id] * len(extraction)
        extraction["mag_lim"] = [row.get("magLim", None)] * len(extraction)
    
        # Ordenar columnas según el schema de PyArrow
        column_order = [name.name for name in config["calexps_photometry_schema"]]
        # Asegurarse de que todas las columnas existan
        for col in column_order:
            if col not in extraction.colnames:
                extraction[col] = [None] * len(extraction)
    
        # Convertir Astropy Table a PyArrow Table directamente
        table = pa.Table.from_pydict({col: extraction[col] for col in column_order},
                                     schema=config["calexps_photometry_schema"])
    
        if os.path.exists(calexp_photometry_file):
            os.remove(calexp_photometry_file)
        pq.write_table(table, calexp_photometry_file, compression="snappy")
    
    except Exception as e:
        logger.error(f"Calexp {calexp_id}: Failed saving photometry{traceback.format_exc()}")
        return {
            "calexp_id": calexp_id,
            "status": "failed",
            "error": f"Failed to save photometry results: {str(e)}\n{traceback.format_exc()}",
        }
    
    logger.info(f"Calexp {calexp_id}: End")
    return {"calexp_id": calexp_id, "status": "success", "error": ""}

def compute_chi2(row: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute chi-squared statistics for an event and save results to a temporary Parquet file.

    Parameters
    ----------
    row : pd.Series
        Series with columns 'event_id', 'bands'.
    config : Dict[str, Any]
        Configuration with keys:
        - temp_dir: Directory for temporary files.
        - calexps_photometry_file: Path to calexps photometry Parquet file.
        - events_file: Path to events Parquet file.
        - chi2_statistic: Statistic for chi-squared computation ('mean', 'median', 'constant').
        - log_name: Logger name.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'event_id', 'status', and optional 'error'.
        
    """
    logger = logging.getLogger(config.get("log_name", "pipeline"))
    event_id = row["event_id"]
    try:
        logger.info(f"Event {event_id}: Loading for chi2 computation")
        event = Event.from_parquet(event_id, config["calexps_photometry_file"], config["events_file"])
    except Exception as e:
        logger.error(f"Event {event_id}: Failed loading")
        return {
            "event_id": event_id,
            "status": "failed",
            "error": f"Failed to load event ({event_id=}): {str(e)}\n{traceback.format_exc()}",
        }

    logger.info(f"Event {event_id}: Computing chi2")
    try:
        data_chi2 = event.compute_chi2(cumulative=False, statistic=config["chi2_statistic"])
    except Exception as e:
        logger.error(f"Event {event_id}: Failed computing chi2")
        return {
            "event_id": event_id,
            "status": "failed",
            "error": f"Failed computing chi2 ({event_id=}): {str(e)}\n{traceback.format_exc()}",
        }

    logger.info(f"Event {event_id}: Saving chi2 results")
    try:
        os.makedirs(config["temp_dir"], exist_ok=True)
        chi2_file = os.path.join(config["temp_dir"], f"temp_chi2_{event_id}.parquet")
        chi_df = pd.DataFrame({
            "event_id": [event_id] * len(data_chi2),
            "band": list(data_chi2.keys()),
            "chi2": [data_chi2[b]["chi2"] for b in data_chi2],
            "p_value": [data_chi2[b]["p_value"] for b in data_chi2],
            "dof": [data_chi2[b]["dof"] for b in data_chi2],
        })
        table = pa.Table.from_pandas(chi_df, preserve_index=False)
        if os.path.exists(chi2_file):
            os.remove(chi2_file)
        pq.write_table(table, chi2_file, compression="snappy")
    except Exception as e:
        logger.error(f"Event {event_id}: Failed saving chi2 results")
        return {
            "event_id": event_id,
            "status": "failed",
            "error": f"Failed saving chi2 results ({event_id=}): {str(e)}\n{traceback.format_exc()}",
        }

    logger.info(f"Event {event_id}: Success")
    return {"event_id": event_id, "status": "success", "error": ""}