"""
Simulation pipeline module for the LSST pipeline project.

This module provides the `SimPipeline` class for orchestrating the simulation of microlensing
and non-microlensing events, including catalog generation, sky position sampling, light curve
simulation, and photometry processing. It integrates with `parallel_utils.ParallelProcessor`
for parallel processing, `catalogs_utils.Catalog` for Parquet file management, and other pipeline
modules to support experiments such as microlensing light curve recoverability and blending effect
studies. The pipeline is optimized for memory efficiency through chunked processing and selective
data loading, and for time efficiency via parallel execution and streamlined operations.

Classes
-------
SimPipeline : Manages the simulation pipeline for generating and processing light curve catalogs.

Constants
---------
EVENT_PROCESSORS : Dictionary mapping event processor names to their functions.
PHOTOMETRY_PROCESSORS : Dictionary mapping photometry processor names to their functions.
SKY_DISTRIBUTIONS : Dictionary mapping sky distribution types to their sampling functions.
"""

# Standard library imports
import logging
import os
import shutil
from typing import Dict, Any, Optional, List, Tuple, Callable

# Third-party imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
import astropy.units as u
from dl import queryClient as qc
from tqdm.auto import tqdm

# Local imports
from ulens_lsst.catalogs_utils import Catalog
from ulens_lsst.light_curves import Event
from ulens_lsst.lsst_data import LSSTData
from ulens_lsst.parallel_utils import ParallelProcessor
from ulens_lsst.processing_event_pipelines import process_cte_event, process_ulens_event, process_SNANA_ulens_event, compute_chi2, LSST_synthetic_photometry
from ulens_lsst.region_sky import SkyRegion
from ulens_lsst.utils import setup_logger, get_nearby_objects, sky_catalog_query


class SimPipeline:
    """
    Class to manage the simulation pipeline for LSST-like microlensing and non-microlensing events.

    Handles catalog generation, light curve simulation, and photometry processing using configurable
    event and photometry processors. Supports parallel processing and efficient Parquet file
    management for large datasets.

    Attributes
    ----------
    main_path : str
        Base directory for output files.
    output_dir : str
        Directory for simulation outputs.
    temp_dir : str
        Directory for temporary files.
    events_file : str
        Path to events Parquet file.
    photometry_file : str
        Path to photometry Parquet file.
    calexps_photometry_file : str
        Path to calexps photometry Parquet file.
    results_events_summary_file : str
        Path to events processing summary CSV.
    results_photometry_summary_file : str
        Path to photometry processing summary CSV.
    input_path : str
        Path to temporary input events Parquet file.
    logger : logging.Logger
        Logger for pipeline operations.
    photometry_catalog : catalogs_utils.Catalog
        Catalog object for photometry data.
    events_catalog : catalogs_utils.Catalog
        Catalog object for events data.
    event_ids : List[int]
        List of event IDs to process.
    events_ra : np.ndarray
        Array of RA coordinates for events.
    events_dec : np.ndarray
        Array of Dec coordinates for events.
    """

    EVENT_PROCESSORS = {
        "cte": process_cte_event,
        "ulens": process_ulens_event,
        "snana_ulens": process_SNANA_ulens_event,
    }
    PHOTOMETRY_PROCESSORS = {
        "synthetic": LSST_synthetic_photometry,
    }
    SKY_DISTRIBUTIONS = {
        "circle": lambda center_ra, center_dec, radius, n: SkyRegion(center_ra, center_dec, "circle", radius).sample(n, seed=42),
    }

    def __init__(self, config_path: str = None, from_folder: str = None, setup_dir: bool = True, new: bool = True) -> None:
        """
        Initialize the simulation pipeline with a YAML configuration file.

        Parameters
        ----------
        config_path : str, optional
            Path to the YAML configuration file.
        from_folder : str, optional
            Folder containing a previous config file to load.
        setup_dir : bool, optional
            Create output directories if True (default: True).
        new : bool, optional
            Create new output files if True; otherwise, use existing (default: True).

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        ValueError
            If required configuration parameters are missing or invalid.
        """
        self.main_path = "runs/"
        if from_folder:
            config_path = os.path.join(self.main_path, from_folder, "config_file.yaml")
        self._load_config(config_path)
        if setup_dir:
            self._setup_directories(mode="new" if new else "current")
        if not from_folder:
            if setup_dir:
                shutil.copy2(config_path, os.path.join(self.output_dir, "config_file.yaml"))
        

    def __repr__(self) -> str:
        """
        Return a string representation of the SimPipeline object.

        Returns
        -------
        str
            Formatted summary of pipeline attributes.
        """
        return self._formatted_summary()

    def __str__(self) -> str:
        """
        Return a string representation of the SimPipeline object for printing.

        Returns
        -------
        str
            Formatted summary of pipeline attributes.
        """
        return self._formatted_summary()

    def _formatted_summary(self) -> str:
        """
        Generate a formatted summary string of the main attributes.

        Returns
        -------
        str
            Summary of pipeline configuration and file paths.
        """
        summary = f"SimPipeline: {self.name}\n"
        summary += "=" * (len(self.name) + 13) + "\n\n"
        summary += "General configuration:\n"
        summary += f" - Number of events: {self.n_events}\n"
        summary += f" - Model: {self.model}\n"
        summary += f" - System type: {self.system_type}\n"
        summary += f" - Sources catalog: {self.sources_catalog}\n"
        summary += f" - Bands: {', '.join(self.bands)}\n"
        summary += f" - Simulation type: {self.sim_type}\n"
        if self.sim_type == "lsst_images":
            summary += f" - Data Preview: {self.data_preview}\n\n"
        elif self.sim_type == "rubin_sim":
            summary += f" - OpSim version: {self.opsim_version}\n"
            summary += f" - Use OpSim noise: {self.opsim_noise}\n"
        summary += "\nSky region:\n"
        summary += f" - Distribution: {self.sky_distribution}\n"
        if self.sky_center['frame'] == "galactic":
            summary += f" - Center: (l={self.sky_center['l']}, b={self.sky_center['b']}) deg (galactic)\n"
        else:
            summary += f" - Center: (ra={self.sky_center['coord'][0]}, dec={self.sky_center['coord'][1]}) deg (icrs)\n"
        summary += f" - Radius: {self.sky_radius} deg\n"
        summary += f" - Blend distance: {self.blend_distance} deg\n\n"
        summary += f" - Survey dates (MJD): {self.survey_dates[0]} to {self.survey_dates[1]}\n"
        summary += f" - Peak range (MJD): {self.peak_range[0]} to {self.peak_range[1]}\n\n"
        summary += "Processing:\n"
        summary += f" - Event processor: {self.event_processor_name}\n"
        summary += f" - Photometry processor: {self.photometry_processor_name}\n\n"
        summary += "Directories:\n"
        summary += f" - Output dir: {self.output_dir}\n"
        summary += f" - Data events: {self.events_file}\n"
        summary += f" - Photometry: {self.photometry_file}\n"
        summary += f" - Calexps photometry: {self.calexps_photometry_file}"
        return summary

    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from a YAML file and set attributes.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        ValueError
            If required configuration parameters are missing or invalid.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"YAML config file {config_path} does not exist.")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.name = config.get("name", "test")
        self.n_events = config.get("n_events", 10000)
        self.model = config.get("model", "USBL")
        self.system_type = config.get("system_type", "Planets_systems")
        self.sources_catalog = config.get("sources_catalog", "TRILEGAL")
        self.bands = config.get("bands", ["u", "g", "r", "i", "z", "y"])
        self.max_len_lc = config.get("max_len_lc", 4000)
        self.pylima_blend = config.get("pylima_blend", True)
        self.max_temp_files = config.get("max_temp_files", 10000)
        self.sim_type = config.get("sim_type", "ideal")
        self.data_preview = config.get("lsst_images", {}).get("data_preview", "dp0")
        self.max_load_calexps = config.get("lsst_images", {}).get("max_load_calexps", None)
        self.opsim_version = config.get("opsim", {}).get("opsim_version", "baseline")
        self.opsim_noise = config.get("opsim", {}).get("opsim_noise", False)
        survey_start = config.get("lsst", {}).get("survey_start", 60849)
        duration = config.get("lsst", {}).get("duration", 3)
        self.survey_dates = (survey_start, survey_start + duration * 365)
        self.peak_range = config.get("peak_range", (self.survey_dates[0] + 2 * 365, self.survey_dates[1] - 2 * 365))
        self.mag_sat = config.get("lsst", {}).get("mag_sat", {})
        self.mag_5sigma = config.get("lsst", {}).get("mag_5sigma", {}).get("single_exp", {})
        self.event_processor_name = config.get("event_processor", "snana_ulens")
        self.photometry_processor_name = config.get("photometry_processor", "synthetic")
        self.injection_config = config.get("injection_config", {})
        self.measurement_config = config.get("measurement_config", {})
        self.sky_distribution = config.get("sky_distribution", "circle")
        self.sky_center = config.get("sky_center", {"l": 0.5, "b": -1.25, "frame": "galactic"})
        self.sky_radius = config.get('sky_radius', 1.75)
        self.blend_distance = config.get("blend_distance", 0.001)
        self.log_config = config.get("logging", {})
        self.chi2_statistic = config.get("chi2_statistic", "median")

        required = ["name", "n_events", "model", "system_type", "bands"]
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

        if self.event_processor_name not in self.EVENT_PROCESSORS:
            raise ValueError(
                f"Invalid event_processor: {self.event_processor_name}. Choose from {list(self.EVENT_PROCESSORS.keys())}"
            )
        if self.photometry_processor_name not in self.PHOTOMETRY_PROCESSORS:
            raise ValueError(
                f"Invalid photometry_processor: {self.photometry_processor_name}. Choose from {list(self.PHOTOMETRY_PROCESSORS.keys())}"
            )
        if self.sky_distribution not in self.SKY_DISTRIBUTIONS:
            raise ValueError(
                f"Invalid sky_distribution: {self.sky_distribution}. Choose from {list(self.SKY_DISTRIBUTIONS.keys())}"
            )

        self.photometry_schema = config.get(
            "photometry_schema",
            {
                "event_id": pa.int64(),
                "time": pa.float64(),
                "band": pa.string(),
                "ideal_mag": pa.float64(),
                "meas_mag": pa.float64(),
                "meas_mag_err": pa.float64(),
                "meas_flux": pa.float64(),
                "meas_flux_err": pa.float64(),
                "magnification": pa.float64(),
                "injection_flag": pa.string(),
                "extraction_flag": pa.string(),
            },
        )
        self.events_schema = config.get(
            "events_schema",
            {
                "event_id": pa.int64(),
                "ra": pa.float64(),
                "dec": pa.float64(),
                "model": pa.string(),
                "system_type": pa.string(),
                "points": pa.int64(),
                "logL": pa.float64(),
                "logTe": pa.float64(),
                "D_L": pa.float64(),
                "D_S": pa.float64(),
                "mu_rel": pa.float64(),
                "nearby_object_ra": pa.float64(),
                "nearby_object_dec": pa.float64(),
                "nearby_object_objId": pa.string(),
                "nearby_object_distance": pa.float64(),
                "cadence_noise": pa.string(),
                "peak_time": pa.float64(),
                **{f"nearby_object_mag_{band}": pa.float64() for band in self.bands},
                **{f"nearby_object_fwhm_{band}": pa.float64() for band in self.bands},
                **{band: pa.float64() for band in self.bands},
                "param_t0": pa.float64(),
                "param_tE": pa.float64(),
                "param_u0": pa.float64(),
                "param_q": pa.float64(),
                "param_rho": pa.float64(),
                "param_s": pa.float64(),
                "param_alpha": pa.float64(),
                "param_piEE": pa.float64(),
                "param_piEN": pa.float64(),
            },
        )
        self.pylima_schema = config.get(
            "pylima_schema",
            {
                "param-pylima_t_center": pa.float32(),
                "param-pylima_u_center": pa.float32(),
                "param-pylima_tE": pa.float32(),
                "param-pylima_rho": pa.float32(),
                "param-pylima_separation": pa.float32(),
                "param-pylima_mass_ratio": pa.float32(),
                "param-pylima_alpha": pa.float32(),
                "param-pylima_piEN": pa.float32(),
                "param-pylima_piEE": pa.float32(),
                "param-pylima_fsource_u": pa.float32(),
                "param-pylima_ftotal_u": pa.float32(),
                "param-pylima_fsource_g": pa.float32(),
                "param-pylima_ftotal_g": pa.float32(),
                "param-pylima_fsource_r": pa.float32(),
                "param-pylima_ftotal_r": pa.float32(),
                "param-pylima_t0": pa.float32(),
                "param-pylima_u0": pa.float32(),
                "param-pylima_fblend_u": pa.float32(),
                "param-pylima_gblend_u": pa.float32(),
                "param-pylima_fblend_g": pa.float32(),
                "param-pylima_gblend_g": pa.float32(),
                "param-pylima_fblend_r": pa.float32(),
                "param-pylima_gblend_r": pa.float32(),
            },
        )

    def _setup_directories(self, mode: str = "new") -> None:
        """
        Create output and temporary directories with indexed filenames.

        Parameters
        ----------
        mode : str, optional
            File handling mode: 'new' for new files, 'current' for existing (default: 'new').

        Raises
        ------
        ValueError
            If mode is invalid.
        """
        if mode not in ["new", "current"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'new' or 'current'.")

        self.output_dir = os.path.join(self.main_path, self.name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "analysis"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "event_plots"), exist_ok=True)
        self.temp_dir = os.path.join(self.output_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        existing_event_files = [
            f for f in os.listdir(self.output_dir) if f.startswith("data-events_") and f.endswith(".parquet")
        ]
        max_index = max(
            (int(f.split("_")[-1].split(".")[0]) for f in existing_event_files), default=1
        ) if existing_event_files else 0
        self.new_file_index = max_index + 1 if mode == "new" else max_index

        log_level_str = self.log_config.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        log_path = os.path.join(self.output_dir, f"{self.log_config["log_name"]}_{self.new_file_index}.log")
        self.logger = setup_logger(
            log_file=log_path,
            log_to_console=self.log_config.get("log_to_console", True),
            logger_name=self.log_config["log_name"],
        )

        self.events_file = os.path.join(self.output_dir, f"data-events_{self.new_file_index}.parquet")
        self.photometry_file = os.path.join(self.output_dir, f"photometry_{self.new_file_index}.parquet")
        self.calexps_photometry_file = os.path.join(self.output_dir, f"calexps-photometry_{self.new_file_index}.parquet")
        # self.results_events_summary_file = os.path.join(self.output_dir, f"processed_events_{self.new_file_index}.csv")
        # self.results_photometry_summary_file = os.path.join(self.output_dir, f"processed_photometry_{self.new_file_index}.csv")
        self.input_path = os.path.join(self.output_dir, "temp_input_events.parquet")

        self.logger.info(f"Setup directories: output={self.output_dir}, temp={self.temp_dir}")
        self.logger.info(f"Events file: {self.events_file} (mode={mode})")
        self.logger.info(f"Photometry file: {self.photometry_file} (mode={mode})")

    def _initialize_catalogs(self) -> None:
        """
        Initialize photometry and events catalogs with their schemas.

        Raises
        ------
        ValueError
            If schema is invalid or file creation fails.
        """
        try:
            self.photometry_catalog = Catalog(self.photometry_file, schema=self.photometry_schema)
        except Exception as e:
            self.logger.error(f"Failed to initialize photometry catalog at {self.photometry_file}: {str(e)}")
            raise ValueError(f"Failed to initialize photometry catalog: {str(e)}")

        try:
            if self.model != "Pacz":
                self.events_schema.update(self.pylima_schema)
            self.events_catalog = Catalog(self.events_file, schema=self.events_schema)
        except Exception as e:
            self.logger.error(f"Failed to initialize events catalog at {self.events_file}: {str(e)}")
            raise ValueError(f"Failed to initialize events catalog: {str(e)}")

    def _sample_sky_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample sky positions based on the configured distribution.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Arrays of RA and Dec coordinates.
        """
        if self.sky_center["frame"] == "galactic":
            center = SkyCoord(l=self.sky_center["l"] * u.degree, b=self.sky_center["b"] * u.degree, frame="galactic").icrs
        else:
            center = SkyCoord(ra=self.sky_center["coord"][0] * u.degree, dec=self.sky_center["coord"][1] * u.degree, frame="icrs")
        sampler = self.SKY_DISTRIBUTIONS[self.sky_distribution]
        return sampler(center.ra.deg, center.dec.deg, self.sky_radius * u.deg, self.n_events)

    def _setup_event_ids(self) -> None:
        """
        Set up event IDs based on the maximum event_id in existing data-events files.

        Raises
        ------
        ValueError
            If Parquet file reading fails.
        """
        existing_event_files = [
            f for f in os.listdir(self.output_dir) if f.startswith("data-events_") and f.endswith(".parquet")
        ]
        max_existing_event_id = 0
        if existing_event_files:
            latest_file = max(existing_event_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            self.logger.info(f"Found latest events file: {latest_file}")
            try:
                catalog = Catalog(os.path.join(self.output_dir, latest_file), main_path=self.main_path)
                max_id = catalog.get_max_value("event_id")
                max_existing_event_id = int(max_id) if max_id is not None and not pd.isna(max_id) else 0
                self.logger.info(f"Starting from event_id = {max_existing_event_id + 1}")
            except Exception as e:
                self.logger.warning(f"Error reading event_id from {latest_file}: {str(e)}. Starting from event_id=1.")
        else:
            self.logger.info(f"No existing data-events files in {self.output_dir}. Starting with event_id=1.")

        self.event_ids = list(range(max_existing_event_id + 1, max_existing_event_id + self.n_events + 1))
        self.logger.info(f"Generated {len(self.event_ids)} event IDs: {self.event_ids[:5]}{'...' if len(self.event_ids) > 5 else ''}")

    def simulate_lightcurves(self, event_processor: Optional[str] = None) -> pd.DataFrame:
        """
        Generate photometry and events catalogs for microlensing simulations.

        Parameters
        ----------
        event_processor : str, optional
            Name of the event processing function; uses config if None.

        Returns
        -------
        pd.DataFrame
            Processing results summary.

        Raises
        ------
        ValueError
            If event processor is invalid or processing fails.
        """
        self.logger.info("Starting simulate_lightcurves")
        self._initialize_catalogs()
        self._setup_event_ids()
        processor_name = event_processor or self.event_processor_name
        if processor_name not in self.EVENT_PROCESSORS:
            raise ValueError(
                f"Invalid event_processor: {processor_name}. Choose from {list(self.EVENT_PROCESSORS.keys())}"
            )

        self.events_ra, self.events_dec = self._sample_sky_positions()
        self.load_event_sources_catalog()

        if self.sim_type == "rubin_sim":
            from utils import baseline_name
            self.cadence_noise = baseline_name()
            from utils import get_lsst_mjds_per_band
            self.epochs = get_lsst_mjds_per_band(
                ra=self.sky_center.get("coord", [0.5])[0] if self.sky_center["frame"] == "icrs" else SkyCoord(
                    l=self.sky_center["l"] * u.degree, b=self.sky_center["b"] * u.degree, frame="galactic"
                ).icrs.ra.deg,
                dec=self.sky_center.get("coord", [-1.25])[1] if self.sky_center["frame"] == "icrs" else SkyCoord(
                    l=self.sky_center["l"] * u.degree, b=self.sky_center["b"] * u.degree, frame="galactic"
                ).icrs.dec.deg,
                bands=self.bands,
                mjd_range=self.survey_dates,
                opsim=self.opsim_version,
            )
        elif self.sim_type == "ideal":
            start, end = self.survey_dates
            interval = int((end - start) * 2)
            self.epochs = {band: np.linspace(int(start), int(end), interval) for band in self.bands}
            self.cadence_noise = "ideal"
        elif self.sim_type == "lsst_images":
            lsst_data = LSSTData(
                ra=self.sky_center.get("coord", [0.5])[0] if self.sky_center["frame"] == "icrs" else SkyCoord(
                    l=self.sky_center["l"] * u.degree, b=self.sky_center["b"] * u.degree, frame="galactic"
                ).icrs.ra.deg,
                dec=self.sky_center.get("coord", [-1.25])[1] if self.sky_center["frame"] == "icrs" else SkyCoord(
                    l=self.sky_center["l"] * u.degree, b=self.sky_center["b"] * u.degree, frame="galactic"
                ).icrs.dec.deg,
                radius=self.sky_radius,
                data_preview=self.data_preview,
                name=self.name,
            )
            try:
                lsst_data.calexp_catalog(bands=self.bands, n_max=self.max_load_calexps)
                self.logger.info(f"Loaded {len(lsst_data.calexps_data)} calexps for bands {self.bands}")
                lsst_data.calexps_data.to_csv(os.path.join(self.output_dir, "data_calexps.csv"), index=False)
            except Exception as e:
                self.logger.error(f"Failed to load calexps: {str(e)}")
                raise ValueError(f"Failed to load calexps: {str(e)}")
            self.cadence_noise = self.data_preview
            self.epochs = lsst_data.epochs
            self.survey_dates = lsst_data.survey_dates
            self.peak_range = self.survey_dates

        input_df = pd.DataFrame({
            "event_id": self.event_ids,
            "ra": self.events_ra,
            "dec": self.events_dec,
            "bands": [self.bands] * self.n_events,
            "model": [self.model] * self.n_events,
            "system_type": [self.system_type] * self.n_events,
        })
        pq.write_table(pa.Table.from_pandas(input_df), self.input_path, compression="snappy")

        num_pools = min(os.cpu_count() or 4, 8)
        chunk_size = max(self.n_events // num_pools, 1)
        processor = ParallelProcessor(
            input_path=self.input_path,
            output_dir=self.output_dir,
            process_fn=self.EVENT_PROCESSORS[processor_name],
            config={
                "temp_dir": self.temp_dir,
                "survey_dates": self.survey_dates,
                "peak_range": self.peak_range,
                "bands": self.bands,
                "epochs": self.epochs,
                "cadence_noise": self.cadence_noise,
                "blend_distance": self.blend_distance,
                "m_sat": self.mag_sat,
                "m_5sigma": self.mag_5sigma,
                "pylima_blend": self.pylima_blend,
                "main_path": self.main_path,
                "photometry_schema": self.photometry_schema,
                "events_schema": self.events_schema,
                "log_name": self.log_config["log_name"],
                "sources_catalog": self.sources_catalog,
            },
            chunk_size=chunk_size,
            checkpoint_interval=chunk_size,
            debug=True,
            max_temp_files=self.max_temp_files,
            photometry_catalog=self.photometry_catalog,
            events_catalog=self.events_catalog,
        )
        processor.process(num_pools=num_pools)

        self.logger.info("Combining temporary Parquet files")
        self.photometry_catalog.combine_parquet_files(
            temp_dir=self.temp_dir,
            schema=self.photometry_catalog.get_schema(),
            batch_size=500,
            cleanup=True,
        )
        self.events_catalog.combine_parquet_files(
            temp_dir=self.temp_dir,
            schema=self.events_catalog.get_schema(),
            batch_size=500,
            cleanup=True,
        )

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        if os.path.exists(self.input_path):
            os.remove(self.input_path)

        results_df = processor.get_results()
        # results_df.to_csv(self.results_events_summary_file, index=False)
        self.logger.info("\nProcessing Summary:")
        self.logger.info(f"Total events processed: {len(results_df)}")
        self.logger.info(f"Successful: {len(results_df[results_df['status'] == 'success'])}")
        self.logger.info(f"Failed: {len(results_df[results_df['status'] == 'failed'])}")
        if not results_df[results_df["status"] == "failed"].empty:
            self.logger.info("\nErrors encountered:")
            self.logger.info(results_df[results_df["status"] == "failed"][["event_id", "error"]].head())

        self.logger.info("Ending simulate_lightcurves")
        return results_df

    def load_nearby_objects(self):
        """
        Load nearby LSST catalog objects for each event and merge them with the events DataFrame.
    
        This function:
          1. Loads the existing events table from the Parquet file.
          2. Queries the LSST catalog for objects near the simulation sky center.
          3. Finds the closest object to each event based on RA/Dec coordinates.
          4. Merges the matched objects into the events table.
          5. Saves the updated table back to the Parquet file.
    
        Notes
        -----
        - Only executed if `self.sim_type == "lsst_images"`.
        - Handles both `dp0` and `dp1` LSST data previews.
        - Existing overlapping columns in `events_df` are dropped before merging.
    
        """
        self.logger.info("Starting load_nearby_objects")
    
        # Skip if simulation type does not require LSST objects
        if self.sim_type != "lsst_images":
            self.logger.info(f"Simulation type: {self.sim_type}. Nothing to do here.")
            return
    
        # --- Load events (only RA, Dec needed here) ---
        events_df = pd.read_parquet(self.events_file, engine="pyarrow")
        coords_events = events_df[["ra", "dec"]].values
    
        # --- Initialize LSST catalog access ---
        lsst_data = LSSTData(
            ra=self.sky_center["coord"][0],
            dec=self.sky_center["coord"][1],
            radius=self.sky_radius,
            data_preview=self.data_preview,
            bands=self.bands,
            name=self.name
        )
    
        # --- Build list of catalog columns depending on DP version ---
        columns = []
        if lsst_data.data_preview=='dp0':
            for band in self.bands:
                columns.append(f"scisql_nanojanskyToAbMag({band}_cModelFlux) AS mag_{band} ")
                columns.append(f"scisql_nanojanskyToAbMagSigma({band}_cModelFlux, {band}_cModelFluxErr) AS mag_err_{band}, "
                            f"{band}_fwhm AS fwhm_{band} ")
        elif lsst_data.data_preview=="dp1":
            for band in self.bands:
                columns.append(f"{band}_cModelMag AS mag_{band}")
                columns.append(f"{band}_cModelMagErr AS mag_err_{band}")
    
        # --- Query catalog objects near the sky center ---
        objects_data = lsst_data.load_catalog("Object", columns=columns)
        coords_objects = objects_data[["coord_ra", "coord_dec"]].values
    
        # --- Find nearest catalog object for each event ---
        idx, min_dist_arcsec, nearest_coords_deg = get_nearby_objects(coords_events, coords_objects)
        nearby_objects = objects_data.iloc[idx].reset_index(drop=True)
    
        # --- Rename columns to avoid collisions ---
        rename_map = {
            col: (
                f"nearby_object_{col.split('_')[-1]}" if col.startswith("coord")
                else "nearby_object_objId" if col.startswith("Object")
                else f"nearby_object_{col}"
            )
            for col in nearby_objects.columns
        }
        nearby_objects = nearby_objects.rename(columns=rename_map)
        nearby_objects["nearby_object_distance"] = min_dist_arcsec
    
        # --- Merge objects into events table ---
        cols_to_add = nearby_objects.columns
        events_df = events_df.drop(columns=[col for col in cols_to_add if col in events_df.columns], errors="ignore")
        events_with_objects = pd.concat([events_df.reset_index(drop=True), nearby_objects], axis=1)
    
        # --- Save updated table ---
        events_with_objects.to_parquet(self.events_file, index=False)
    
        self.logger.info("Ending load_nearby_objects")


    def load_event_sources_catalog(self) -> None:
        """
        Load or generate a sources catalog for events.

        If sources_catalog is 'TRILEGAL', uses existing TRILEGAL data. Otherwise, queries
        AstroDataLab to create a custom sources catalog.

        Raises
        ------
        ValueError
            If the catalog source is invalid or query fails.
        """
        if self.sources_catalog != "TRILEGAL":
            columns = ["ra", "dec", "logl", "logte", "umag", "gmag", "rmag", "imag", "zmag", "ymag", "gc", "label"]
            catalog = "lsst_sim.simdr2" + ("_binary" if self.sources_catalog.endswith("binary") else "")
            regions = [(ra, dec, 0.02) for ra, dec in zip(self.events_ra, self.events_dec)]
            filters = [f"({band}mag > {self.mag_sat[band]} AND {band}mag < {self.mag_5sigma[band] + 1})" for band in self.bands]
            final_filter = " OR ".join(filters)

            self.logger.info(f"Loading AstroDataLab sources from {catalog}")
            results = []
            for i, region in enumerate(tqdm(regions, desc = f"Loading AstroDataLab sources from {catalog}")):
                query = sky_catalog_query(catalog, columns, [region], [final_filter], language="sql")
                # print(query)
                try:
                    res = qc.query(sql=query, fmt="table")
                    df = res.to_pandas()
                    if df.empty:
                        larger_region = [(region[0], region[1], 0.1)]
                        query = sky_catalog_query(catalog, columns, larger_region, [final_filter], language="sql")
                        res = qc.query(sql=query, fmt="table")
                        df = res.to_pandas()
                    if not df.empty:
                        df = df.sample(n=1, random_state=i)
                        results.append(df)
                except Exception as e:
                    self.logger.warning(f"Query failed for region {region}: {str(e)}")

            if results:
                df = pd.concat(results, ignore_index=True)
                df = df.rename(columns={col: col[0] if col.endswith("mag") else col for col in df.columns})
                df = df.rename(columns={"logl": "logL", "logte": "logTe"})
                self.sources_catalog = os.path.join(self.output_dir, f"{self.sources_catalog}_event_sources_catalog.csv")
                df.to_csv(self.sources_catalog, index=False)
                self.logger.info(f"Saved sources catalog to {self.sources_catalog}")
            else:
                self.logger.warning("No sources retrieved from AstroDataLab. Using empty catalog.")
                self.sources_catalog = os.path.join(self.output_dir, "empty_event_sources_catalog.csv")
                pd.DataFrame(columns=["ra", "dec", "logL", "logTe"] + self.bands).to_csv(self.sources_catalog, index=False)

    def compute_events_chi2(self) -> pd.DataFrame:
        """
        Compute chi-squared statistics for events and update the events catalog.

        Returns
        -------
        pd.DataFrame
            Processing results summary.

        Raises
        ------
        ValueError
            If processing fails or files are missing.
        """
        self.logger.info("Starting compute_chi2")
        event_ids = pq.read_table(self.events_file, columns=["event_id"])["event_id"].to_numpy()
        table = pa.table({"event_id": event_ids, "bands": [self.bands] * len(event_ids)})
        pq.write_table(table, self.input_path, compression="snappy")

        num_pools = min(os.cpu_count() or 4, 8)
        chunk_size = max(len(event_ids) // num_pools, 1)
        processor = ParallelProcessor(
            input_path=self.input_path,
            output_dir=self.output_dir,
            process_fn=compute_chi2,
            config={
                "temp_dir": self.temp_dir,
                "output_dir": self.output_dir,
                "chi2_statistic": self.chi2_statistic,
                "calexps_photometry_file": self.calexps_photometry_file,
                "events_file": self.events_file,
                "log_name": self.log_config["log_name"],
            },
            chunk_size=chunk_size,
            checkpoint_interval=chunk_size,
            debug=True,
            max_temp_files=self.max_temp_files,
        )
        processor.process(num_pools=num_pools)

        self.logger.info("Combining temporary chi2 Parquet files")
        chi2_files = [os.path.join(self.temp_dir, f) for f in os.listdir(self.temp_dir) if f.startswith("temp_chi2_")]
        if not chi2_files:
            self.logger.warning("No temporary chi2 files found!")
            return pd.DataFrame()

        chi2_tables = [pq.read_table(f) for f in chi2_files]
        chi2_all = pa.concat_tables(chi2_tables).to_pandas()

        for f in chi2_files:
            os.remove(f)

        chi2_pivot = chi2_all.pivot(index="event_id", columns="band", values=["chi2", "p_value", "dof"])
        chi2_pivot.columns = [f"{stat}_{band}" for stat, band in chi2_pivot.columns]
        chi2_pivot.reset_index(inplace=True)

        events_df = pq.read_table(self.events_file).to_pandas()
        events_df = events_df.merge(chi2_pivot, on="event_id", how="left")
        pq.write_table(pa.Table.from_pandas(events_df, preserve_index=False), self.events_file)

        results_df = processor.get_results()
        # results_df.to_csv(self.results_events_summary_file, index=False)
        self.logger.info("\nProcessing Summary:")
        self.logger.info(f"Total events processed: {len(results_df)}")
        self.logger.info(f"Successful: {len(results_df[results_df['status'] == 'success'])}")
        self.logger.info(f"Failed: {len(results_df[results_df['status'] == 'failed'])}")
        if not results_df[results_df["status"] == "failed"].empty:
            self.logger.info("\nErrors encountered:")
            self.logger.info(results_df[results_df["status"] == "failed"][["event_id", "error"]].head())

        self.logger.info("Ending compute_chi2")
        return results_df

    def process_synthetic_photometry(
        self, bands: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Process calexps by injecting sources and measuring photometry.

        Parameters
        ----------
        events_file_index : int, optional
            Index of the data-events_{i}.parquet file (default: 1).
        photometry_file_index : int, optional
            Index of the photometry_{i}.parquet file (default: 1).
        bands : List[str], optional
            Bands to process; uses self.bands if None.

        Returns
        -------
        pd.DataFrame
            Processing results summary.

        Raises
        ------
        FileNotFoundError
            If required files do not exist.
        ValueError
            If input data or processing fails.
        """
        self.logger.info("Starting process_synthetic_photometry")
        self.calexps_photometry_file = os.path.join(self.output_dir, f"calexps-photometry_{self.new_file_index}.parquet")

        if not os.path.exists(self.events_file):
            raise FileNotFoundError(f"Events file {self.events_file} does not exist.")
        if not os.path.exists(os.path.join(self.output_dir, "data_calexps.csv")):
            raise FileNotFoundError("data_calexps.csv not found. Run LSSTData.calexp_catalog first.")

        os.makedirs(self.temp_dir, exist_ok=True)
        self.calexps_photometry_schema = pa.schema([
            ("event_id", pa.int32()),
            ("calexp_id", pa.int32()),
            ("time", pa.float64()),
            ("band", pa.string()),
            ("ideal_mag", pa.float64()),
            ("meas_mag", pa.float64()),
            ("meas_mag_err", pa.float64()),
            ("meas_flux", pa.float64()),
            ("meas_flux_err", pa.float64()),
            ("magnification", pa.float64()),
            ("injection_flag", pa.int32()),
            ("measure_flag", pa.string()),
            ("mag_lim", pa.float64()),
        ])
        self.calexps_photometry_catalog = Catalog(
            self.calexps_photometry_file,  schema=self.calexps_photometry_schema
        )

        self.calexps = pd.read_csv(os.path.join(self.output_dir, "data_calexps.csv"))
        events_data = pd.read_parquet(self.events_file, columns=["event_id", "ra", "dec"])

        num_pools = min(os.cpu_count() or 4, 8)
        chunk_size = int(np.ceil(len(self.calexps) / num_pools))
        processor = ParallelProcessor(
            input_path=os.path.join(self.output_dir, "data_calexps.csv"),
            output_dir=self.output_dir,
            process_fn=self.PHOTOMETRY_PROCESSORS[self.photometry_processor_name],
            config={
                "output_dir": self.output_dir,
                "temp_dir": self.temp_dir,
                "survey_dates": self.survey_dates,
                "peak_range": self.peak_range,
                "main_path": self.main_path,
                "photometry_file": self.photometry_file,
                "events_data": events_data,
                "calexps_photometry_schema": self.calexps_photometry_schema,
                "data_preview": self.data_preview,
                "log_name": self.log_config["log_name"],
            },
            chunk_size=chunk_size,
            checkpoint_interval=chunk_size,
            debug=True,
            max_temp_files=self.max_temp_files,
            photometry_catalog=self.calexps_photometry_catalog,
        )
        processor.process(num_pools=num_pools)

        self.logger.info("Combining temporary Parquet files")
        results = processor.get_results()
        # results.to_csv(self.results_photometry_summary_file, index=False)
        self.calexps_photometry_catalog.combine_parquet_files(
            temp_dir=self.temp_dir,
            schema=self.calexps_photometry_catalog.get_schema(),
            batch_size=500,
            cleanup=True,
        )
        
        self.logger.info("Ending process_synthetic_photometry")
        return results

    def transform_format_catalogs(self, event_processor: Optional[str] = None) -> None:
        """
        Apply format transformation to catalogs if specified in config.

        Parameters
        ----------
        event_processor : str, optional
            Event processor to use for transformation; uses config if None.
        """
        output_format = self.config.get("format_output", "none")
        if output_format != "none":
            self.logger.info(f"Transforming catalogs to format: {output_format}")
            processor_name = event_processor or self.event_processor_name
            self.events_catalog.transform_format(output_format, self.config.get("format_output_params", {}))
            self.logger.info(f"Catalogs transformed to {output_format}")