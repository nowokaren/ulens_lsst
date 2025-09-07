"""
Light curve management module for the LSST pipeline project.

This module provides the `Event` class for managing microlensing events, including
their coordinates, light curves, and parameters. It supports generating and simulating
light curves using models like Paczynski or pyLIMA (USBL, FSPL, PSPL), storing data
in Parquet files, and visualizing results. The module is optimized for memory efficiency
by using selective data loading and pyarrow for Parquet operations, and for time efficiency
through streamlined parameter handling and parallel processing capabilities. It integrates
with other pipeline modules (e.g., `ulens_utils`, `catalogs_utils`) for comprehensive
microlensing experiments, such as studying light curve recoverability and blending effects.

Classes
-------
Event : Manages a microlensing event, including coordinates, light curves, and parameters.

Constants
---------
DEFAULT_COLORS : Dictionary mapping LSST bands to colors for plotting.
TRILEGAL_DIR : Directory for TRILEGAL and Genulens data files.
ULENS_COLUMNS : Columns for ulens data (e.g., D_L, D_S, mu_rel).
SOURCE_COLUMNS_BASE : Base columns for source data (logL, logTe).
ROWS_PER_CHUNK : Number of rows per chunk in TRILEGAL/Genulens files.
"""

# Standard library imports
import gc
import io
import os
from typing import Dict, Any, Optional, List, Union, Tuple
from copy import copy

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import chi2
import contextlib

# Local imports
from catalogs_utils import Catalog
from ulens_utils import pacz_parameters, pylima_parameters, simulate_pacz_lightcurve, simulate_pylima_lightcurve


class Event:
    """
    Class to manage a microlensing event, including coordinates, light curves, and parameters.

    Supports simulation of light curves using Paczynski or pyLIMA models, storage in Parquet
    files, and visualization with customizable plotting options. Optimized for memory by loading
    only necessary data and using pyarrow for efficient Parquet operations.

    Attributes
    ----------
    event_id : int
        Unique identifier for the event.
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    bands : List[str]
        List of filter names (e.g., ['u', 'g', 'r']).
    model : str
        Microlensing model ('Pacz', 'USBL', 'FSPL', 'PSPL').
    system_type : str
        Type of system (e.g., 'Planets_systems', 'Binary_stars').
    parameters : Dict[str, Any]
        Model parameters (e.g., {'t0': 100, 'u0': 0.1}).
    parallax : bool
        Whether to include parallax effects.
    cadence_noise : str
        Source of epochs and errors (e.g., 'dp0', 'opsim_v4.3.1', 'ideal').
    source_data : Dict[str, Any]
        Source properties (e.g., {'logL': 1, 'logTe': 1, 'u': 20.0}).
    ulens_data : Dict[str, Any]
        Microlensing parameters (e.g., {'D_L': 0.5, 'D_S': 1, 'mu_rel': 0.5}).
    photometry : pd.DataFrame
        Photometry data with specified schema.
    nearby_object : Dict[str, Any]
        Data for the nearest object (e.g., objectId, ra, dec, magnitudes).
    """

    DEFAULT_COLORS = {
        "u": "purple",
        "g": "green",
        "r": "red",
        "i": "orange",
        "z": "brown",
        "y": "blue",
        "Y": "blue",
    }
    TRILEGAL_DIR = "../roman_rubin/chunks_TRILEGAL_GENULENS/"
    ULENS_COLUMNS = ["D_L", "D_S", "mu_rel", "tE", "thetaE", "piE", "piEN", "piEE"]
    SOURCE_COLUMNS_BASE = ["logL", "logTe"]
    ROWS_PER_CHUNK = 10000

    def __init__(
        self,
        event_id: int,
        ra: float,
        dec: float,
        bands: List[str],
        model: str,
        system_type: str,
        cadence_noise: str = "ideal",
        parameters: Optional[Dict[str, Any]] = None,
        parallax: bool = False,
        source_data: Optional[Dict[str, Any]] = None,
        ulens_data: Optional[Dict[str, Any]] = None,
        photometry: Optional[pd.DataFrame] = None,
        photometry_schema: Optional[Dict[str, str]] = None,
        events_schema: Optional[Dict[str, pa.DataType]] = None,
        sources_catalog: str = "TRILEGAL",
    ):
        """
        Initialize an Event object.

        Parameters
        ----------
        event_id : int
            Unique identifier for the event.
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.
        bands : List[str]
            List of filter names (e.g., ['u', 'g', 'r']).
        model : str
            Microlensing model ('Pacz', 'USBL', 'FSPL', 'PSPL').
        system_type : str
            Type of system (e.g., 'Planets_systems', 'Binary_stars').
        cadence_noise : str, optional
            Source of epochs and errors ('dp0', 'opsim_v4.3.1', 'ideal') (default: 'ideal').
        parameters : Dict[str, Any], optional
            Model parameters (e.g., {'t0': 100, 'u0': 0.1}).
        parallax : bool, optional
            Include parallax effects (default: False).
        source_data : Dict[str, Any], optional
            Source properties (e.g., {'logL': 1, 'logTe': 1, 'u': 20.0}).
        ulens_data : Dict[str, Any], optional
            Microlensing parameters (e.g., {'D_L': 0.5, 'D_S': 1, 'mu_rel': 0.5}).
        photometry : pd.DataFrame, optional
            Photometry data with specified columns.
        photometry_schema : Dict[str, str], optional
            Schema for photometry DataFrame with column dtypes.
        events_schema : Dict[str, pyarrow.DataType], optional
            Schema for events Parquet file.
        sources_catalog : str, optional
            Catalog source for source data ('TRILEGAL' or path to CSV) (default: 'TRILEGAL').
        """
        self.event_id = event_id
        self.ra = ra
        self.dec = dec
        self.bands = [band.lower() for band in bands]
        self.model = model
        self.system_type = system_type
        self.parameters = parameters or {}
        self.parallax = parallax
        self.cadence_noise = cadence_noise
        self.events_schema = events_schema

        # Initialize photometry with explicit dtypes
        if photometry_schema is None:
            photometry_schema = {
                "event_id": "int32",
                "band": "object",
                "time": "float32",
                "ideal_mag": "float32",
                "meas_mag": "float32",
                "meas_mag_err": "float32",
                "meas_flux": "float32",
                "meas_flux_err": "float32",
                "magnification": "float32",
                "injection_flag": "object",
                "measure_flag": "object",
            }
        self.photometry = (
            photometry.astype(photometry_schema)
            if photometry is not None
            else pd.DataFrame(columns=photometry_schema.keys()).astype(photometry_schema)
        )

        # Initialize source_data and ulens_data
        self.source_data = source_data or self._load_source_data(catalog=sources_catalog)
        self.ulens_data = ulens_data or self._load_ulens_data()

        # Initialize nearby_object
        self.nearby_object = {
            "objectId": None,
            "ra": None,
            "dec": None,
            "distance": None,
            **{f"mag_{band}": np.nan for band in self.bands},
            **{f"fwhm_{band}": np.nan for band in self.bands},
        }

    def _load_source_data(self, catalog: str = "TRILEGAL") -> Dict[str, Any]:
        """
        Load source data from TRILEGAL or a custom catalog for the given event_id.

        Parameters
        ----------
        catalog : str, optional
            Catalog source ('TRILEGAL' or path to CSV file) (default: 'TRILEGAL').

        Returns
        -------
        Dict[str, Any]
            Source data including logL, logTe, and magnitudes.
        """

        bands = self.bands
        if catalog == "TRILEGAL":
            chunk_id = (self.event_id // self.ROWS_PER_CHUNK) + 1
            row_index = self.event_id % self.ROWS_PER_CHUNK
            if chunk_id < 1 or chunk_id > 20:
                print(f"Invalid chunk_id: {chunk_id}")
                return {"logL": np.nan, "logTe": np.nan, **{band: np.nan for band in self.bands}}
            file_path = os.path.join(self.TRILEGAL_DIR, f"TRILEGAL_chunk_{chunk_id}.csv")
            if "y" in self.bands:
                i = bands.index("y")
                bands[i] = "Y"
        elif catalog.endswith("event_sources_catalog.csv"):
            file_path = catalog
            row_index = self.event_id - 1
        else:
            raise ValueError(f"Invalid catalog: {catalog}")
        source_columns = self.SOURCE_COLUMNS_BASE + bands
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, usecols=source_columns, skiprows=range(1, row_index + 1), nrows=1)
                if "Y" in df.columns:
                    df["y"] = df.pop("Y")
                data = df.iloc[0].to_dict()
                return {**{k: data[k] for k in ["logL", "logTe"]}, **{band: data.get(band, np.nan) for band in self.bands}}
            except (KeyError, ValueError, IndexError) as e:
                print(f"Error loading source data for event_id {self.event_id}, {row_index=}: {e}")
        return {"logL": np.nan, "logTe": np.nan, **{band: np.nan for band in self.bands}}



    def _load_ulens_data(self) -> Dict[str, Any]:
        """
        Load ulens data from Genulens for the given event_id.

        Returns
        -------
        Dict[str, Any]
            Ulens data including D_L, D_S, mu_rel, etc.
        """
        chunk_id = (self.event_id // self.ROWS_PER_CHUNK) + 1
        row_index = self.event_id % self.ROWS_PER_CHUNK
        if chunk_id < 1 or chunk_id > 20:
            print(f"Invalid chunk_id: {chunk_id}")
            return {col: np.nan for col in self.ULENS_COLUMNS}

        file_path = os.path.join(self.TRILEGAL_DIR, f"Genulens_chunk_{chunk_id}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, usecols=self.ULENS_COLUMNS, skiprows=range(1, row_index + 1), nrows=1)
                return df.iloc[0].to_dict()
            except (KeyError, ValueError, IndexError) as e:
                print(f"Error loading ulens data for event_id {self.event_id}: {e}")
        return {col: np.nan for col in self.ULENS_COLUMNS}

    def simulate_ulens_parameters(
        self,
        ulens_data: Optional[Dict[str, Any]] = None,
        source_data: Optional[Dict[str, Any]] = None,
        peak_range: Optional[Tuple[float, float]] = None,
        blend: Union[bool, float] = False,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Simulate microlensing parameters for the event.

        Parameters
        ----------
        ulens_data : Dict[str, Any], optional
            Microlensing parameters; uses self.ulens_data if None.
        source_data : Dict[str, Any], optional
            Source properties; uses self.source_data if None.
        peak_range : Tuple[float, float], optional
            Time range for t0 simulation (MJD).
        blend : Union[bool, float], optional
            Blending fraction or True for random blending (default: False).
        save : bool, optional
            Save parameters to self.parameters (default: True).

        Returns
        -------
        Dict[str, Any]
            Simulated parameters.

        Raises
        ------
        ValueError
            If model is not supported.
        """
        ulens_data = ulens_data or self.ulens_data
        source_data = source_data or self.source_data
        if peak_range and peak_range[0] < 2400000.5:
            peak_range = (peak_range[0] + 2400000.5, peak_range[1] + 2400000.5)

        if self.model == "Pacz":
            params = pacz_parameters(source_data, self.bands, peak_range or (0, 1), self.event_id)
        elif self.model in ["USBL", "FSPL", "PSPL"]:
            params = pylima_parameters(
                self.model,
                self.system_type,
                self.bands,
                source_data,
                ulens_data,
                peak_range or (0, 1),
                blend=blend,
                event_id=self.event_id,
            )
        else:
            raise ValueError(f"Model {self.model} not implemented. Try: 'Pacz', 'USBL', 'FSPL', 'PSPL'")

        if params["t0"] > 2400000.5:
            params["t0"] -= 2400000.5
        if save:
            self.parameters = params
        return params

    def simulate_lc(
        self,
        epochs: Union[np.ndarray, Dict[str, np.ndarray]],
        params: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        parallax: Optional[bool] = None,
        num_chunks: int = 1,
        processes: int = 1,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Simulate light curves for the event.

        Parameters
        ----------
        epochs : Union[np.ndarray, Dict[str, np.ndarray]]
            Observation times per band or single array.
        params : Dict[str, Any], optional
            Model parameters; uses self.parameters if None.
        model : str, optional
            Microlensing model; uses self.model if None.
        parallax : bool, optional
            Include parallax; uses self.parallax if None.
        num_chunks : int, optional
            Number of chunks for parallel processing (default: 1).
        processes : int, optional
            Number of parallel processes (default: 1).
        save : bool, optional
            Append results to self.photometry (default: True).

        Returns
        -------
        pd.DataFrame
            Simulated photometry with specified columns.

        Raises
        ------
        ValueError
            If model is not valid.
        """
        params = params or self.parameters
        model = model or self.model
        parallax = parallax if parallax is not None else self.parallax
        bands = self.bands if isinstance(epochs, np.ndarray) else list(epochs.keys())

        if params["t0"] < 2400000.5:
            params["t0"] += 2400000.5
        for band in epochs:
            if epochs[band][0] < 2400000.5:
                epochs[band] = np.array(epochs[band], dtype=float) + 2400000.5

        photometry_data = []
        with contextlib.redirect_stdout(io.StringIO()):
            if model == "Pacz":
                mags = simulate_pacz_lightcurve(params, epochs)
                magnification = {band: np.ones_like(epochs[band]) for band in bands}  # Placeholder
            elif model in ["USBL", "FSPL", "PSPL"]:
                mags, magnification, pyLIMA_parameters, warns = simulate_pylima_lightcurve(
                    epochs, params, self.event_id, self.ra, self.dec, model, parallax, num_chunks, processes
                )
                self.pylima_parameters = pyLIMA_parameters
            else:
                raise ValueError(f"Model {model} is not valid. Try: 'Pacz', 'USBL', 'FSPL', 'PSPL'")

            for band in bands:
                for t, mag, A in zip(epochs[band], mags[band], magnification[band]):
                    photometry_data.append({
                        "event_id": self.event_id,
                        "time": t - 2400000.5,
                        "band": band,
                        "ideal_mag": mag,
                        "meas_mag": np.nan,
                        "meas_mag_err": np.nan,
                        "meas_flux": np.nan,
                        "meas_flux_err": np.nan,
                        "magnification": A,
                        "injection_flag": None,
                        "measure_flag": None,
                    })

        df = pd.DataFrame(photometry_data).astype(self.photometry.dtypes)
        if save:
            self.photometry = pd.concat([self.photometry, df], ignore_index=True)
        return df

    def plot(
        self,
        bands: Optional[List[str]] = None,
        show_ideal: bool = True,
        show_measured: bool = True,
        show_baseline: bool = True,
        simulate_ideal: bool = False,
        show_params: Union[bool, List[str]] = True,
        show_nearby_object: bool = True,
        survey_dates: Optional[Tuple[float, float]] = None,
        ax: Optional[plt.Axes] = None,
        colors: Optional[Dict[str, str]] = None,
        mark_flagged: bool = True,
        clean: bool = False,
        show_times: bool = True,
        show_mag_limits: Union[bool, Dict[str, Dict[str, float]]] = False,
        mag_limits: Union[bool, Dict[str, Dict[str, float]]] = False,
        mark_mag_limits: Union[bool, Dict[str, Dict[str, float]]] = False,
        show_legend: bool = True,
        y_limits: Union[bool, Tuple[float, float]] = False,
        max_points_to_plot: Optional[int] = None,
    ) -> plt.Axes:
        """
        Plot light curves for the event with inverted y-axis (magnitude).

        Parameters
        ----------
        bands : List[str], optional
            Bands to plot; defaults to self.bands.
        show_ideal : bool, optional
            Plot ideal light curve (default: True).
        show_measured : bool, optional
            Plot measured light curve with error bars (default: True).
        show_baseline : bool, optional
            Plot baseline magnitudes (default: True).
        simulate_ideal : bool, optional
            Simulate ideal light curve for plotting (default: False).
        show_params : Union[bool, List[str]], optional
            Show model parameters; if list, specifies parameters (default: True).
        show_nearby_object : bool, optional
            Show nearby object data in plot legend (default: True).
        survey_dates : Tuple[float, float], optional
            Survey start and end dates (MJD) to mark.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on; creates new if None.
        colors : Dict[str, str], optional
            Custom color mapping for bands.
        mark_flagged : bool, optional
            Mark flagged points (injection/measurement) (default: True).
        clean : bool, optional
            Plot only unflagged points within magnitude limits (default: False).
        show_times : bool, optional
            Show t0, t_center, and tE spans (default: True).
        show_mag_limits : Union[bool, Dict[str, Dict[str, float]], optional
            Show saturation and 5-sigma magnitude limits (default: False).
        mag_limits : Union[bool, Dict[str, Dict[str, float]], optional
            Dictionary magnitude limits (default: False).
        mark_mag_limits : bool, optional
            Mark points outside magnitude limits (default: False).
        show_legend : bool, optional
            Show legend (default: True).
        y_limits : Union[bool, Tuple[float, float]], optional
            Set y-axis limits; if True, uses baseline ± margins (default: False).
        max_points_to_plot : int, optional
            Maximum number of points to plot per band.

        Returns
        -------
        matplotlib.axes.Axes
            Plot axes.
        """
        bands = bands or self.bands
        colors = colors or self.DEFAULT_COLORS
        mag_sat = mark_mag_limits["m_sat"] if isinstance(mark_mag_limits, dict) else {}
        mag_5sigma = mark_mag_limits["m_5sigma"] if isinstance(mark_mag_limits, dict) else {}
        type_order = []

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        for band in bands:
            color = colors.get(band, "black")
            df_band = self.photometry[self.photometry["band"] == band].head(max_points_to_plot)

            # Handle magnitude limits
            m_5sigma = df_band["mag_lim"] if "mag_lim" in df_band.columns else mag_5sigma.get(band, np.inf)

            # Plot ideal light curve
            if show_ideal and "ideal_mag" in df_band.columns:
                ideal = df_band[["time", "ideal_mag"]].dropna()
                if not ideal.empty:
                    ax.plot(
                        ideal["time"],
                        ideal["ideal_mag"],
                        ".",
                        color=color,
                        label=f"{band}_ideal",
                        alpha=0.8,
                    )
                    type_order.append("ideal")

            # Plot measured light curve
            if show_measured and "meas_mag" in df_band.columns:
                if clean:
                    df_clean = df_band[
                        (df_band["injection_flag"].isna() | (df_band["injection_flag"] == 0))
                        & (df_band["measure_flag"].isna() | (df_band["measure_flag"] == ""))
                        & (df_band["meas_mag"] > mag_sat.get(band, -np.inf))
                        & (df_band["meas_mag"] < m_5sigma + 1)
                    ]
                    measured = df_clean[["time", "meas_mag", "meas_mag_err"]].dropna()
                else:
                    measured = df_band[["time", "meas_mag", "meas_mag_err"]].dropna()

                if not measured.empty:
                    ax.errorbar(
                        measured["time"],
                        measured["meas_mag"],
                        yerr=abs(measured["meas_mag_err"]),
                        fmt="o",
                        markersize=3,
                        color=color,
                        label=f"{band}_meas",
                        alpha=0.6,
                    )
                    type_order.append("meas")

                # Mark flagged points
                if mark_flagged:
                    for flag_type, marker in [("injection_flag", "^"), ("measure_flag", "s")]:
                        df_flag = df_band[
                            (df_band[flag_type].notna()) & (df_band[flag_type] != (0 if flag_type == "injection_flag" else ""))
                        ]
                        if not df_flag.empty:
                            ax.scatter(
                                df_flag["time"],
                                df_flag["meas_mag"],
                                marker=marker,
                                color=color,
                                edgecolor="k",
                                s=25,
                                alpha=0.8,
                                label=flag_type.split("_")[0] + "_flag" if band == bands[0] else None,
                            )
                            type_order.append("other")

                # Mark points outside magnitude limits
                if mark_mag_limits and isinstance(mark_mag_limits, dict):
                    df_bad = df_band[
                        (df_band["meas_mag"] < mag_sat.get(band, -np.inf))
                        | (df_band["meas_mag"] > m_5sigma + 1)
                    ]
                    if not df_bad.empty:
                        ax.scatter(
                            df_bad["time"],
                            df_bad["meas_mag"],
                            marker="o",
                            color="k",
                            s=25,
                            label="off_limits" if band == bands[0] else None,
                        )
                        type_order.append("other")

            # Plot baseline
            if show_baseline and band in self.parameters:
                ax.axhline(
                    y=self.parameters[band],
                    linestyle="--",
                    color=color,
                    label=f"{band}_base",
                    alpha=0.8,
                )
                type_order.append("base")

            # Plot magnitude limits
            if show_mag_limits and isinstance(show_mag_limits, dict):
                for limit, style, label in [
                    (show_mag_limits["m_sat"].get(band), "dotted", f"{band}_sat"),
                    (show_mag_limits["m_5sigma"].get(band), "dashdot", f"{band}_5sigma"),
                ]:
                    if limit is not None:
                        ax.axhline(
                            y=limit,
                            linewidth=0.6,
                            linestyle=style,
                            color=color,
                            label=label,
                            alpha=0.8,
                        )
                        type_order.append(label.split("_")[1])

            # Simulate ideal light curve
            if simulate_ideal:
                cadence = 0.5
                epochs = {
                    band: np.arange(min(self.photometry["time"]), max(self.photometry["time"]), cadence) + 2400000.5
                    for band in self.bands
                }
                params = self.parameters.copy()
                if params["t0"] < 2400000.5:
                    params["t0"] += 2400000.5
                mags, magnification, pyLIMA_parameters, warns = simulate_pylima_lightcurve(
                    epochs, params, self.event_id, self.ra, self.dec, self.model, self.parallax
                )
                ax.plot(
                    epochs[band] - 2400000.5,
                    mags[band],
                    lw=1,
                    label=f"{band}_theo",
                    color=color,
                )
                type_order.append("theo")

        # Plot survey dates
        if survey_dates:
            ax.axvline(x=survey_dates[0], color="black")
            ax.axvline(x=survey_dates[1], color="black", label="survey_dates")
            type_order.append("other")

        # Plot time markers
        if show_times and hasattr(self, "pylima_parameters"):
            t0 = self.pylima_parameters.get("t0", np.nan) - 2400000.5
            t_center = self.pylima_parameters.get("t_center", np.nan) - 2400000.5
            tE = self.pylima_parameters.get("tE", np.nan)
            if not np.isnan(t0):
                ax.axvline(t0, color="gray", linewidth=1, label=r"$t_0$")
            if not np.isnan(t_center):
                ax.axvline(t_center, linewidth=1, linestyle=":", color="gray", label=r"$t_{center}$")
            if not np.isnan(t0) and not np.isnan(tE):
                ax.axvspan(t0 - tE, t0 + tE, color="gray", alpha=0.3, label=r"$t_0 \pm t_E$")
            type_order.append("other")

        # Display parameters
        if show_params:
            default_params = ["tE", "u0", "t0"]
            selected_params = show_params if isinstance(show_params, list) else default_params
            param_lines = [f"Model: {self.model}", f"System Type: {self.system_type}"]
            for param in selected_params:
                if param in self.parameters:
                    val = self.parameters[param]
                    try:
                        val = float(val)
                        if abs(val) < 1e-2:
                            param_lines.append(f"{param:>8}: {val:.5e}")
                        elif param == "tE":
                            param_lines.append(f"{param:>8}: {val:.2f} days")
                        else:
                            param_lines.append(f"{param:>8}: {val:.2f}")
                    except (ValueError, TypeError):
                        param_lines.append(f"{param:>8}: {val}")
            if show_nearby_object and self.nearby_object:
                ns = self.nearby_object
                param_lines.append("")
                param_lines.append("Nearest source data:")
                param_lines.append(f" Distance: {ns['distance']:.2f} arcsec")
                param_lines.append(f" {'band':^4} {'mag':>6} {'FWHM':>6}")
                for band in self.bands:
                    mag = np.round(ns.get(f"mag_{band}", np.nan), 2)
                    fwhm = np.round(ns.get(f"fwhm_{band}", np.nan), 2)
                    if not np.isnan(mag) and not np.isnan(fwhm):
                        param_lines.append(f" {band:^4} {str(mag):>6} {str(fwhm):>6}")
            param_text = "\n".join(param_lines)
            plt.text(
                1.02,
                0.99,
                param_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Set axes properties
        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel("Magnitude")
        ax.invert_yaxis()
        baseline_values = [self.parameters.get(band, np.nan) for band in self.bands if band in self.parameters]
        if isinstance(y_limits, tuple):
            ax.set_ylim(y_limits[0], y_limits[1])
        elif y_limits and baseline_values:
            ax.set_ylim(max(baseline_values) + 2, min(baseline_values) - 7)

        ax.set_title(f"Event {self.event_id}")

        # Handle legend
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                band_order = self.bands
                type_order = set(type_order)
                grouped = {typ: {band: None for band in band_order} for typ in type_order if typ != "other"}
                others = []
                for h, l in zip(handles, labels):
                    if "_" in l:
                        band, typ = l.split("_", 1)
                        if typ in grouped and band in grouped[typ]:
                            grouped[typ][band] = (h, l)
                    else:
                        others.append((h, l))

                ordered_handles = []
                ordered_labels = []
                for band in band_order:
                    for typ in type_order:
                        if typ != "other" and grouped[typ][band]:
                            h, l = grouped[typ][band]
                            ordered_handles.append(h)
                            ordered_labels.append(l)
                ordered_handles.extend(h for h, _ in others)
                ordered_labels.extend(l for _, l in others)

                ncol = len(band_order) + (len(others) // (len(type_order) - 1) if type_order else 0)
                ax.legend(
                    ordered_handles,
                    ordered_labels,
                    fontsize=7,
                    ncol=int(ncol),
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.1),
                    frameon=True,
                )

        return ax

    def to_parquet(
        self,
        photometry_path: str,
        events_path: str,
        main_path: str = ".",
        photometry_mode: str = "overwrite",
        events_mode: str = "overwrite",
    ) -> None:
        """
        Save event data and photometry to Parquet files using Catalog.

        Parameters
        ----------
        photometry_path : str
            Path to photometry Parquet file.
        events_path : str
            Path to events Parquet file.
        main_path : str, optional
            Main directory for output files (default: '.').
        photometry_mode : str, optional
            Mode for photometry file ('append' or 'overwrite') (default: 'overwrite').
        events_mode : str, optional
            Mode for events file ('append' or 'overwrite') (default: 'overwrite').
        """
        if not self.photometry.empty:
            self.photometry["event_id"] = self.event_id 
            self.photometry["event_id"] = self.photometry["event_id"].astype("int32")
            photometry_table = pa.Table.from_pandas(self.photometry, preserve_index=False)
            photometry_schema = photometry_table.schema
            photometry_catalog = Catalog(photometry_path, schema=photometry_schema)
            photometry_catalog.add_rows(photometry_table.to_pylist(), mode=photometry_mode, schema=photometry_schema)


        row = {
            "event_id": self.event_id,
            "ra": self.ra,
            "dec": self.dec,
            "model": self.model,
            "system_type": self.system_type,
            "points": len(self.photometry),
            "logL": self.source_data.get("logL", np.nan),
            "logTe": self.source_data.get("logTe", np.nan),
            "D_L": self.ulens_data.get("D_L", np.nan),
            "D_S": self.ulens_data.get("D_S", np.nan),
            "mu_rel": self.ulens_data.get("mu_rel", np.nan),
            "nearby_object_ra": self.nearby_object.get("ra", None),
            "nearby_object_dec": self.nearby_object.get("dec", None),
            "nearby_object_objId": self.nearby_object.get("objectId", None),
            "nearby_object_distance": self.nearby_object.get("distance", None),
            "cadence_noise": self.cadence_noise,
            "peak_time": self.parameters.get("t0", np.nan),
            **{f"nearby_object_mag_{band}": self.nearby_object.get(f"mag_{band}", np.nan) for band in self.bands},
            **{f"nearby_object_fwhm_{band}": self.nearby_object.get(f"fwhm_{band}", np.nan) for band in self.bands},
            **{f"param_{k}": v for k, v in self.parameters.items()},
        }
        if self.model != "Pacz" and hasattr(self, "pylima_parameters"):
            row.update({f"param-pylima_{k}": v for k, v in self.pylima_parameters.items()})
        event_schema = self.events_schema or pa.Table.from_pandas(pd.DataFrame([row])).schema
        event_catalog = Catalog(events_path, schema=event_schema)
        
        event_catalog.add_rows([row], mode=events_mode, schema=event_schema)

        del row, event_catalog
        gc.collect()

    @classmethod
    def from_parquet(cls, event_id: int, photometry_path: str, events_path: str) -> "Event":
        """
        Load an event from Parquet files by event_id.

        Parameters
        ----------
        event_id : int
            ID of the event to load.
        photometry_path : str
            Path to photometry Parquet file.
        events_path : str
            Path to events Parquet file.

        Returns
        -------
        Event
            Event instance populated with loaded data.

        Raises
        ------
        FileNotFoundError
            If Parquet files do not exist.
        ValueError
            If event_id is not found in Parquet files.
        """
        if not os.path.exists(photometry_path):
            raise FileNotFoundError(f"Photometry file {photometry_path} does not exist.")
        if not os.path.exists(events_path):
            raise FileNotFoundError(f"Events file {events_path} does not exist.")

        photometry_table = pq.read_table(photometry_path, filters=[("event_id", "=", event_id)])
        photometry_df = photometry_table.to_pandas()
        if photometry_df.empty:
            raise ValueError(f"No photometry data found for event_id {event_id}.")

        events_table = pq.read_table(events_path, filters=[("event_id", "=", event_id)])
        events_df = events_table.to_pandas()
        if events_df.empty:
            raise ValueError(f"No event data found for event_id {event_id}.")

        event_row = events_df.iloc[0]
        event = cls(
            event_id=event_row["event_id"],
            ra=event_row["ra"],
            dec=event_row["dec"],
            bands=event_row.get("bands", ["u", "g", "r", "i", "z", "y"]),
            model=event_row["model"],
            system_type=event_row["system_type"],
            parallax=event_row.get("parallax", True),
            cadence_noise=event_row.get("cadence_noise", "ideal"),
            photometry=photometry_df,
            events_schema={f.name: f.type for f in events_table.schema},
        )

        event.bands = [band.lower() for band in event.bands]
        event.source_data = {
            "logL": event_row.get("logL", np.nan),
            "logTe": event_row.get("logTe", np.nan),
            **{band: event_row.get(f"param_{band}", np.nan) for band in event.bands},
        }
        event.ulens_data = {
            "D_L": event_row.get("D_L", np.nan),
            "D_S": event_row.get("D_S", np.nan),
            "mu_rel": event_row.get("mu_rel", np.nan),
        }
        event.nearby_object = {
            "ra": event_row.get("nearby_object_ra", None),
            "dec": event_row.get("nearby_object_dec", None),
            "objectId": event_row.get("nearby_object_objId", None),
            "distance": event_row.get("nearby_object_distance", None),
            **{f"mag_{band}": event_row.get(f"nearby_object_mag_{band}", np.nan) for band in event.bands},
            **{f"fwhm_{band}": event_row.get(f"nearby_object_fwhm_{band}", np.nan) for band in event.bands},
        }
        event.parameters = {
            key.replace("param_", ""): value for key, value in event_row.items() if key.startswith("param_") and not key.startswith("param-pylima_")
        }
        event.pylima_parameters = {
            key.replace("param-pylima_", ""): value for key, value in event_row.items() if key.startswith("param-pylima_")
        }
        return event

    def compute_chi2(
        self, cumulative: bool = True, statistic: str = "mean", flux_constant: Optional[float] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute chi-squared (χ²), degrees of freedom (dof), and p-value per band.

        Chi-squared is defined as:
            χ² = Σ_i [(f_i - F_ref) / σ_i]^2
        where f_i is measured flux, σ_i is flux error, and F_ref is a reference flux
        (mean, median, or constant).

        Parameters
        ----------
        cumulative : bool, optional
            Compute cumulative χ² up to each time point (default: True).
        statistic : str, optional
            Method to compute F_ref: 'mean', 'median', or 'constant' (default: 'mean').
        flux_constant : float, optional
            Reference flux if statistic='constant'.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with band as key and results as {'mjd': List[float], 'chi2': List[float],
            'dof': List[int], 'p_value': List[float]} for cumulative=True, or
            {'chi2': float, 'dof': int, 'p_value': float} for cumulative=False.
        """
        results = {}
        for band in self.bands:
            mask = self.photometry["band"] == band
            mjd = self.photometry["time"][mask].to_numpy()
            flux = self.photometry["meas_flux"][mask].to_numpy()
            flux_err = self.photometry["meas_flux_err"][mask].to_numpy()

            if len(flux) < 2:
                continue

            sort_idx = np.argsort(mjd)
            mjd = mjd[sort_idx]
            flux = flux[sort_idx]
            flux_err = flux_err[sort_idx]
            valid_mask = ~np.isnan(flux) & ~np.isnan(flux_err)
            mjd = mjd[valid_mask]
            flux = flux[valid_mask]
            flux_err = flux_err[valid_mask]

            if len(flux) < 2:
                continue

            def get_F_ref(f: np.ndarray, fe: np.ndarray) -> float:
                if statistic == "mean":
                    return np.sum(f / fe**2) / np.sum(1 / fe**2)
                elif statistic == "median":
                    return np.median(f)
                elif statistic == "constant":
                    if flux_constant is None:
                        raise ValueError("flux_constant must be provided when statistic='constant'")
                    return flux_constant
                raise ValueError("statistic must be 'mean', 'median', or 'constant'")

            results[band] = {}
            if cumulative:
                results[band]["mjd"] = []
                results[band]["chi2"] = []
                results[band]["dof"] = []
                results[band]["p_value"] = []
                for x in range(2, len(flux) + 1):
                    f = flux[:x]
                    fe = flux_err[:x]
                    F_ref = get_F_ref(f, fe)
                    chi2_val = np.sum(((f - F_ref) / fe) ** 2)
                    dof = len(f) - 1
                    p_value = chi2.sf(chi2_val, dof)
                    results[band]["mjd"].append(mjd[x - 1])
                    results[band]["chi2"].append(chi2_val)
                    results[band]["dof"].append(dof)
                    results[band]["p_value"].append(p_value)
            else:
                F_ref = get_F_ref(flux, flux_err)
                chi2_val = np.sum(((flux - F_ref) / flux_err) ** 2)
                dof = len(flux) - 1
                p_value = chi2.sf(chi2_val, dof)
                results[band] = {"chi2": chi2_val, "dof": dof, "p_value": p_value}

        return results