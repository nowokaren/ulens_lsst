"""
Analysis module for the LSST pipeline project.

This module provides functions to analyze and visualize microlensing simulation results,
including plotting light curves with field-of-view cutouts, generating sky maps, creating
GIF animations of light curves, and analyzing pipeline performance through log file parsing.
It is designed to integrate with :class:`simulation_pipeline.SimPipeline` and
:class:`light_curves.Event` for comprehensive post-processing of simulation data. The module
is optimized for memory efficiency by using selective data loading and pyarrow for Parquet
operations, and for time efficiency through streamlined plotting and data processing.

Functions
---------
plot_event_fov : Plots light curves and calexp cutouts for an event across multiple bands.
plot_sky_map : Generates a sky map showing injected events, sources, and calexp footprints.
event_gif : Creates a GIF animation of a light curve and calexp cutout for a single band.
generate_dataframe : Parses pipeline log files to create a DataFrame of task durations.
generate_summary_plot : Generates summary plots of pipeline performance, including task durations and event statistics.
get_folder_size : Calculates the total size of a folder and its contents.
format_size : Converts file size from bytes to a human-readable format.
"""

# Standard library imports
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from shapely.geometry import Polygon
from tqdm.auto import tqdm

# Local imports
from light_curves import Event
from lsst_data import Calexp, LSSTData
from lsst_tools import LSSTTools
from simulation_pipeline import SimPipeline


def plot_event_fov(
    event_id: int,
    sim_name: str,
    index_run: int = 1,
    bands: Optional[List[str]] = None,
    save: bool = True,
) -> None:
    """
    Plot light curves and calexp cutouts for an event across multiple bands.

    Generates a figure with two rows: the top row shows light curves for each band,
    and the bottom row shows corresponding calexp cutouts with injected sources.

    Parameters
    ----------
    event_id : int
        ID of the event to plot.
    sim_name : str
        Name of the simulation folder.
    index_run : int, optional
        Index of the run for file naming (default: 1).
    bands : List[str], optional
        Bands to plot; uses event.bands if None.
    save : bool, optional
        Save the plot to file if True; otherwise, display it (default: True).

    Raises
    ------
    ValueError
        If no valid photometry data is found or calexp loading fails.
    """
    sim = SimPipeline(from_folder=sim_name)
    events_file = os.path.join(sim.output_dir, f"data-events_{index_run}.parquet")
    calexps_photometry_file = os.path.join(sim.output_dir, f"calexps-photometry_{index_run}.parquet")
    calexps = pd.read_csv(os.path.join(sim.output_dir, "data_calexps.csv"))

    event = Event.from_parquet(event_id, calexps_photometry_file, events_file)
    bands = bands or event.bands
    fig, axs = plt.subplots(2, len(bands), figsize=(4 * len(bands), 8))

    for i, band in enumerate(tqdm(bands, desc=f"Event {event_id} - Plotting filter light curves")):
        # Plot light curve
        event.plot(
            bands=[band],
            ax=axs[0, i],
            show_params=False,
            show_measured=True, 
            clean=True,
            show_ideal=False,
            simulate_ideal=True,
            mag_limits={"m_sat": sim.mag_sat, "m_5sigma": sim.mag_5sigma},
            mark_mag_limits=False, 
            mark_flagged=False,
            show_times=False,
            show_legend=False,
            y_limits=False,
        )
        axs[0, i].grid()
        axs[0, i].legend(loc="upper right", fontsize=6)

        # Plot calexp cutout
        new_cutout = None
        i_calexp = 0
        while not isinstance(new_cutout, Calexp):
            phot_band = event.photometry[event.photometry["band"] == band]
            if i_calexp >= len(phot_band):
                break
            phot_data = phot_band.iloc[i_calexp]
            m_5sigma = phot_data["mag_lim"] if "mag_lim" in phot_band.columns else sim.mag_5sigma[band]
            if (
                phot_data["injection_flag"] != 0
                or phot_data["measure_flag"] != ""
                or phot_data["ideal_mag"] < sim.mag_sat[band]
                or phot_data["ideal_mag"] > m_5sigma + 1
            ):
                i_calexp += 1
                continue

            calexp_id = phot_data["calexp_id"]
            calexp_data = calexps.loc[calexp_id]
            calexp = Calexp(data_id=calexp_data[["visit", "detector"]].to_dict(), data_preview=sim.data_preview)
            tools = LSSTTools(calexp)
            injection_catalog = tools.create_injection_catalog(
                ra_values=[event.ra],
                dec_values=[event.dec],
                mag_values=[phot_data["ideal_mag"]],
                magnification_values=[phot_data["magnification"]],
                expMidptMJD=phot_data["time"],
                visit=calexp_data["visit"],
                detector=calexp_data["detector"],
                event_ids=[event_id],
            )
            injection_output = tools.inject_sources(injection_catalog)
            new_calexp = Calexp(data_id=injection_output["output_exposure"])

            try:
                new_cutout = new_calexp.cutout(roi=((event.ra, event.dec), 200))
            except RuntimeError as e:
                if "lies outside Exposure" in str(e):
                    print(f"Event {event_id}: Injected source outside calexp. (RA, Dec) = ({event.ra}, {event.dec})")
                    new_cutout = "no_overlap"
                    i_calexp += 1
                else:
                    raise

        if isinstance(new_cutout, Calexp):
            axs[1, i].remove()
            axs[1, i] = fig.add_subplot(2, len(bands), len(bands) + i + 1, projection=WCS(new_cutout.wcs.getFitsMetadata()))
            ax = new_cutout.plot(ax=axs[1, i], title=f"Epoch: {round(phot_data['time'])}; inj_mag = {round(phot_data['ideal_mag'], 2)}")
            new_cutout.add_point(ax, event.ra, event.dec, r=10)
            new_cutout.add_point(ax, event.nearby_object["ra"], event.nearby_object["dec"], c="b", r=10)
    red_circle = mpatches.Patch(color="red", label="Injected event")
    blue_circle = mpatches.Patch(color="blue", label="Nearby object")
    fig.legend(
        handles=[red_circle, blue_circle],
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=False
    )
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3, bottom=0.12)  # dejar espacio para la leyenda
    if save:
        os.makedirs(os.path.join(sim.main_path, sim.name, "event_plots"), exist_ok=True)
        plt.savefig(
            os.path.join(sim.main_path, sim.name, "event_plots", f"event_fov_{event_id}.png"),
            bbox_inches="tight",
            dpi=150,
        )
    else:
        plt.show()
    plt.close()


def plot_sky_map(
    sim_name: Optional[str] = None,
    circle_region: Optional[Tuple[Tuple[float, float], float]] = None,
    n_calexps: int = 5,
    zoom: float = 0.1,
    plot_injected: bool = True,
    file_name: str = "sky_map.png",
    sources_alpha: float = 0.8,
    sources_ms: float = 0.8,
    data_preview: str = "dp0",
    save: Optional[str] = None,
) -> None:
    """
    Generate a sky map showing injected events, pre-existing sources, and calexp footprints.

    Parameters
    ----------
    sim_name : str, optional
        Name of the simulation folder.
    circle_region : Tuple[Tuple[float, float], float], optional
        Tuple of ((ra, dec), radius) in degrees for the region; required if sim_name is None.
    n_calexps : int, optional
        Number of calexps to plot (default: 40).
    zoom : float, optional
        Zoom level in degrees for the plot (default: 0.1).
    plot_injected : bool, optional
        Plot injected event positions if True (default: True).
    file_name : str, optional
        Output file name for the plot (default: 'sky_map.png').
    sources_alpha : float, optional
        Alpha transparency for source markers (default: 0.6).
    sources_ms : float, optional
        Marker size for sources (default: 0.5).
    data_preview : str, optional
        Data preview type ('dp0', 'dp1') (default: 'dp0').
    save : str, optional
        Directory to save the plot; uses sim.output_dir if None and sim_name is provided.

    Raises
    ------
    ValueError
        If neither sim_name nor circle_region is provided.
    """
    if sim_name:
        sim = SimPipeline(from_folder=sim_name, new=False)
        ra0 = sim.sky_center["coord"][0] * u.deg if sim.sky_center["frame"] == "icrs" else SkyCoord(
            l=sim.sky_center["l"] * u.deg, b=sim.sky_center["b"] * u.deg, frame="galactic"
        ).icrs.ra
        dec0 = sim.sky_center["coord"][1] * u.deg if sim.sky_center["frame"] == "icrs" else SkyCoord(
            l=sim.sky_center["l"] * u.deg, b=sim.sky_center["b"] * u.deg, frame="galactic"
        ).icrs.dec
        radius = sim.sky_radius * u.deg
        data_preview = sim.data_preview
        name = sim.name
        DE = pd.read_parquet(sim.events_file)
        frame0 = sim.sky_center["frame"]
    elif circle_region:
        ra0, dec0, radius = circle_region[0][0] * u.deg, circle_region[0][1] * u.deg, circle_region[1] * u.deg
        frame0 = "icrs"
        name = f"Region {circle_region}"
        DE = pd.DataFrame(columns=["ra", "dec"])
    else:
        raise ValueError("Either sim_name or circle_region must be provided.")

    center = SkyCoord(ra=ra0, dec=dec0, frame=frame0).icrs
    circle = center.directional_offset_by(np.linspace(0, 360, 200) * u.deg, radius)

    lsst_data = LSSTData(ra=ra0.value, dec=dec0.value, data_preview=data_preview, radius=radius.value, name=name)
    sources = lsst_data.load_catalog("Source")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(circle.ra.deg, circle.dec.deg, "r-", label="Injected region")
    ax.scatter(sources["coord_ra"], sources["coord_dec"], c="r", s=sources_ms, alpha=sources_alpha, label="Pre-existing sources")
    if plot_injected and not DE.empty:
        ax.scatter(DE["ra"], DE["dec"], c="b", s=1, label="Injected sources")

    if n_calexps > 0 and sim_name:
        calexps = pd.read_csv(os.path.join(sim.output_dir, "data_calexps.csv")).head(n_calexps)
        for _, data in calexps.iterrows():
            calexp = Calexp(data[["visit", "detector"]].to_dict(), data_preview=data_preview)
            polygon = Polygon(zip(*calexp.get_corners()))
            x, y = polygon.exterior.xy
            ax.fill(x, y, alpha=0.2, fc="lightblue", ec="blue", label="calexp" if _ == 0 else None)

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.legend(loc="upper right")
    ax.invert_xaxis()
    ax.set_title(f"Injected region - {name} - ({len(sources)} sources)")
    if zoom:
        ra_zoom = zoom / np.cos(np.deg2rad(dec0.value))
        ax.set_xlim(ra0.value - ra_zoom, ra0.value + ra_zoom)
        ax.set_ylim(dec0.value - zoom, dec0.value + zoom)
    ax.grid(True)

    if save:
        os.makedirs(save, exist_ok=True)
        plt.savefig(os.path.join(save, "analysis", file_name), bbox_inches="tight", dpi=150)
    elif sim_name:
        os.makedirs(os.path.join(sim.output_dir, "analysis"), exist_ok=True)
        plt.savefig(os.path.join(sim.output_dir, "analysis", file_name), bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


def event_gif(
    event_id: int,
    sim_name: str,
    band: str,
    output_dir: Optional[str] = None,
    duration: float = 5,
    cutout_size: int = 200,
    point_radius: int = 20,
) -> None:
    """
    Create a GIF showing the light curve and calexp cutout for a single band.

    The light curve progressively adds points, synchronized with calexp cutouts.

    Parameters
    ----------
    event_id : int
        ID of the event to plot.
    sim_name : str
        Name of the simulation folder.
    band : str
        Filter band to plot (e.g., 'i').
    output_dir : str, optional
        Directory to save the GIF; defaults to sim's event_plots folder.
    duration : float, optional
        Duration of each frame in the GIF (seconds) (default: 5).
    cutout_size : int, optional
        Size of the cutout in pixels (default: 200).
    point_radius : int, optional
        Radius of the point marking the event in pixels (default: 20).

    Raises
    ------
    ValueError
        If no photometry data is found for the specified band.
    """
    sim = SimPipeline(from_folder=sim_name)
    calexps_photometry_file = os.path.join(sim.output_dir, f"calexps-photometry_1.parquet")
    events_file = sim.events_file
    calexps = pd.read_csv(os.path.join(sim.output_dir, "data_calexps.csv"))

    event = Event.from_parquet(event_id, calexps_photometry_file, events_file)
    phot_data = event.photometry[event.photometry["band"] == band]
    if phot_data.empty:
        raise ValueError(f"No photometry data found for band {band}")

    output_dir = output_dir or os.path.join(sim.main_path, sim.name, "event_plots")
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"event_fov_{event_id}_{band}.gif")
    temp_files = []

    start = phot_data["time"].min()
    end = phot_data["time"].max()

    for i, phot_row in enumerate(tqdm(phot_data.itertuples(), total=len(phot_data), desc=f"Creating GIF for event {event_id}, band {band}")):
        fig, axs = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [2, 1]})
        event.plot(
            bands=[band],
            ax=axs[0],
            show_params=False,
            clean=True,
            simulate_ideal=True,
            mark_mag_limits={"m_sat": sim.mag_sat, "m_5sigma": sim.mag_5sigma},
            show_times=False,
            show_legend=True,
            y_limits=False,
            max_points_to_plot=i + 1,
        )
        axs[0].grid()
        axs[0].set_xlim(start - 2, end + 2)
        axs[0].legend(loc="upper right", fontsize=6)
        axs[0].set_title(f"Event {event_id} - Band {band}")

        calexp_id = phot_row.calexp_id
        calexp_data = calexps.loc[calexp_id]
        calexp = Calexp(data_id=calexp_data[["visit", "detector"]].to_dict(), data_preview=sim.data_preview)
        tools = LSSTTools(calexp)
        injection_catalog = tools.create_injection_catalog(
            ra_values=[event.ra],
            dec_values=[event.dec],
            mag_values=[phot_row.ideal_mag],
            expMidptMJD=[phot_row.time],
            visit=calexp_data["visit"],
            detector=calexp_data["detector"],
            event_ids=[event_id],
        )
        injection_output = tools.inject_sources(injection_catalog)
        new_calexp = Calexp(data_id=injection_output["output_exposure"])

        try:
            new_cutout = new_calexp.cutout(roi=((event.ra, event.dec), cutout_size))
            axs[1].remove()
            axs[1] = fig.add_subplot(1, 2, 2, projection=WCS(new_cutout.wcs.getFitsMetadata()))
            ax = new_cutout.plot(ax=axs[1])
            new_cutout.add_point(ax, event.ra, event.dec, r=point_radius)
            ax.set_title(f"Epoch: {round(phot_row.time)} MJD")
        except RuntimeError as e:
            if "lies outside Exposure" in str(e):
                print(f"Event {event_id}: Injected source outside calexp. (RA, Dec) = ({event.ra}, {event.dec})")
                plt.close(fig)
                continue
            raise

        temp_file = os.path.join(output_dir, f"temp_frame_{event_id}_{band}_{i}.png")
        plt.tight_layout()
        plt.savefig(temp_file, bbox_inches="tight", dpi=150)
        temp_files.append(temp_file)
        plt.close(fig)

    with imageio.get_writer(gif_path, mode="I", duration=duration) as writer:
        for temp_file in temp_files:
            try:
                image = imageio.imread(temp_file)
                writer.append_data(image)
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    print(f"GIF saved to {gif_path}")


def parse_timestamp(line: str, timestamp_pattern: re.Pattern) -> Optional[datetime]:
    """
    Parse a timestamp from a log line using the provided regex pattern.

    Parameters
    ----------
    line : str
        Log line containing the timestamp.
    timestamp_pattern : re.Pattern
        Compiled regex pattern to extract the timestamp.

    Returns
    -------
    datetime, optional
        Parsed datetime object or None if parsing fails.
    """
    match = timestamp_pattern.match(line)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            return None
    return None


def parse_task(line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse task information from a log line.

    Parameters
    ----------
    line : str
        Log line to parse.

    Returns
    -------
    Tuple[Optional[str], Optional[str], Optional[str]]
        Tuple of (task description, item type, task ID).
    """
    patterns = [
        (r"Event (\d+): (.+)$", "Event"),
        (r"Calexp (\d+): (.+)$", "Calexp"),
        (r"simulate_lightcurves: (.+)$", "simulate_lightcurves"),
        (r"process_synthetic_photometry: (.+)$", "process_synthetic_photometry"),
        (r"\[lsst\.visitInjectTask\] INFO - (.+)$", "lsst.injection"),
        (r"\[lsst\.measurement\] INFO - (.+)$", "lsst.measurement"),
        (r"INFO - (Starting|Ending) (.+)$", None),
    ]
    for pattern, item in patterns:
        match = re.search(pattern, line)
        if match:
            task = match.group(2) if len(match.groups()) > 1 else match.group(1)
            task_id = match.group(1) if item in ["Event", "Calexp"] else None
            item = match.group(2) if item is None else item
            return task.strip(), item, task_id
    return None, None, None


def compute_durations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute task durations by calculating time differences between consecutive rows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'time', 'item', 'id', 'task'.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'duration' column (in seconds).
    """
    df = df.sort_values("time").copy()
    df["duration"] = np.nan
    for item_name in ["Event", "Calexp", "load_nearby_objects", "simulate_lightcurves", "process_synthetic_photometry"]:
        sub_df = df[df["item"] == item_name].copy()
        if not sub_df.empty:
            sub_df = sub_df.sort_values("time")
            if sub_df["id"].notna().any():
                sub_df["next_time"] = sub_df.groupby("id")["time"].shift(-1)
            else:
                sub_df["next_time"] = sub_df["time"].shift(-1)
            sub_df["duration"] = (sub_df["next_time"] - sub_df["time"]).dt.total_seconds()
            df.loc[sub_df.index, "duration"] = sub_df["duration"]
    return df


def generate_time_log(sim_name: str, output_folder: str = "runs") -> pd.DataFrame:
    """
    Parse the pipeline log file to create a DataFrame of task durations.

    Saves the DataFrame to a Parquet file for further analysis.

    Parameters
    ----------
    sim_name : str
        Simulation name (used as folder name).
    output_folder : str, optional
        Root folder where the log file is stored (default: 'runs').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'time', 'task', 'item', 'id', 'duration'.
    """
    timestamp_pattern = re.compile(r"\[([^\]]+)\]")
    log_path = os.path.join(output_folder, sim_name, "pipeline_1.log")

    data = []
    with open(log_path, "r") as file:
        for line in file:
            t, task, item, task_id = parse_timestamp(line, timestamp_pattern), *parse_task(line)
            if t and task and item:
                data.append({"time": t, "task": task, "item": item, "id": task_id})

    df = pd.DataFrame(data, columns=["time", "task", "item", "id"])
    df["detail"] = df["task"].str.extract(r"(\d+)").astype(float)
    df["task"] = df["task"].str.replace(r"^(Injecting).*", r"\1", regex=True)
    df["task"] = df["task"].str.replace(r"^(Measuring).*", r"\1", regex=True)
    mask = df["task"].str.startswith("End (")
    df["task"] = df["task"].str.replace(r"(Parallel_process|run_parallel_process)", "parallel_process", regex=True)
    df.loc[mask, "detail"] = 0
    df.loc[mask, "task"] = "End"
    df = compute_durations(df)

    output_path = os.path.join(output_folder, sim_name, "time_log.parquet")
    df.to_parquet(output_path, index=False)
    return df


def generate_summary_plot(sim_name: str, output_folder: str = "runs") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate summary plots of pipeline performance based on log data.

    Creates plots for task durations, event and calexp processing times, and event statistics.
    Saves the plots to the simulation's analysis directory.

    Parameters
    ----------
    sim_name : str
        Simulation name (used as folder name).
    output_folder : str, optional
        Root folder where the log file and outputs are stored (default: 'runs').

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Task-level and item-level summary DataFrames.
    """
    # Load preprocessed DataFrame
    input_path = os.path.join(output_folder, sim_name, "time_log.parquet")
    df = pd.read_parquet(input_path)

    # Summarize data
    summary_item = df.groupby("item").agg(count=("item", "count"), total_duration=("duration", "sum")).reset_index().sort_values("total_duration", ascending=False)
    summary_task = df.groupby(["item", "task"]).agg(count=("task", "count"), total_duration=("duration", "sum")).reset_index()
    summary_task = summary_task.sort_values(["item", "total_duration"], ascending=[True, False])
    count_map = summary_item.set_index("item")["count"].to_dict()
    summary_task["mean_duration"] = summary_task.apply(
        lambda row: row["total_duration"] / count_map.get(row["item"], np.nan), axis=1
    )
    summary_task = summary_task[summary_task["task"] != "Ending"]
    mask_starting = summary_task["task"] == "Starting"
    summary_task.loc[mask_starting, "task"] = summary_task.loc[mask_starting, "item"]

    event_duration = df[df["item"] == "Event"].groupby("id")["duration"].sum().reset_index().rename(columns={"duration": "total_duration"})
    calexp_duration = df[df["item"] == "Calexp"].groupby("id")["duration"].sum().reset_index().rename(columns={"duration": "total_duration"})

    sim = SimPipeline(from_folder=sim_name, new=False)
    tasks_only = [
        "Loading event",
        "All-faint light curves filter",
        "Simulate parameters",
        "Simulate light curve",
        "Save event",
        "Loading calexp",
        "Loading events contained in calexp",
        "Initialize LSSTTools",
        "Create injection catalog",
        "Injecting",
        "Measuring",
        "Save measurement calexp",
    ]
    custom_ec_tasks = [
        "Calexps",
        "Events",
        "process_synthetic_photometry.combine_temp_parquet_files",
        "process_synthetic_photometry.parallel_process",
        "process_synthetic_photometry.Starting",
        "load_nearby_objects",
        "simulate_lightcurves.combine_temp_parquet_files",
        "simulate_lightcurves.parallel_process",
        "simulate_lightcurves.Starting",
    ][::-1]
    rename_dict = {
        "simulate_lightcurves.Starting": "Events-setting",
        "simulate_lightcurves.parallel_process": "Events-parallel_processing",
        "simulate_lightcurves.combine_temp_parquet_files": "Events-combine_temp_files",
        "load_nearby_objects": "Load nearby objects",
        "process_synthetic_photometry.Starting": "Photometry-setting",
        "process_synthetic_photometry.parallel_process": "Photometry-parallel_processing",
        "process_synthetic_photometry.combine_temp_parquet_files": "Photometry-combine_temp_files",
        "Events": "Events",
        "Calexps": "Calexps",
    }

    # Prepare plot data
    duration_per_task = df.groupby("task")["duration"].sum().reset_index()
    bar_data_tasks = pd.DataFrame({"task": tasks_only, "total_duration": 0.0})
    for _, row in duration_per_task.iterrows():
        if row["task"] in tasks_only:
            bar_data_tasks.loc[bar_data_tasks["task"] == row["task"], "total_duration"] = row["duration"] / 60

    total_duration_events = event_duration["total_duration"].sum() / 60
    total_duration_calexps = calexp_duration["total_duration"].sum() / 60
    events_bar = pd.DataFrame({"task": ["Events"], "total_duration": [total_duration_events], "item": ["Events"]})
    calexps_bar = pd.DataFrame({"task": ["Calexps"], "total_duration": [total_duration_calexps], "item": ["Calexps"]})

    parallel_df = df[df["item"].isin(["simulate_lightcurves", "load_nearby_objects", "process_synthetic_photometry"]) & pd.notna(df["duration"])]
    parallel_bar = parallel_df[parallel_df["task"].isin(["Starting", "parallel_process", "combine_temp_parquet_files", "load_nearby_objects"])].groupby(["item", "task"])["duration"].sum().reset_index()
    parallel_bar["total_duration"] = parallel_bar["duration"] / 60
    parallel_bar.loc[parallel_bar["item"] == "simulate_lightcurves", "task"] = parallel_bar["item"] + "." + parallel_bar["task"]
    parallel_bar.loc[parallel_bar["item"] == "process_synthetic_photometry", "task"] = parallel_bar["item"] + "." + parallel_bar["task"]
    parallel_bar.loc[parallel_bar["item"] == "load_nearby_objects", "task"] = parallel_bar["item"]

    bar_data_ec = pd.concat([events_bar, calexps_bar, parallel_bar[["task", "total_duration", "item"]]], ignore_index=True)
    bar_data_ec["task"] = pd.Categorical(bar_data_ec["task"], categories=custom_ec_tasks, ordered=True)
    bar_data_ec = bar_data_ec.sort_values("task")

    tasks_box = df[df["task"].isin(tasks_only) & (~df["duration"].isna())][["task", "duration"]].copy()
    box_data_tasks = tasks_box.rename(columns={"duration": "duration"})
    box_data_tasks["duration"] = box_data_tasks["duration"] / 60
    box_data_tasks["task"] = pd.Categorical(box_data_tasks["task"], categories=tasks_only, ordered=True)

    events_box = event_duration.copy()
    events_box["duration"] = events_box["total_duration"] / 60
    events_box["task"] = "Events"
    events_box["item"] = "Events"

    calexps_box = calexp_duration.copy()
    calexps_box["duration"] = calexps_box["total_duration"] / 60
    calexps_box["task"] = "Calexps"
    calexps_box["item"] = "Calexps"

    parallel_box = parallel_df[parallel_df["task"].isin(["Starting", "parallel_process", "combine_temp_parquet_files", "load_nearby_objects"])][["item", "task", "duration"]].copy()
    parallel_box["duration"] = parallel_box["duration"] / 60
    parallel_box.loc[parallel_box["item"] == "simulate_lightcurves", "task"] = parallel_box["item"] + "." + parallel_box["task"]
    parallel_box.loc[parallel_box["item"] == "process_synthetic_photometry", "task"] = parallel_box["item"] + "." + parallel_box["task"]
    parallel_box.loc[parallel_box["item"] == "load_nearby_objects", "task"] = parallel_box["item"]
    box_data_ec = pd.concat([events_box[["task", "duration", "item"]], calexps_box[["task", "duration", "item"]], parallel_box[["task", "duration", "item"]]], ignore_index=True)
    box_data_ec["task"] = pd.Categorical(box_data_ec["task"], categories=custom_ec_tasks, ordered=True)
    box_data_ec.loc[~box_data_ec["task"].isin(["Events", "Calexps"]), "duration"] = 0

    # Simulation statistics
    objects_file = os.path.join(output_folder, sim_name, "object_catalog.csv")
    events_file = os.path.join(output_folder, sim_name, "data-events_1.parquet")
    coords = pd.read_parquet(events_file, columns=["ra", "dec"])
    objects = pd.read_csv(objects_file, usecols=["coord_ra", "coord_dec"])
    start_time = df["time"].min()
    end_time = df["time"].max()
    parallel_duration = df[df["item"].isin(["simulate_lightcurves", "load_nearby_objects", "process_synthetic_photometry"])]["duration"].sum()
    serial_duration = df[df["item"].isin(["Event", "Calexp"])]["duration"].sum()
    num_events = df[df["item"] == "Event"]["id"].nunique()
    net_events = len(df[ (df["item"] == "Event") & (df["task"].str.startswith("Saving"))])
    num_calexps = df[df["item"] == "Calexp"]["id"].nunique()
    net_calexps = len(df[(df["item"] == "Calexp") & (df["task"].str.startswith("Saving"))])

    calexps_photometry_file = os.path.join(output_folder, sim_name, "calexps-photometry_1.parquet")
    from collections import Counter
    counter = Counter()
    counter_valid = Counter()
    import pyarrow.dataset as ds
    dataset = ds.dataset(calexps_photometry_file, format="parquet")
    scanner = dataset.scanner(columns=["event_id", "band", "injection_flag", "measure_flag"], batch_size=100_000)
    for batch in scanner.to_batches():
        event_ids = batch.column("event_id").to_pylist()
        bands = batch.column("band").to_pylist()
        inj_flags = batch.column("injection_flag").to_pylist()
        meas_flags = batch.column("measure_flag").to_pylist()
        for eid, b, inj, meas in zip(event_ids, bands, inj_flags, meas_flags):
            counter[(eid, b)] += 1
            if inj == 0 and (meas is None or meas == ""):
                counter_valid[(eid, b)] += 1
    counts = list(counter.values())
    counts_valid = list(counter_valid.values())

    # Create figure
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 0.1, 0.7, 0.7], hspace=0.5, wspace=0.3)

    # Summary text
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.axis("off")
    radius = sim.sky_radius
    radius_arcmin = radius * 60
    radius_arcsec = radius * 3600
    area_deg2 = np.pi * (radius**2)
    area_arcmin2 = np.pi * (radius_arcmin**2)
    area_arcsec2 = np.pi * (radius_arcsec**2)
    density_events = num_events / area_arcmin2
    num_sources = len(objects)
    density_sources = num_sources / area_arcmin2
    info_text = (
        f"Name: {sim_name}\n\n"
        f"Simulation type: {sim.sim_type}: {sim.data_preview}\n"
        f"Total duration (parallel): {parallel_duration/3600:.1f} hours\n"
        f"Total duration (serial): {serial_duration/3600:.1f} hours\n"
        f"Number of events: {num_events}\n"
        f"    Net events: {net_events} ({100*net_events/num_events:.1f}%)\n"
        f"Number of calexps: {num_calexps}\n"
        f"    Net calexps: {net_calexps} ({100*net_calexps/num_calexps:.1f}%)\n\n"
        f"Radius: {radius:.1f} deg = {radius_arcmin:.1f} arcmin = {radius_arcsec:.1f} arcsec\n"
        f"Area: {area_deg2:.1f} $\\text{{deg}}^2$ = {area_arcmin2:.1f} $\\text{{arcmin}}^2$ = {area_arcsec2:.1f} $\\text{{arcsec}}^2$\n"
        f"Injected density: {density_events:.1f} $\\frac{{events}}{{arcmin^2}}$\n"
        f"Objects density: {density_sources:.1f} $\\frac{{sources}}{{arcmin^2}}$"
    )
    ax1.text(-0.2, 0.5, info_text, fontsize=8, va="center", ha="left")

    # Sky map
    ax2 = fig.add_subplot(gs[0, 3])
    ax2.scatter(objects["coord_ra"], objects["coord_dec"], s=0.6, c="gray", alpha=0.2, label="Objects")
    ax2.scatter(coords["ra"], coords["dec"], c="red", s=0.6, label="Injected events")
    ax2.grid()
    ax2.set_title("Sky map")
    ax2.legend(loc=(-0.3, -0.37))
    ax2.set_xlabel("RA (deg)")
    ax2.set_ylabel("Dec (deg)")

    # Task bar and boxplot
    ax3 = fig.add_subplot(gs[0, :2])
    y_pos_tasks = np.arange(len(tasks_only))
    ax3.barh(y=y_pos_tasks, width=bar_data_tasks["total_duration"], color="skyblue", edgecolor="k", alpha=0.5)
    ax3.set_yticks(y_pos_tasks)
    ax3.set_yticklabels(tasks_only, fontsize=8)
    ax3.grid(axis="x", linestyle="--", alpha=0.7)
    ax3.set_xlabel("Total duration (minutes)", fontsize=8)
    ax3.set_title("Time analysis per task")

    ax_box3 = ax3.twiny()
    sns.boxplot(
        x="duration",
        y="task",
        data=box_data_tasks,
        ax=ax_box3,
        showcaps=True,
        showfliers=False,
        boxprops=dict(alpha=1),
        medianprops=dict(color="red"),
        whiskerprops=dict(alpha=1),
        capprops=dict(alpha=1, lw=2),
        width=0.4,
        palette=["lightcoral"] * len(tasks_only),
        legend=False,
        order=tasks_only,
    )
    ax_box3.set_yticks(y_pos_tasks)
    ax_box3.grid(axis="x", alpha=0.8)
    ax_box3.set_xlim(0, None)
    ax_box3.set_xlabel("Duration (min)", fontsize=8)

    # Histogram of events per calexp
    ax5 = fig.add_subplot(gs[1, 2])
    inject_per_calexp = [n for n in df.detail[df.task.str.startswith("Measuring")]]
    ax5.hist(inject_per_calexp, bins=20, color="steelblue")
    ax5.grid(axis="y", linestyle="--", alpha=0.7)
    ax5.set_title("Events per calexp")
    ax5.set_xlabel("Number of events")
    ax5.set_ylabel("Frequency")

    # Histogram of points per event
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.hist(counts, bins=30, color="skyblue", edgecolor="k", alpha=0.7, label="all")
    ax6.hist(counts_valid, bins=30, color="green", edgecolor="k", alpha=0.7, label="clean")
    ax6.set_title("Points per event")
    ax6.set_xlabel("Number of dots")
    ax6.set_ylabel("Frequency")
    ax6.grid(axis="y", linestyle="--", alpha=0.7)
    ax6.legend(loc="upper right")

    # Item bar and boxplot
    ax7 = fig.add_subplot(gs[1, :2])
    y_pos_ec = np.arange(len(custom_ec_tasks))
    bar_data_ec = bar_data_ec[bar_data_ec["task"].isin(custom_ec_tasks)].copy()
    for task in custom_ec_tasks:
        if task not in bar_data_ec["task"].values:
            bar_data_ec = pd.concat(
                [bar_data_ec, pd.DataFrame({"task": [task], "total_duration": [0.0], "item": ["unknown"]})],
                ignore_index=True,
            )
    bar_data_ec["task"] = pd.Categorical(bar_data_ec["task"], categories=custom_ec_tasks, ordered=True)
    ax7.barh(y=y_pos_ec, width=bar_data_ec["total_duration"], color="skyblue", edgecolor="k", alpha=0.5)
    ax7.set_yticks(y_pos_ec)
    ax7.set_yticklabels([rename_dict.get(task, task) for task in custom_ec_tasks], fontsize=8)
    ax7.grid(axis="x", linestyle="--", alpha=0.7)
    ax7.set_xlabel("Total duration (minutes)", fontsize=8)
    ax7.set_title("Time analysis per step")

    ax_box7 = ax7.twiny()
    box_data_ec = box_data_ec[box_data_ec["task"].isin(custom_ec_tasks)].copy()
    for task in custom_ec_tasks:
        if task not in box_data_ec["task"].values:
            box_data_ec = pd.concat(
                [box_data_ec, pd.DataFrame({"task": [task], "duration": [0.0], "item": ["unknown"]})],
                ignore_index=True,
            )
    box_data_ec["task"] = pd.Categorical(box_data_ec["task"], categories=custom_ec_tasks, ordered=True)
    sns.boxplot(
        x="duration",
        y="task",
        data=box_data_ec,
        ax=ax_box7,
        showcaps=True,
        showfliers=False,
        boxprops=dict(alpha=1),
        medianprops=dict(color="red"),
        whiskerprops=dict(alpha=1),
        capprops=dict(alpha=1, lw=2),
        width=0.4,
        palette=["lightcoral"] * len(custom_ec_tasks),
        legend=False,
        order=custom_ec_tasks,
    )
    ax_box7.set_yticks(y_pos_ec)
    ax_box7.grid(axis="x", alpha=0.8)
    ax_box7.set_xlim(0, None)
    ax_box7.set_xlabel("Duration (min)", fontsize=8)

    plt.savefig(os.path.join(output_folder, sim_name, "analysis", "sim_summary.png"), bbox_inches="tight", dpi=150)
    plt.close()
    return summary_task, summary_item


def get_folder_size(folder_path: str) -> int:
    """
    Calculate the total size of a folder and its contents in bytes.

    Parameters
    ----------
    folder_path : str
        Path to the folder.

    Returns
    -------
    int
        Total size in bytes.
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, filename))
    return total_size


def format_size(size_in_bytes: int) -> str:
    """
    Convert size from bytes to a human-readable format (B, KB, MB, GB).

    Parameters
    ----------
    size_in_bytes : int
        Size in bytes.

    Returns
    -------
    str
        Human-readable size string.
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} TB"