"""
Microlensing utilities module for the LSST pipeline project.

This module provides tools for generating microlensing parameters and simulating light curves
using pyLIMA, tailored for LSST/Rubin Observatory data (e.g., DP0 simulations, DP1 real data).
It supports parameter generation for various system types (e.g., planets, binaries, black holes,
free-floating planets) and efficient light curve simulation with optional parallax effects.
The module is optimized for memory and time efficiency through streamlined parameter calculations
and parallel processing capabilities, making it suitable for large-scale microlensing experiments
such as studying light curve recoverability and blending effects.

This module contains helper functions for processing microlensing events,
originally authored by [Colleague's Name]. The code has been adapted and
refined by [Your Name] for integration with the ulens_lsst pipeline,
ensuring compatibility and optimization for LSST data processing.

Original Author: Anibal Varela
Adapted by: Karen Nowogrodzki

Classes
-------
MicrolensingParams : Computes microlensing parameters for different system types.

Functions
-------
generate_event_params : Generates microlensing parameters for a given system type.
create_rubin_telescopes : Creates pyLIMA Telescope objects for Rubin filters.
create_rubin_event : Creates a pyLIMA Event for Rubin data.
simulate_lightcurve : Simulates a microlensing light curve using pyLIMA.
flux_to_mag : Converts flux to magnitude using zero points.
compute_model_lightcurves : Computes light curves (magnification, flux, magnitude) for a pyLIMA model.
simulate_ideal_microlensing_event : Simulates an ideal microlensing event for Rubin data.
pylima_parameters : Generates pyLIMA-compatible parameters with blending.
pacz_parameters : Generates Paczynski model parameters.
simulate_pacz_lightcurve : Simulates light curves using the Paczynski model.
suppress_stdout : Context manager to suppress stdout during pyLIMA operations.
"""

# Standard library imports
import os
import sys
from contextlib import contextmanager
from functools import partial
from typing import Dict, List, Optional, Tuple, Union
import copy


# Third-party imports
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import QTable
from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA.models import FSPLarge_model, PSBL_model, PSPL_model, USBL_model
from pyLIMA.simulations import simulator
from tqdm.auto import tqdm
from multiprocessing import Pool
import warnings

# Constants
C = const.c
G = const.G
K = 4 * G / (C**2)
TSTART_ROMAN = 2461508.763828608
T0 = TSTART_ROMAN + 20
ZP = {
    "W149": 27.615,
    "u": 27.03,
    "g": 28.38,
    "r": 28.16,
    "i": 27.85,
    "z": 27.46,
    "y": 26.68,
}


class MicrolensingParams:
    """
    Class to compute microlensing parameters for various system types.

    Attributes
    ----------
    name : str
        Name of the system type (e.g., 'Planets_systems').
    orbital_period : astropy.units.Quantity
        Orbital period in years.
    semi_major_axis : astropy.units.Quantity
        Semi-major axis in AU.
    DL : astropy.units.Quantity
        Lens distance in parsecs.
    mass_star : astropy.units.Quantity
        Star mass in solar masses.
    mass_planet : astropy.units.Quantity
        Planet mass in Jupiter masses.
    DS : astropy.units.Quantity
        Source distance in parsecs.
    mu_rel : astropy.units.Quantity
        Relative proper motion in mas/year.
    logTe : float
        Log10 of effective temperature in K.
    logL : float
        Log10 of luminosity in L_sun.
    """

    def __init__(
        self,
        name: str,
        orbital_period: float,
        semi_major_axis: float,
        DL: float,
        star_mass: float,
        mass_planet: float,
        DS: float,
        mu_rel: float,
        logTe: float,
        logL: float,
    ):
        """
        Initialize MicrolensingParams with system parameters.

        Parameters
        ----------
        name : str
            System type name.
        orbital_period : float
            Orbital period in days.
        semi_major_axis : float
            Semi-major axis in AU.
        DL : float
            Lens distance in parsecs.
        star_mass : float
            Star mass in solar masses.
        mass_planet : float
            Planet mass in Jupiter masses.
        DS : float
            Source distance in parsecs.
        mu_rel : float
            Relative proper motion in mas/year.
        logTe : float
            Log10 of effective temperature in K.
        logL : float
            Log10 of luminosity in L_sun.
        """
        self.name = name
        self.orbital_period = (orbital_period * u.day).to(u.year)
        self.semi_major_axis = semi_major_axis * u.au
        self.DL = DL * u.pc
        self.mass_star = star_mass * u.M_sun
        self.mass_planet = mass_planet * u.M_jup
        self.DS = DS * u.pc
        self.mu_rel = mu_rel * (u.mas / u.year)
        self.logTe = logTe
        self.logL = logL

    def mass_ratio(self) -> float:
        """
        Compute the mass ratio (planet/star).

        Returns
        -------
        float
            Mass ratio (dimensionless).
        """
        return (self.mass_planet / self.mass_star).decompose()

    def m_lens(self) -> u.Quantity:
        """
        Compute the total lens mass.

        Returns
        -------
        astropy.units.Quantity
            Total lens mass in solar masses.
        """
        if np.isnan(self.mass_planet):
            return self.mass_star.decompose().to(u.M_sun)
        return (self.mass_star + self.mass_planet).decompose().to(u.M_sun)

    def pi_rel(self) -> u.Quantity:
        """
        Compute the relative parallax.

        Returns
        -------
        astropy.units.Quantity
            Relative parallax in radians.

        Raises
        ------
        ValueError
            If DL >= DS.
        """
        if self.DL < self.DS:
            return ((1 / self.DL) - (1 / self.DS)) * u.rad
        raise ValueError("Invalid distance combination: DL >= DS")

    def theta_E(self) -> u.Quantity:
        """
        Compute the Einstein radius.

        Returns
        -------
        astropy.units.Quantity
            Einstein radius in milliarcseconds (mas).
        """
        theta_E_rad = np.sqrt(K * self.pi_rel() * self.m_lens())
        return theta_E_rad.to(u.mas, equivalencies=u.dimensionless_angles())

    def tE(self) -> u.Quantity:
        """
        Compute the Einstein crossing time.

        Returns
        -------
        astropy.units.Quantity
            Einstein crossing time in days.
        """
        return (self.theta_E() / self.mu_rel).to(u.day)

    def piE(self) -> float:
        """
        Compute the microlensing parallax (dimensionless).

        Returns
        -------
        float
            Microlensing parallax.
        """
        return (u.au * self.pi_rel() / self.theta_E()).decompose()

    def source_radius(self) -> u.Quantity:
        """
        Compute the source radius.

        Returns
        -------
        astropy.units.Quantity
            Source radius in solar radii.
        """
        L_star = 10**self.logL * const.L_sun
        Teff = (10**self.logTe) * u.K
        return np.sqrt(L_star / (4 * np.pi * const.sigma_sb * Teff**4)).to("R_sun")

    def thetas(self) -> u.Quantity:
        """
        Compute the angular source radius.

        Returns
        -------
        astropy.units.Quantity
            Angular source radius in milliarcseconds (mas).

        Raises
        ------
        ValueError
            If DL >= DS.
        """
        if self.DL < self.DS:
            theta_S_rad = (self.source_radius() / self.DS).decompose()
            return theta_S_rad.to(u.mas, equivalencies=u.dimensionless_angles())
        raise ValueError("Invalid distance combination: DL >= DS")

    def rho(self) -> float:
        """
        Compute the normalized source radius (dimensionless).

        Returns
        -------
        float
            Normalized source radius.
        """
        return (self.thetas() / self.theta_E()).decompose()

    def s(self) -> float:
        """
        Compute the normalized separation (dimensionless).

        Returns
        -------
        float
            Normalized separation.

        Raises
        ------
        ValueError
            If DL >= DS.
        """
        if self.DL < self.DS:
            s_rad = (self.semi_major_axis / self.DL).decompose()
            s_mas = s_rad.to(u.mas, equivalencies=u.dimensionless_angles())
            return s_mas / self.theta_E()
        raise ValueError("Invalid distance combination: DL >= DS")

    def u0(self, criterion: str = "caustic_proximity") -> float:
        """
        Compute the impact parameter based on criterion.

        Parameters
        ----------
        criterion : str, optional
            Criterion for u0 calculation ('caustic_proximity' or 'resonant_region') (default: 'caustic_proximity').

        Returns
        -------
        float
            Impact parameter.

        Raises
        ------
        ValueError
            If criterion is unknown.
        """
        random_factor = np.random.uniform(0, 3)
        if criterion == "caustic_proximity":
            return random_factor * self.rho()
        if criterion == "resonant_region":
            return 1 / self.s() - self.s()
        raise ValueError(f"Unknown criterion: {criterion}")

    def piE_comp(self) -> Tuple[float, float]:
        """
        Compute components of microlensing parallax (piEE, piEN).

        Returns
        -------
        Tuple[float, float]
            piEE and piEN components.
        """
        phi = np.random.uniform(0, np.pi)
        piEE = self.piE() * np.cos(phi)
        piEN = self.piE() * np.sin(phi)
        return piEE, piEN

    def orbital_motion(self, sz: float = 2, a_s: float = 1) -> Tuple[float, float, float, float, float]:
        """
        Compute orbital motion velocities.

        Parameters
        ----------
        sz : float, optional
            Scaling factor for separation (default: 2).
        a_s : float, optional
            Scaling factor for semi-major axis (default: 1).

        Returns
        -------
        Tuple[float, float, float, float, float]
            r_s, a_s, gamma1, gamma2, gamma3 (orbital motion parameters).
        """
        r_s = sz / self.s()
        n = 2 * np.pi / self.orbital_period
        denominator = a_s * np.sqrt((-1 + 2 * a_s) * (1 + r_s**2))
        velocity_magnitude = n * denominator

        gamma = np.random.normal(size=3)
        gamma *= velocity_magnitude.value / np.linalg.norm(gamma)
        return r_s, a_s, gamma[0], gamma[1], gamma[2]


def generate_event_params(
    random_seed: int,
    data_SOURCE: Dict[str, float],
    data_Genulens: Dict[str, float],
    system_type: str,
    t0_range: List[float] = [T0 - 365.25 * 2, T0 + 365.25 * 6],
    custom_system: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Generate microlensing parameters for a given system type.

    Parameters
    ----------
    random_seed : int
        Seed for random number generation.
    data_SOURCE : Dict[str, float]
        Source data including 'logL' and 'logTe'.
    data_Genulens : Dict[str, float]
        Genulens data including 'D_L', 'D_S', 'mu_rel'.
    system_type : str
        Type of system ('Planets_systems', 'Binary_stars', 'BH', 'FFP', 'custom').
    t0_range : List[float], optional
        Range for t0 [min, max] in JD (default: [T0 - 2 years, T0 + 6 years]).
    custom_system : Dict[str, float], optional
        Custom parameters for 'custom' system type.

    Returns
    -------
    Dict[str, float]
        Microlensing parameters.

    Raises
    ------
    ValueError
        If system_type is unknown or custom_system is missing for 'custom' type.
    """
    np.random.seed(random_seed)
    DL = data_Genulens["D_L"]
    DS = data_Genulens["D_S"]
    mu_rel = data_Genulens["mu_rel"]
    logL = data_SOURCE["logL"]
    logTe = data_SOURCE["logTe"]
    orbital_period = 0
    semi_major_axis = np.random.uniform(0.1, 28)

    if system_type == "Planets_systems":
        star_mass = np.random.uniform(1, 100)
        mass_planet = np.random.uniform(1 / 300, 13)
    elif system_type == "Binary_stars":
        star_mass = np.random.uniform(1, 50)
        mass_planet = np.random.uniform(1, 50) * u.M_sun.to("M_jup")
    elif system_type == "BH":
        star_mass = np.random.uniform(1, 100)
        mass_planet = 0
    elif system_type == "FFP":
        star_mass = 0
        mass_planet = np.random.uniform(3.146351865506143e-05, 20)
    elif system_type == "custom":
        if custom_system is None:
            raise ValueError("custom_system must be provided for 'custom' system_type")
        star_mass = custom_system.get("star_mass")
        mass_planet = custom_system.get("mass_planet")
    else:
        raise ValueError(f"Unknown system_type: {system_type}")

    event_params = MicrolensingParams(
        system_type,
        orbital_period,
        semi_major_axis,
        DL,
        star_mass,
        mass_planet,
        DS,
        mu_rel,
        logTe,
        logL,
    )

    t0 = np.random.uniform(*t0_range)
    rho = event_params.rho()
    tE = event_params.tE()
    piE = event_params.piE()
    u0 = rho.value * np.random.uniform(-3, 3) if system_type == "Planets_systems" else np.random.uniform(-2, 2)
    alpha = np.random.uniform(0, np.pi)
    angle = np.random.uniform(0, 2 * np.pi)
    piEE = piE * np.cos(angle)
    piEN = piE * np.sin(angle)

    params_ulens = {
        "t0": t0,
        "u0": u0,
        "tE": tE.value,
        "piEN": piEN.value,
        "piEE": piEE.value,
        "radius": float(event_params.source_radius().value),
        "mass_star": star_mass,
        "mass_planet": mass_planet,
        "thetaE": event_params.theta_E().value,
    }

    if system_type in ["FFP", "Binary_stars", "Planets_systems"]:
        params_ulens["rho"] = rho.value
    if system_type in ["Binary_stars", "Planets_systems"]:
        params_ulens["s"] = event_params.s().value
        params_ulens["q"] = event_params.mass_ratio().value
        params_ulens["alpha"] = alpha

    return params_ulens


def create_rubin_telescopes(rubin_ts: Dict[str, np.ndarray]) -> Dict[str, telescopes.Telescope]:
    """
    Create pyLIMA Telescope objects for Rubin filters.

    Parameters
    ----------
    rubin_ts : Dict[str, np.ndarray]
        Dictionary of time series per band, each with shape (n, 3) for [time, mag, err_mag].

    Returns
    -------
    Dict[str, telescopes.Telescope]
        Dictionary of pyLIMA Telescope objects keyed by band.
    """
    dict_tels = {}
    for band in rubin_ts:
        dict_tels[band] = telescopes.Telescope(
            name=band,
            camera_filter=band,
            location="Earth",
            lightcurve=rubin_ts[band],
            lightcurve_names=["time", "mag", "err_mag"],
            lightcurve_units=["JD", "mag", "mag"],
        )
    return dict_tels


def create_rubin_event(name: str, ra: float, dec: float, ts_dict: Dict[str, np.ndarray]) -> event.Event:
    """
    Create a pyLIMA Event for Rubin data.

    Parameters
    ----------
    name : str
        Event name.
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    ts_dict : Dict[str, np.ndarray]
        Dictionary of timestamps per band.

    Returns
    -------
    pyLIMA.event.Event
        Configured event with telescopes.
    """
    my_event = event.Event(ra=ra, dec=dec)
    my_event.name = name
    lsst_filterlist = "ugrizy"
    rubin_ts = {
        band: np.column_stack((ts_dict[band], np.ones(len(ts_dict[band])) * 20, np.ones(len(ts_dict[band])) * 20)).astype(float)
        for band in lsst_filterlist
        if band in ts_dict
    }
    for band in lsst_filterlist:
        if band in ts_dict:
            my_event.telescopes.append(create_rubin_telescopes(rubin_ts)[band])
    return my_event


def simulate_lightcurve(
    event_id: int,
    event_params: Dict[str, float],
    pylima_event: event.Event,
    model: str,
    parallax: bool,
    g: float = 0,
) -> Tuple["pyLIMA.models.Model", Dict]:
    """
    Simulate a microlensing light curve using pyLIMA.

    Parameters
    ----------
    event_id : int
        Event identifier (used for seeding).
    event_params : Dict[str, float]
        Parameters including magnitudes and microlensing parameters.
    pylima_event : pyLIMA.event.Event
        pyLIMA event object.
    model : str
        Model type ('USBL', 'FSPL', 'PSPL').
    parallax : bool
        Include parallax effects.
    g : float, optional
        Blending fraction (default: 0).

    Returns
    -------
    Tuple[pyLIMA.models.Model, Dict]
        pyLIMA model and parameters.

    Raises
    ------
    ValueError
        If model is unknown.
    """
    magstar = {band: event_params[band] for band in ZP if band in event_params}
    new_event = copy.deepcopy(pylima_event)
    np.random.seed(event_id)
    t0 = event_params["t0"]
    tE = event_params["tE"]

    if model == "USBL":
        params = {
            "t0": event_params["t0"],
            "u0": event_params["u0"],
            "tE": event_params["tE"],
            "rho": event_params["rho"],
            "s": event_params["s"],
            "q": event_params["q"],
            "alpha": event_params["alpha"],
            "piEN": event_params["piEN"],
            "piEE": event_params["piEE"],
        }
        choice = np.random.choice(["central_caustic", "second_caustic", "third_caustic"])
        my_model = USBL_model.USBLmodel(
            new_event,
            origin=[choice, [0, 0]],
            blend_flux_parameter="ftotal",
            parallax=["Full", t0] if parallax else ["None", 0.0],
        )
    elif model == "FSPL":
        params = {
            "t0": event_params["t0"],
            "u0": event_params["u0"],
            "tE": event_params["tE"],
            "rho": event_params["rho"],
            "piEN": event_params["piEN"],
            "piEE": event_params["piEE"],
        }
        my_model = FSPLarge_model.FSPLargemodel(
            new_event, parallax=["Full", t0] if parallax else ["None", 0.0]
        )
    elif model == "PSPL":
        params = {
            "t0": event_params["t0"],
            "u0": event_params["u0"],
            "tE": event_params["tE"],
            "piEN": event_params["piEN"],
            "piEE": event_params["piEE"],
        }
        my_model = PSPL_model.PSPLmodel(
            new_event, parallax=["Full", t0] if parallax else ["None", 0.0]
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    my_parameters = list(params.values())
    my_flux_parameters = []
    np.random.seed(event_id)
    for telescope in new_event.telescopes:
        band = telescope.name
        flux_baseline = 10 ** ((ZP[band] - magstar[band]) / 2.5)
        f_source = flux_baseline / (1 + g)
        f_total = f_source * (1 + g)
        my_flux_parameters.extend(
            [f_source, f_total] if my_model.blend_flux_parameter == "ftotal" else [f_source, f_source * g]
        )

    my_parameters += my_flux_parameters
    pyLIMA_params = my_model.compute_pyLIMA_parameters(my_parameters)
    simulator.simulate_lightcurve(my_model, pyLIMA_params, add_noise=False)
    return my_model, pyLIMA_params


def flux_to_mag(zp: float, flux: np.ndarray) -> np.ndarray:
    """
    Convert flux to magnitude.

    Parameters
    ----------
    zp : float
        Zero point for the band.
    flux : np.ndarray
        Flux values.

    Returns
    -------
    np.ndarray
        Magnitude values.
    """
    return zp - 2.5 * np.log10(np.abs(flux))


def compute_model_lightcurves(my_model: "pyLIMA.models.Model", pyLIMA_parameters: Dict) -> "pyLIMA.models.Model":
    """
    Compute light curves (magnification, flux, magnitude) for the model.

    Parameters
    ----------
    my_model : pyLIMA.models.Model
        Fitted pyLIMA model.
    pyLIMA_parameters : Dict
        Model parameters.

    Returns
    -------
    pyLIMA.models.Model
        Model with updated light curves.
    """
    for telescope in my_model.event.telescopes:
        magnification = my_model.model_magnification(telescope, pyLIMA_parameters)
        model_flux = my_model.compute_the_microlensing_model(telescope, pyLIMA_parameters)["photometry"]
        telescope.lightcurve["magnification"] = magnification
        telescope.lightcurve["flux"] = model_flux
        telescope.lightcurve["mag"] = flux_to_mag(ZP[telescope.name], model_flux)
    return my_model


def simulate_ideal_microlensing_event(
    name: str,
    event_id: int,
    ra: float,
    dec: float,
    model: str,
    event_params: Dict[str, float],
    epochs: Dict[str, np.ndarray],
    parallax: bool,
    g: float = 0,
) -> Tuple["pyLIMA.models.Model", Dict]:
    """
    Simulate an ideal microlensing event using pyLIMA for Rubin data.

    Parameters
    ----------
    name : str
        Event name.
    event_id : int
        Event identifier.
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    model : str
        Model type ('USBL', 'FSPL', 'PSPL').
    event_params : Dict[str, float]
        Event parameters including magnitudes.
    epochs : Dict[str, np.ndarray]
        Epochs per band.
    parallax : bool
        Include parallax effects.
    g : float, optional
        Blending fraction (default: 0).

    Returns
    -------
    Tuple[pyLIMA.models.Model, Dict]
        pyLIMA model with light curves and parameters.
    """
    pylima_event = create_rubin_event(name, ra, dec, epochs)
    pylima_model, pyLIMA_parameters = simulate_lightcurve(event_id, event_params, pylima_event, model, parallax, g=g)
    pylima_model = compute_model_lightcurves(pylima_model, pyLIMA_parameters)
    return pylima_model, pyLIMA_parameters


def pylima_parameters(
    model: str,
    system_type: str,
    bands: List[str],
    data_SOURCE: Dict[str, float],
    data_Genulens: Dict[str, float],
    t_0_limits: List[float],
    blend: Optional[Union[bool, float]] = None,
    event_id: int = 0,
) -> Dict[str, float]:
    """
    Generate pyLIMA-compatible microlensing parameters with blending.

    Parameters
    ----------
    model : str
        Microlensing model ('USBL', 'FSPL', 'PSPL').
    system_type : str
        System type ('Planets_systems', 'Binary_stars', 'BH', 'FFP', 'custom').
    bands : List[str]
        Photometric bands.
    data_SOURCE : Dict[str, float]
        Source data including magnitudes and 'logL', 'logTe'.
    data_Genulens : Dict[str, float]
        Genulens data including 'D_L', 'D_S', 'mu_rel'.
    t_0_limits : List[float]
        Range for t0 [min, max] in JD.
    blend : Union[bool, float], optional
        Blending fraction or True for random blending (default: None).
    event_id : int, optional
        Event identifier for seeding (default: 0).

    Returns
    -------
    Dict[str, float]
        Event parameters including magnitudes and blending.
    """
    np.random.seed(event_id)
    params = generate_event_params(event_id, data_SOURCE, data_Genulens, system_type, t_0_limits)
    mag_baseline = {k.lower(): v for k, v in data_SOURCE.items() if k.lower() in bands}
    if blend is not None:
        params["f_blend"] = np.random.uniform(0, 0.99) if blend is True else blend
    return {**mag_baseline, **params}


def pacz_parameters(
    data_SOURCE: Dict[str, float], bands: List[str], t_0_limits: List[float], event_id: int
) -> Dict[str, float]:
    """
    Generate Paczynski model parameters.

    Parameters
    ----------
    data_SOURCE : Dict[str, float]
        Source data including magnitudes.
    bands : List[str]
        Photometric bands.
    t_0_limits : List[float]
        Range for t0 [min, max] in JD.
    event_id : int
        Event identifier for seeding.

    Returns
    -------
    Dict[str, float]
        Paczynski model parameters.
    """
    np.random.seed(event_id)
    params = {
        "t0": np.random.uniform(*t_0_limits),
        "tE": np.random.uniform(1, 360),
        "u0": np.random.uniform(0.1, 1),
    }
    mag_baseline = {k.lower(): v for k, v in data_SOURCE.items() if k.lower() in bands}
    return {**mag_baseline, **params}


def simulate_pacz_lightcurve(params: Dict[str, float], epochs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Simulate light curves using the Paczynski model.

    Parameters
    ----------
    params : Dict[str, float]
        Paczynski model parameters including magnitudes.
    epochs : Dict[str, np.ndarray]
        Epochs per band.

    Returns
    -------
    Dict[str, np.ndarray]
        Magnitudes per band.
    """
    mags = {}
    for band in epochs:
        u_t = np.sqrt(params["u0"] ** 2 + ((epochs[band] - params["t0"]) / params["tE"]) ** 2)
        A_t = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
        mags[band] = params[band] - 2.5 * np.log10(A_t)
    return mags


@contextmanager
def suppress_stdout():
    """
    Context manager to suppress stdout during pyLIMA operations.

    Yields
    ------
    None
        Suppresses stdout within the context.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def process_chunk(
    args: Tuple[int, Dict[str, np.ndarray]],
    params: Dict[str, float],
    event_id: int,
    ra: float,
    dec: float,
    model: str,
    parallax: bool,
) -> Tuple[int, Dict[str, np.ndarray], List[str]]:
    """
    Process a single chunk of epochs for parallel light curve simulation.

    Parameters
    ----------
    args : Tuple[int, Dict[str, np.ndarray]]
        Tuple of (chunk_idx, epochs_chunk).
    params : Dict[str, float]
        Microlensing parameters.
    event_id : int
        Event identifier.
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    model : str
        Microlensing model.
    parallax : bool
        Include parallax effects.

    Returns
    -------
    Tuple[int, Dict[str, np.ndarray], List[str]]
        Chunk index, magnitudes, and warnings.
    """
    chunk_idx, epochs_chunk = args
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with suppress_stdout():
                pylima_model, pyLIMA_parameters = simulate_ideal_microlensing_event(
                    str(event_id), event_id, ra, dec, model, params, epochs_chunk, parallax
                )
            magnitudes = {
                telescope.filter: telescope.lightcurve["mag"].value
                for telescope in pylima_model.event.telescopes
            }
            magnification = {
                telescope.filter: telescope.lightcurve["magnification"].value
                for telescope in pylima_model.event.telescopes
            }
        return chunk_idx, magnitudes, magnification, pyLIMA_parameters, [str(warn.message) for warn in w]
    except Exception as e:
        print(f"Error in chunk {chunk_idx} for event_id {event_id}: {str(e)}")
        return chunk_idx, None, None, None, [str(e)]


def simulate_pylima_lightcurve(
    epochs: Dict[str, np.ndarray],
    params: Dict[str, float],
    event_id: int,
    ra: float,
    dec: float,
    model: str,
    parallax: bool = True,
    num_chunks: int = 1,
    processes: int = 4,
    shut_down_tqdm: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict, List[str]]:
    """
    Simulate a microlensing light curve using pyLIMA with optional parallel processing.

    Parameters
    ----------
    epochs : Dict[str, np.ndarray]
        Time points for each band.
    params : Dict[str, float]
        Microlensing parameters.
    event_id : int
        Unique event identifier.
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    model : str
        Microlensing model ('USBL', 'FSPL', 'PSPL').
    parallax : bool, optional
        Include parallax effects (default: True).
    num_chunks : int, optional
        Number of chunks for parallel processing (default: 1).
    processes : int, optional
        Number of parallel processes (default: 4).
    shut_down_tqdm : bool, optional
        Disable tqdm progress bar (default: False).

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict, List[str]]
        Magnitudes, pyLIMA parameters, and warnings.

    Notes
    -----
    If num_chunks <= 1, runs serially; otherwise, uses multiprocessing.
    """
    np.random.seed(event_id)
    g = params.get("f_blend", 0)

    def split_epochs(epochs: Dict[str, np.ndarray], n: int) -> List[Dict[str, np.ndarray]]:
        """Split epochs into chunks for parallel processing."""
        chunks = []
        for i in range(n):
            chunk = {}
            for band, times in epochs.items():
                start = i * len(times) // n
                end = (i + 1) * len(times) // n
                if start < len(times):
                    chunk[band] = times[start:end]
            if chunk:
                chunks.append(chunk)
        return chunks

    if num_chunks <= 1:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with suppress_stdout():
                pylima_model, pyLIMA_parameters = simulate_ideal_microlensing_event(
                    str(event_id), event_id, ra, dec, model, params, epochs, parallax, g=g
                )
            magnitudes = {
                telescope.filter: telescope.lightcurve["mag"].value
                for telescope in pylima_model.event.telescopes
            }
            magnification = {
                telescope.filter: telescope.lightcurve["magnification"].value
                for telescope in pylima_model.event.telescopes
            }
        return magnitudes, magnification, pyLIMA_parameters, [str(warn.message) for warn in w]
    else:
        epoch_chunks = split_epochs(epochs, num_chunks)
        if not epoch_chunks:
            print(f"Warning: No valid chunks created for event_id {event_id}")
            return {}, {}, [], []

        worker = partial(
            process_chunk,
            params=params,
            event_id=event_id,
            ra=ra,
            dec=dec,
            model=model,
            parallax=parallax,
        )

        all_mags = {band: [] for band in epochs}
        all_magnifications = {band: [] for band in epochs}
        all_warnings = []
        pyLIMA_parameters = None

        with Pool(processes=processes) as pool:
            results = list(
                tqdm(
                    pool.imap(worker, enumerate(epoch_chunks)),
                    total=len(epoch_chunks),
                    desc=f"Simulating lightcurve for event_id {event_id}",
                    disable=shut_down_tqdm,
                )
            )

        for chunk_idx, mags, mags_m, params, warnings_list in sorted(results, key=lambda x: x[0]):
            if mags is None:
                print(f"Chunk {chunk_idx} failed for event_id {event_id}")
                continue
            for band in mags:
                all_mags[band].append(mags[band])
                all_magnifications[band].append(mags_m[band])
            all_warnings.extend(warnings_list)
            if pyLIMA_parameters is None:
                pyLIMA_parameters = params

        final_mags = {band: np.concatenate(all_mags[band]) if all_mags[band] else np.array([]) for band in all_mags}
        final_magnifications = {
            band: np.concatenate(all_magnifications[band]) if all_magnifications[band] else np.array([])
            for band in all_magnifications
        }
        return final_mags, final_magnifications, pyLIMA_parameters, all_warnings