"""
LSST Microlensing Simulation Pipeline.

This package provides tools for simulating microlensing events using LSST data
(DP0, DP1, or rubin_sim), designed for versatile experimentation.
"""
__version__ = "1.0.0"
from .catalogs_utils import Catalog
from .simulation_pipeline import SimPipeline

import logging

logger = logging.getLogger(__name__)

def check_lsst_setup():
    try:
        import lsst.afw
    except ImportError:
        raise ImportError(
            "LSST stack not found. Run 'setup lsst_distrib' first. "
            "See https://ulens-lsst.readthedocs.io/en/latest/installation.html"
        )
    logger.info("LSST stack detected.")

check_lsst_setup()