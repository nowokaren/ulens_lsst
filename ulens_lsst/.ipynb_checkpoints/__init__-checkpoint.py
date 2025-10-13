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
        logger.info("LSST stack detected.")
    except ImportError:
        logger.info("LSST stack NOT detected.")
    

check_lsst_setup()