"""
LSST Microlensing Simulation Pipeline.

This package provides tools for simulating microlensing events using LSST data
(DP0, DP1, or rubin_sim), designed for versatile experimentation.
"""
__version__ = "1.0.0"
from .catalogs_utils import Catalog


def __getattr__(name):
    """Load the event-simulation pipeline only when it is requested."""
    if name == "SimPipeline":
        from .simulation_pipeline import SimPipeline

        return SimPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

import logging

logger = logging.getLogger(__name__)

def check_lsst_setup():
    try:
        import lsst.afw
        logger.info("LSST stack detected.")
    except ImportError:
        logger.info("LSST stack NOT detected.")
    

check_lsst_setup()
