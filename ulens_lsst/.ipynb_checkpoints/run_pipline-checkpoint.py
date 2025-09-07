"""
Command-line script to run the ulens_lsst pipeline.

This script provides a CLI for executing the SimPipeline with optional config file
and step selection. It allows running the full pipeline or specific steps via
arguments, supporting flexible experimentation.

Usage
-----
ulens-lsst [--config CONFIG_FILE] [--steps STEPS]

Arguments
---------
--config : str, optional
    Path to the config.yaml file (default: 'config.yaml').
--steps : str, optional
    Comma-separated steps to run: 'all', 'simulate', 'load_nearby', 'process_photometry', 'chi2' (default: 'all').

Examples
--------
Run full pipeline:
ulens-lsst --config config.yaml

Run specific steps:
ulens-lsst --steps simulate,process_photometry
"""
import argparse
import logging
from ulens_lsst.simulation_pipeline import SimPipeline

def main() -> None:
    """
    Main entry point for the CLI script.
    """
    parser = argparse.ArgumentParser(description="Run the ulens_lsst microlensing simulation pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml (default: 'config.yaml')")
    parser.add_argument("--steps", type=str, default="all", help="Comma-separated steps: all, simulate, load_nearby, process_photometry, chi2 (default: 'all')")
    args = parser.parse_args()

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    sim = SimPipeline(args.config)
    steps = args.steps.split(",") if args.steps != "all" else ["simulate", "load_nearby", "process_photometry", "chi2"]

    if "simulate" in steps:
        logger.info("Running simulate_lightcurves...")
        sim.simulate_lightcurves()

    if "load_nearby" in steps:
        logger.info("Running load_nearby_objects...")
        sim.load_nearby_objects()

    if "process_photometry" in steps:
        logger.info("Running process_synthetic_photometry...")
        sim.process_synthetic_photometry()

    if "chi2" in steps:
        logger.info("Running compute_events_chi2...")
        sim.compute_events_chi2()

    logger.info("Pipeline completed.")

if __name__ == "__main__":
    main()