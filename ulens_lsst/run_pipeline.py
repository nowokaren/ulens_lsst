"""
Command-line script to run the ulens_lsst pipeline.

This script provides a CLI for executing the SimPipeline with optional config file,
step selection, and parameters for simulation customization. It supports running the
full pipeline or specific steps, with parameters like RA, Dec, model, system type,
and initial event ID for flexible microlensing experiments.

Usage
-----
ulens-lsst [--config CONFIG_FILE] [--steps STEPS] [--sim_name SIM_NAME] [--ra RA] [--dec DEC]
           [--init_event_id INIT_EVENT_ID] [--model MODEL] [--system_type SYSTEM_TYPE]

Arguments
---------
--config : str, optional
    Path to the config.yaml file (default: 'config.yaml').
--steps : str, optional
    Comma-separated steps to run: 'all', 'simulate', 'load_nearby', 'process_photometry', 'chi2' (default: 'all').
--sim_name : str, optional
    Name of the simulation.
--ra : float, optional
    Sky center RA in degrees.
--dec : float, optional
    Sky center Dec in degrees.
--init_event_id : int, optional
    Initial event ID for the simulation (default: None, uses config value).
--model : str, optional
    Microlensing model ('Pacz', 'USBL', 'FSPL', 'PSPL') (default: None, uses config value).
--system_type : str, optional
    System type (e.g., 'Planets_systems', 'FFP', 'BH', 'Binary_stars') (default: None, uses config value).
--resume : bool, optional
    Resume from existing folder (default: False).

Examples
--------
Run full pipeline:
    ulens-lsst --config config.yaml --sim_name dataset_dp0_0 --ra 180.0 --dec 0.0 --init_event_id 0 --model USBL --system_type Planets_systems
Run specific steps:
    ulens-lsst --steps simulate,process_photometry --sim_name dataset_dp0_1 --ra 185.0 --dec 5.0
"""
import argparse
import logging
import yaml
import shutil
import os
from ulens_lsst.simulation_pipeline import SimPipeline

def main() -> None:
    """
    Main entry point for the CLI script.
    """
    parser = argparse.ArgumentParser(description="Run the ulens_lsst microlensing simulation pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml (default: 'config.yaml')")
    parser.add_argument("--folder", type=str, default="test", help="Name of pre-existing simulation")
    parser.add_argument("--steps", type=str, default="all", help="Comma-separated steps: all, simulate, load_nearby, process_photometry, chi2 (default: 'all')")
    parser.add_argument("--resume", action="store_true", help="Resume from existing folder")
    parser.add_argument("--ra", type=float, default=None, help="Sky center RA in degrees")
    parser.add_argument("--dec", type=float, default=None, help="Sky center Dec in degrees")
    parser.add_argument("--radius", type=float, default=None, help="Sky radius in degrees")
    parser.add_argument("--n_events", type=int, default=None, help="Number of events to simulate")
    parser.add_argument("--sim_name", type=str, default=None, help="Simulation name")
    parser.add_argument("--init_event_id", type=int, default=None, help="Initial event ID for the simulation")
    parser.add_argument("--model", type=str, default=None, help="Microlensing model (Pacz, USBL, FSPL, PSPL)")
    parser.add_argument("--system_type", type=str, default=None, help="System type (e.g., Planets_systems, FFP, BH, Binary_stars)")
    args = parser.parse_args()

    # Set up logging
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    # Initialize SimPipeline
    if not args.resume:
        sim = SimPipeline(args.config, setup_dir=False)
        modified = False
        if args.sim_name is not None:
            sim.name = args.sim_name
            modified = True
        if args.n_events is not None:
            sim.n_events = args.n_events
            modified = True
        if args.radius is not None:
            sim.sky_radius = args.radius
            modified = True
        if args.ra is not None:
            sim.sky_center["coord"] = [args.ra, args.dec]
            modified = True
        if args.init_event_id is not None:
            sim.init_event_id = args.init_event_id
            modified = True
        if args.model is not None:
            sim.model = args.model
            modified = True
        if args.system_type is not None:
            sim.system_type = args.system_type
            modified = True
        sim._setup_directories()
        config_copy = os.path.join(sim.output_dir, "config_file.yaml")
        shutil.copy2(args.config, config_copy)
        if modified:
            # Update config file with modified parameters
            with open(config_copy, 'r') as f:
                config_data = yaml.safe_load(f)
            if args.sim_name is not None:
                config_data['name'] = args.sim_name
            if args.n_events is not None:
                config_data['n_events'] = args.n_events
            if args.radius is not None:
                config_data['sky_radius'] = args.radius
            if args.ra is not None:
                config_data['sky_center']['coord'] = [args.ra, args.dec]
            if args.init_event_id is not None:
                config_data['init_event_id'] = args.init_event_id
            if args.model is not None:
                config_data['model'] = args.model
            if args.system_type is not None:
                config_data['system_type'] = args.system_type
            with open(config_copy, 'w') as f:
                yaml.safe_dump(config_data, f)
    else:
        sim = SimPipeline(from_folder=args.folder, new=False)
    print(sim)

    # Run specified steps
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