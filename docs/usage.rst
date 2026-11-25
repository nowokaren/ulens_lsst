.. _usage:

Usage
=====

Command-Line Interface
-----------------------

Run the pipeline via the command-line:

.. code-block:: bash

   ulens-lsst --config config.yaml --steps all

Available steps: `all`, `simulate`, `load_nearby`, `process_photometry`, `chi2`. Example for specific steps:

.. code-block:: bash

   ulens-lsst --config config.yaml --steps simulate,process_photometry

Outputs are saved in a directory named after `name` under `main_path` (e.g., `/runs/my_simulation/`) as specified in `config.yaml`. See :ref:`data` for details.

Programmatic Usage
------------------

For programmatic use:

.. code-block:: python

   from ulens_lsst.simulation_pipeline import SimPipeline
   sim = SimPipeline('config.yaml')
   results = sim.simulate_lightcurves(event_processor='ulens')
   print(results)

**Output Files**:
Simulation outputs (temporary files and results) are saved in a directory named after `name` under `main_path` as specified in `config.yaml` (e.g., `/runs/my_simulation/`). See :ref:`data` for details.

See the :ref:`api` section for detailed class and method documentation.

(Soon) See the :ref:`tutorials` section for interactive examples and the :ref:`api` section for detailed class and method documentation.

Process count and independent Slurm runs
----------------------------------------

The top-level n_processes setting controls the worker count passed to
multiprocessing.Pool in simulate_lightcurves, process_synthetic_photometry, and
compute_events_chi2. It does not distribute one run across nodes and does not
set thread limits for NumPy, BLAS, or LSST libraries.

Set one worker explicitly for a serial run:

.. code-block:: yaml

   main_path: /path/to/scratch/ulens_runs
   name: serial_run_001
   n_processes: 1

With n_processes: 1, each of the three stages creates a
multiprocessing.Pool(processes=1). If n_processes is omitted or null, the
pipeline preserves its previous automatic choice, min(os.cpu_count() or 4, 8).
Values must be positive integers.
The former format_output_params.n_processes entry did not control these
pipeline pools and has been removed from the example configs; use only the
top-level n_processes setting.

To run many independent simulations, use one Slurm array task per run. Give
each task its own name (which determines main_path/name) and config file. The
pipeline's Parquet lock protects concurrent appends within one run; it does not
make two independent runs safe when they share an output directory. Do not
point concurrent jobs at the same main_path/name.

Example array script (adapt the LSST environment setup and scratch path to the
USDF allocation):

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=ulens
   #SBATCH --array=0-15%4
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=1
   #SBATCH --output=slurm-%A_%a.out
   set -euo pipefail

   # Activate the LSST environment as required by the USDF/RSP session.
   # source /opt/lsst/software/stack/loadLSST.bash
   # setup lsst_distrib

   BASE_CONFIG=config/config_file.yaml
   SCRATCH_ROOT=$(printenv SCRATCH || pwd)
   RUNS_ROOT=$SCRATCH_ROOT/ulens_runs
   CONFIG_ROOT=$SCRATCH_ROOT/ulens_configs
   mkdir -p "$RUNS_ROOT" "$CONFIG_ROOT"

   TASK_KEY=$(printf '%s_%s' "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID")
   RUN_NAME=ulens_$TASK_KEY
   RUN_CONFIG=$CONFIG_ROOT/$RUN_NAME.yaml

   python - "$BASE_CONFIG" "$RUN_CONFIG" "$RUN_NAME" "$RUNS_ROOT" <<'PY'
   import sys
   import yaml

   source, destination, run_name, runs_root = sys.argv[1:]
   with open(source) as stream:
       config = yaml.safe_load(stream)
   config["name"] = run_name
   config["main_path"] = runs_root
   config["n_processes"] = 1
   with open(destination, "w") as stream:
       yaml.safe_dump(config, stream, sort_keys=False)
   PY

   ulens_lsst --config "$RUN_CONFIG" --steps all

The Slurm cpus-per-task=1 request allocates one CPU; n_processes: 1 separately
ensures the pipeline creates one multiprocessing worker in each stage. Neither
setting alone guarantees that threaded numerical libraries use only one thread.
If those libraries use threads, control them separately through their documented
environment variables and record those settings with the run.

Image access backends
--------------------

Image retrieval is selected independently from ``data_preview``. Existing
configurations without an ``image_access`` block continue to use Butler. TAP
image access is currently validated for DP1 ``lsst.visit_image`` products:

.. code-block:: yaml

   lsst_images:
     data_preview: dp1
   image_access:
     backend: tap
     auth: rsp
     collection: LSST.DP1
     product_subtype: lsst.visit_image

Use ``auth: rsp`` inside Rubin Science Platform. For an external authenticated
session, use ``auth: token`` and provide ``token_path``; the configuration stores
only the path, never the token. The worker receives the serializable settings and
reconstructs its local TAP service and HTTP session.

Each saved exposure row contains the stable ``obs_publisher_did``, its original
DataLink URL, and the visit/detector/band association. The final ``#this`` URL is
resolved near download time and is not persisted. ``expMidptMJD`` comes from the
matching ``CcdVisit`` row; ``t_min``, ``t_max``, and ``obscore_midpoint_mjd`` are
kept separately from ObsCore. For the checked DP1 product these midpoint values
differed; no cause is inferred from that one observation.

The DP1 TAP service does not support ADQL ``INTERSECTS``. Discovery uses a
0.25-degree center preselection margin, then checks the returned ObsCore polygon
against the requested circle locally. The margin covers the checked DP1 detector
footprint (about 0.23 degrees across); it only broadens the candidate query, while
the local footprint test applies the requested region.

DP1 ``seeing`` is published in arcseconds as the mean measured FWHM of the PSF.
``psfSigma`` is a separate quantity in pixels: the PSF model second-moments
determinant radius at the chip center. The pipeline preserves both fields without
converting one into the other. The legacy ``region_sky.LSSTDataLoader`` remains a
Butler-backed path outside the image-access integration described here.

Concurrency and output limits
-----------------------------

Workers in one pipeline stage append result Parquet rows through the catalog's
lock-file path; the parent combines temporary Parquet files after the pool has
finished. A run using one pool worker therefore has no concurrent worker
writers within that stage. Temporary inputs, results_*.parquet, and the temp
directory are scoped to main_path/name.

Concurrent jobs must use distinct names or otherwise distinct output
directories. Sharing a run directory can make jobs select the same file index,
overwrite the temporary input or calexp catalog, append to the same results
file, or combine and remove each other's temporary files. If a job is killed
forcefully while holding a catalog lock, it may leave a .lock file; only remove
a stale lock after confirming no process still owns that run.
