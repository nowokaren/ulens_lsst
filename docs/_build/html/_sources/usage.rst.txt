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