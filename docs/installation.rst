Prerequisites
-------------

The `ulens_lsst` package requires the LSST stack (e.g., lsst-scipipe-10.0.0) for modules like `lsst_data` and `lsst_tools`. Follow the official LSST installation guide: `LSST Stack Installation <https://pipelines.lsst.io/install/lsstinstall.html>`_.

External datasets (e.g., `chunks_TRILEGAL_Genulens`) are required for certain configurations. Download and place them in `ulens_lsst/data/chunks_TRILEGAL_Genulens/`. See :ref:`data` for setup instructions.

Installation
============

Install `ulens_lsst` via pip:

.. code-block:: bash

   pip install ulens_lsst

For development, clone the repository and install locally:

.. code-block:: bash

   git clone https://github.com/nowokaren/ulens_lsst.git
   cd ulens_lsst
   pip install -e .


Configuration
-------------

1. Copy the example configuration file:

   .. code-block:: bash

      cp ulens_lsst/config_file.yaml config.yaml

2. Update `config.yaml` with paths to your datasets (e.g., `TRILEGAL_Genulens_path`).



Optional Dependencies
---------------------

For `sim_type="rubin_sim"`, install `rubin_sim` and Opsim via the LSST stack:

.. code-block:: bash

   conda install -c conda-forge rubin-sim

See the `rubin_sim documentation <https://rubin-sim.lsst.io/>` for details on Opsim versions (e.g., "baseline").

