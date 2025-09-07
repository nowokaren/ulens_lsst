.. _data:

Data Setup
==========

The `ulens_lsst` package requires external datasets for certain configurations, such as `chunks_TRILEGAL_Genulens` for TRILEGAL sources when `sources_catalog="TRILEGAL"` in `config.yaml`.

Downloading Datasets
--------------------

1. **TRILEGAL Datasets**:
   - Download the `chunks_TRILEGAL_Genulens` datasets from `data/chunks_TRILEGAL_Genulens/` or your preferred source.
   - Place the datasets in `ulens_lsst/data/chunks_TRILEGAL_Genulens/` within your project directory.

2. **Configure `config.yaml`**:
    - Copy the example configuration:

     .. code-block:: bash

        cp ulens_lsst/config/config_example.yaml config.yaml

   - Update the `sources_catalog` and dataset path in `config.yaml`:

     .. code-block:: yaml

        sources_catalog: "TRILEGAL"
        TRILEGAL_Genulens_path: "data/chunks_TRILEGAL_Genulens/"

   - Alternatively, specify a custom CSV path:

     .. code-block:: yaml

        sources_catalog: "path/to/custom.csv"

3. **Optimization**:
   - The pipeline loads TRILEGAL chunks selectively using `pandas.read_csv` with `skiprows` and `nrows` to minimize memory usage.
   - Ensure sufficient disk space for the datasets (e.g., several GB for TRILEGAL chunks).

Note: Datasets are not bundled with the package to keep the installation lightweight. Users must download them separately.


Simulation Outputs
--------------------

When running `SimPipeline(config.yaml)`, the pipeline creates a directory named after `name` (from `config.yaml`) under `main_path` (default: `/runs`). This directory stores:

- **Temporary Files**: Parquet files (e.g., `temp_photometry_*.parquet`, `temp_data-events_*.parquet`) in a `temp/` subdirectory.
- **Results**: Final outputs like `photometry_*.parquet`, `data-events_*.parquet`, and `results_*.csv` in the simulation directory.

Example `config.yaml` snippet for output configuration:

.. code-block:: yaml

   main_path: "/runs"
   name: "my_simulation"

This creates `/runs/my_simulation/` with `temp/` and result files. Ensure `main_path` is writable and has sufficient disk space.
