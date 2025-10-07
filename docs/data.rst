.. _data:

Data Setup and Outputs
======================

The `ulens_lsst` package relies on external datasets for specific configurations and generates simulation outputs during pipeline execution. This section details the required input datasets, their download and configuration, and the structure of the generated output datasets.

Required Input Datasets
-----------------------

For certain pipeline configurations (e.g., when `sources_catalog="TRILEGAL"` in `config.yaml`), external datasets such as `chunks_TRILEGAL_Genulens` are required to simulate TRILEGAL sources.

Downloading and Configuring Datasets
------------------------------------

1. **TRILEGAL Datasets**:
   - **Source**: Download the `chunks_TRILEGAL_Genulens` datasets from the project repository (`data/chunks_TRILEGAL_Genulens/`) or a designated external source (e.g., Zenodo DOI to be provided).
   - **Placement**: Store the datasets in the `ulens_lsst/data/chunks_TRILEGAL_Genulens/` directory within your project.
   - **Alternative**: Specify a custom CSV file path if using a different source catalog.

2. **Genulens Datasets**: Microlensing event parameters.
   - **Source**: Download the `chunks_TRILEGAL_Genulens` datasets from the project repository (`data/chunks_TRILEGAL_Genulens/`).
   - **Placement**: Store the datasets in the `ulens_lsst/data/chunks_TRILEGAL_Genulens/` directory within your project.
   - **Alternative**: There is a plan to add a custom option in future version.

3. **Configuration**:
   - Copy the example configuration file:
     .. code-block:: bash

        cp ulens_lsst/config/config_example.yaml config.yaml

   - Update `config.yaml` with the appropriate settings:
     .. code-block:: yaml

        sources_catalog: "TRILEGAL"
        TRILEGAL_Genulens_path: "data/chunks_TRILEGAL_Genulens/"

   - For a custom CSV file:
     .. code-block:: yaml

        sources_catalog: "path/to/custom.csv"

**Note**: These input datasets are not bundled with the package to maintain a lightweight installation. Users must download them separately or generate them using the pipeline.

Generated Simulation Outputs
----------------------------

When executing the `SimPipeline` with a configured `config.yaml`, the pipeline creates an output directory named after the `name` parameter (under `main_path`, defaulting to `/runs/`). This directory contains temporary files and final simulation outputs, which are stored as Parquet files.

Dataset Structure
-----------------

The generated datasets are organized into three main Parquet files, each serving a specific purpose in the simulation pipeline. Below are detailed descriptions of their columns:

Photometry Datasets
^^^^^^^^^^^^^^^^^^^

- **photometry_*.parquet**:
  - Contains ideal simulated light curves before injection into LSST-like data.
  - **Columns**:
    - `event_id`: Unique identifier for the microlensing event (int32).
    - `time`: Observation time in Modified Julian Date (MJD) (float64).
    - `band`: Filter band (e.g., 'u', 'g', 'r', 'i', 'z', 'y') (string).
    - `ideal_mag`: Ideal magnitude without noise (float32).
    - `meas_mag`: Measured magnitude with noise (float32, post-injection if applicable).
    - `meas_mag_err`: Error in measured magnitude (float32).
    - `meas_flux`: Measured flux (float32).
    - `meas_flux_err`: Error in measured flux (float32).
    - `magnification`: Magnification factor applied to the light curve (float32).
    - `injection_flag`: Status of injection (e.g., 'none', 'injected') (string).
    - `measure_flag`: Status of measurement (e.g., 'none', 'measured') (string).

- **calexps-photometry_*.parquet**:
  - Contains LSST-like simulated light curves after injection into calibrated exposures (calexps).
  - **Columns**: Identical to `photometry_*.parquet`, with additional noise and blending effects from the LSST simulation.

Events Dataset
^^^^^^^^^^^^^^

- **data-events_*.parquet**:
  - Stores detailed metadata about simulated microlensing events.
  - **Columns**:
    - `event_id`: Unique identifier (int32).
    - `ra`: Right ascension in degrees (float64).
    - `dec`: Declination in degrees (float64).
    - `model`: Microlensing model type (e.g., 'PSPL' for Point Source Point Lens) (string).
    - `system_type`: System classification (e.g., 'microlensing') (string).
    - `points`: Number of photometry points (int).
    - `logL`: Log luminosity (float).
    - `logTe`: Log effective temperature (float).
    - `D_L`: Lens distance in parsecs (float).
    - `D_S`: Source distance in parsecs (float).
    - `mu_rel`: Relative proper motion in mas/year (float).
    - `nearby_object_ra`: RA of nearby object in degrees (float).
    - `nearby_object_dec`: Dec of nearby object in degrees (float).
    - `nearby_object_objId`: Object ID of nearby object (string).
    - `nearby_object_distance`: Distance to nearby object in arcseconds (float).
    - `cadence_noise`: Source of cadence and noise (e.g., 'DP0', 'rubin_sim') (string).
    - `peak_time`: Peak time of the event in MJD (float).
    - `nearby_object_mag_{band}`: Magnitude of nearby object per band (e.g., `mag_u`, `mag_g`) (float).
    - `nearby_object_fwhm_{band}`: FWHM of nearby object per band (e.g., `fwhm_u`, `fwhm_g`) (float).
    - `param_{key}`: Model parameters (e.g., `param_t0`, `param_u0`) (float).
    - `param-pylima_{key}`: pyLIMA-specific parameters if applicable (e.g., `param-pylima_tE`) (float).

Additional Information 
----------------------

- **Format**: All datasets are stored in Parquet format for efficiency with large datasets.
- **Generation**: Datasets are generated by running the pipeline with appropriate `config.yaml` settings (e.g., `steps: simulate,load_nearby,process_photometry`).
- **Limitations**: Requires the LSST stack (e.g., lsst-scipipe-10.0.0) for full compatibility. Dataset size and version may vary with pipeline updates.

Notebook Investigation
----------------------

A Jupyter notebook, `tutorials/dataset_investigation.ipynb`, explores the generated datasets, including column analysis and visualization. To use it:

- Ensure the notebook is in `docs/tutorials/`.
- Run locally with Jupyter: `jupyter notebook docs/tutorials/dataset_analysis.ipynb`.
- The notebook covers initial data inspection.
