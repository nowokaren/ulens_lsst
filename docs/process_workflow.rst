Independent image-processing workflow
=====================================

The release-aware production workflow has three operational stages:

.. code-block:: text

   PREPARE: select images and sources
       |-- images.parquet + work_units.parquet
       v
   PROCESS: one deterministic work unit
       |-- load image -> select sources
       |-- ImageProcessor: in-memory injection and measurement
       |-- atomic photometry part + operational status
       v
   STATUS: inspect durable state
       |-- missing / failed / stale / invalid -> restart plan -> PROCESS
       |-- completed / empty-success -> FINALIZE
       v
   FINALIZE: validate completeness and scientific keys

PREPARE creates the scientific image catalog and deterministic execution
groups. A scientific image is the unit passed to an ``ImageProcessor``. An
execution work unit contains one or more image IDs so that grouping can be
benchmarked without changing scientific identities.

The concrete Rubin repository, collection, dataset type, TAP table, columns,
and data-ID conventions are resolved through ``DataReleaseConfig``. The work
unit itself contains generic image IDs and does not know whether an image is a
``calexp``, ``visit_image``, or another release product.

PROCESS commands
----------------

Process one work unit independently::

   ulens_lsst process RUN_DIR WORK_UNIT_ID

The equivalent Python API is::

   from ulens_lsst.image_processing_run import process_work_unit
   process_work_unit("RUN_DIR", "wu-...")

The operation loads the run configuration, resolves the release, reads only
the selected work unit, selects source rows for each image, loads images, runs
the configured ``ImageProcessor``, and writes only that work unit's products.
It does not know whether it is called from a serial shell, multiprocessing,
SLURM, a SLURM array, or another scheduler.

For ``injection_forced_photometry``, injection and forced measurement happen
on an in-memory image. The modified image is discarded after compact
scientific rows are produced. Injection is not a persisted pipeline stage.

Persistent products
-------------------

Each work unit owns its own scientific part::

   RUN/
       config_file.yaml
       events/
           *.parquet
       images.parquet
       work_units.parquet
       photometry/
           part-<work-unit-id>.parquet
       status/
           part-<work-unit-id>.json

Workers never append to a shared Parquet file. A part is written to a temporary
file, read back for basic validation, and atomically renamed to its deterministic
final path. A killed worker can therefore leave only a recognizable temporary
file, not a scientific part that appears complete.

Photometry rows represent eligible event/source--image attempts. Scientific
failures remain rows with explicit injection or measurement status where the
processor has enough information to provide them. A work-unit operational
failure is recorded separately in ``status/`` and is not the same as a source
being scientifically unrecovered. The logical scientific key is
``(event_id, image_id)`` for the current processor.

STATUS and restart
------------------

Inspect durable operational state::

   ulens_lsst status RUN_DIR

The summary distinguishes pending, running, completed, failed, stale, invalid,
and empty-success work. A running record older than the stale threshold is
classified as stale, so a dead process cannot remain running forever.

Generate deterministic work units for external scheduler submission::

   ulens_lsst restart-plan RUN_DIR

The restart plan includes missing, failed, stale, and invalid work units. It
excludes valid completed and empty-success work. Re-running a completed work
unit skips it by default; an explicit ``--force`` option is available when a
deliberate replacement is required::

   ulens_lsst process RUN_DIR WORK_UNIT_ID --force

No scheduler job ID, worker number, process count, retry number, or execution
order contributes to scientific or work-unit identity.

FINALIZE
--------

Validate a campaign without merging all Parquet parts::

   ulens_lsst finalize RUN_DIR

FINALIZE checks that every expected work unit is completed or empty-success,
that deterministic parts are readable and contain required identity columns,
and that scientific keys are unique across parts. It reports incomplete work
and duplicate keys without creating a giant merged photometry file.
