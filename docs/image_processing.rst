.. _image_processing:

Image processing architecture
=============================

The image-processing campaign is being migrated to a generic image-based
workflow. The release-specific Rubin names are resolved once through the RSP
``DataReleaseConfig`` abstraction; scientific code uses generic concepts such
as an image, an image identifier, and an image context.

The migration path is:

.. code-block:: text

   user/run configuration
       -> get_data_release(...)
       -> resolved DataReleaseConfig
       -> LSSTData image access
       -> ImageContext and image processing

For the new path, resolve the release before constructing ``LSSTData``:

.. code-block:: python

   from data_release_config import get_data_release
   from ulens_lsst.lsst_data import LSSTData

   release_name = config["data_release"]
   release = get_data_release(release_name)
   data = LSSTData(..., release_config=release)
   contexts = data.select_image_contexts(limit=10)
   image = data.load_image(contexts[0])

In the YAML run configuration, ``data_release`` is the selected release name
resolved by RSP and ``image_processor`` selects the image operation:

.. code-block:: yaml

   data_release: "<release name supplied by the RSP environment>"
   image_processor: "injection_forced_photometry"

The placeholder above is intentional. ``ulens_lsst`` does not define a
permanent RAC release or duplicate the RSP release table.

An ``ImageContext`` describes one image without loading its pixels. It records
the deterministic ``image_id``, the resolved dataset type, the Butler data ID,
and lightweight observation metadata such as visit, detector, band, and MJD.
``LSSTData.load_image(context)`` then loads the image for processing.

The older ``data_preview`` constructor path remains temporarily for backwards
compatibility with existing DP0/DP1 notebooks and output folders. It is not a
second target architecture and will not receive new RAC functionality.

The intended campaign flow is:

.. code-block:: text

   PREPARE: resolve the release and select images
       |
       +--> write event truth, image contexts, and work units
       |
   PROCESS: each work unit is independent
       |
       +--> load one image
       +--> select relevant simulated events
       +--> run the selected ImageProcessor
       +--> write scientific product parts and operational status
       |
   FINALIZE: validate completeness, schemas, and duplicate keys

The first processor will perform in-memory injection and forced photometry.
Injected exposures are not persistent products.

RSP integration tests
---------------------

Rubin-dependent tests must run in an initialized Science Pipelines shell. The
repository path may be prepended, but the existing EUPS ``PYTHONPATH`` must be
preserved::

   setup lsst_distrib
   export PYTHONPATH="$PWD:$PYTHONPATH"

The real-data integration test is configured by the environment because
accessible Butler collections vary between RSP deployments. For example::

   export ULENS_LSST_RSP_RELEASE=DP1
   export ULENS_LSST_RSP_COLLECTION=LSSTComCam/DP1
   export ULENS_LSST_RSP_IMAGE_DATASET=visit_image
   pytest -q tests/test_rsp_integration.py

These variables are test inputs, not a release mapping for the RAC path. The
test queries one DatasetRef, loads one exposure, and checks WCS, PSF,
PhotoCalib, mask, and variance availability.

Release validation status
-------------------------

The processor boundary is release-independent. The resolved
``DataReleaseConfig`` supplies the concrete image dataset, collection, and
data-ID convention.

* DP0.2 is real-RSP tested with ``calexp``, injection, and forced photometry.
* DP1 is real-RSP tested with ``visit_image``, injection, and forced
  photometry.
* DP2 resolves as a recognized release, but its current profile advertises
  ``coadd_forced``. Single-image injection is therefore currently unavailable
  for that profile, pending an individual-image product and RSP validation.

This is a current data-capability state, not a permanent architectural
incompatibility. Updating the resolved RSP profile and repeating the same
integration test should be sufficient when individual DP2 images become
available.
