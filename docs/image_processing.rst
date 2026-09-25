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

   release = get_data_release("DP0.2")
   data = LSSTData(..., release_config=release)
   contexts = data.select_image_contexts(limit=10)
   image = data.load_image(contexts[0])

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
