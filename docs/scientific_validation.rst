Scientific photometric validation
=================================

Scientific validation is separate from software execution and Rubin task
integration.  A successful ``PROCESS`` run proves that the configured tasks
executed; it does not prove that the injected flux or its uncertainty is
scientifically calibrated.

The controlled validation path uses a small, reproducible source list.  Each
source has a sky position, instantaneous magnitude, and seed.  The image's
actual ``PhotoCalib`` converts the requested magnitude to the expected
instrumental flux:

.. code-block:: text

   known source truth
          |
          v
      image PhotoCalib
          |
          v
   expected instrumental flux
          |
          v
      VisitInjectTask
          |
          +------> local image-difference diagnostic
          |
          v
    in-memory injected exposure
          |
          v
    ForcedMeasurementTask
          |
          v
    recovered post-injection flux
          |
          v
    baseline-subtracted residual / pull / flags

The validation measures the known position before and after injection.  This
matters for real Rubin images because the pre-injection scene may already
contain astrophysical flux.  The primary recovered injected flux is therefore

.. math::

   F_{\rm recovered} = F_{\rm post} - F_{\rm pre}.

The difference uncertainty is currently reported as the quadrature sum of the
pre- and post-injection forced-measurement uncertainties.  The validation
product also retains the individual fluxes, uncertainties, native flag values,
pixel position, edge distance, local image/variance summaries, and a bounded
image-difference diagnostic.  The local image sum is a diagnostic for sign,
placement, and image representation; the forced PSF measurement is the primary
photometric comparison.

Controlled validation products
------------------------------

``ulens_lsst.scientific_validation`` provides the controlled-source definition,
PhotoCalib conversion, residual calculations, and a fixed Parquet schema.
``ulens_lsst.validation_analysis`` writes descriptive JSON summaries and
diagnostic plots.  These products are separate from the production
``photometry/`` dataset and can be inspected without rerunning Rubin tasks.

The current schema includes:

* ``requested_mag`` and ``expected_inst_flux`` in the image's instrumental
  flux units;
* pre- and post-injection forced fluxes and errors;
* baseline-subtracted recovered flux, residual, fractional residual, and pull;
* pixel coordinates and distance to the image edge;
* local image-difference sum and peak;
* compact native flag diagnostics and measurement-status fields;
* image-condition summaries for local background/variance and PSF shape when
  available.

The first validation samples are deliberately isolated, interior sources at
bright, moderate, and faint forced-photometry magnitudes.  They characterize
behavior rather than imposing arbitrary thresholds.  Boundary, masking,
crowding, saturation, and broader image-condition studies are subsequent
validation cases.

DP0.2 and DP1 are run through the same release-independent validation code;
``get_data_release`` and ``DataReleaseConfig`` provide all concrete Butler
access.  DP2 is structurally recognized but awaits an individual-image product.

This is not a Rubin DIA/alert detection test and is not a microlensing
completeness or detection-efficiency measurement.  Formal recovery and
selection-function analyses require additional criteria and validation.
