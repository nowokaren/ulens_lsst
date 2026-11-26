# Changelog

## Unreleased

This section records work on `feature/tap-data-access-v1.0.3`. It is not the
`v1.0.3` release; no `v1.0.3` tag or GitHub Release exists yet.

### Added

- Select image access through Butler or TAP/DataLink for DP1
  `lsst.visit_image` products.
- Add serializable `ImageProduct` descriptors and the
  `ButlerImageAccess`, `TapImageAccess`, and `make_image_access(...)` APIs.
- Add authenticated TAP/DataLink discovery, product resolution, and FITS
  downloads with a local cache, per-product lock, and atomic file replacement.
- Preserve image provenance and the separate ObsCore time bounds/midpoint and
  `CcdVisit.expMidptMJD` values.

### Changed

- Integrate image descriptors and backend loading with `LSSTData`, `Calexp`,
  photometry workers, plots, and analysis. Workers reconstruct access locally.
- Allow `Calexp` to wrap an `ExposureF` or local FITS path without constructing
  a Butler unnecessarily; support Butler `DataCoordinate` mappings.

### Fixed

- Report a clear import error when image processing is attempted without the
  LSST Science Pipelines, including instructions to activate `lsst_distrib`.

### Validation

- Add offline unit tests for descriptors, Butler/TAP access, DataLink handling,
  caching, configuration, workers, and `Calexp` loading.
- Validate one Butler image and one TAP `lsst.visit_image` through the integrated
  access path in RSP.

## 1.0.2

### Added

- Add a top-level `n_processes` setting controlling the multiprocessing pools
  used by light-curve simulation, synthetic photometry, and chi-squared stages.
- Support explicitly serial runs with `n_processes: 1`; when unset or `null`,
  retain the previous automatic worker selection.
- Include v1.0.1's nearby-object count within the configured angular radius,
  plot improvements, and DP0.2 pipeline tutorial in this cumulative release.

### Changed

- Validate `n_processes` as a positive integer and report invalid values clearly.
- Propagate each run's configured name and main output path to worker-side data
  operations so independent runs use their own output directories.

### Documentation

- Document serial execution, Slurm arrays, unique output directories, worker
  scope, and the distinction between processes and library threads.

### Validation

- Add tests for process-count validation and propagation, output isolation, and
  existing pipeline behavior.
- The v1.0.2 diff changes configuration, orchestration, validation, tests, and
  documentation; it does not intentionally change scientific models, source
  injection, or photometry calculations.

## 1.0.1

### Added

- Count catalog objects within the configured radius of each injected event and
  persist the count and radius with event results.
- Add a DP0.2 pipeline tutorial.

### Changed

- Improve event light-curve plots and the event field-of-view plots, including
  nearby-object display and related plot options.

## 1.0.0

### Added

- Initial tagged pipeline release: microlensing simulation, light-curve and
  event processing, LSST exposure injection/photometry, Parquet outputs, CLI,
  configuration examples, and tutorials.

## Comparisons

- [v1.0.1...v1.0.2](https://github.com/nowokaren/ulens_lsst/compare/v1.0.1...v1.0.2)
- [v1.0.2...feature/tap-data-access-v1.0.3](https://github.com/nowokaren/ulens_lsst/compare/v1.0.2...feature/tap-data-access-v1.0.3)
