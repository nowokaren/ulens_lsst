# Interface and data-flow reference

This document describes the current development branch for the planned v1.0.3.
The interfaces below were checked against the code. This is a guide to the
existing implementation, not a promise that all listed methods are stable API.

## Pipeline

```text
YAML configuration
       ↓
SimPipeline: event setup and light curves
       ↓
LSSTData: catalog queries and calexp_catalog()
       ↓
serializable ImageProduct descriptors
       ↓
ParallelProcessor → worker-local make_image_access(config)
                     ├── ButlerImageAccess
                     └── TapImageAccess → DataLink → cached FITS
       ↓
ExposureF → Calexp
       ↓
LSSTTools: injection and forced photometry
       ↓
Parquet results and run summaries
       ↓
analysis.py: plots and post-run diagnostics
```

The selection between Butler and TAP happens in `make_image_access`, using the
run's `image_access` configuration. Catalog queries and event generation are
separate stages. `image_access` controls image discovery/loading; `data_preview`
continues to identify the data context. Workers receive serializable
configuration and image descriptors, then construct a local backend.

## Module and interface map

### Pipeline-facing modules

```text
SimPipeline(config_path=None, from_folder=None, setup_dir=True, new=True)
├── simulate_lightcurves(event_processor=None) -> DataFrame
├── load_nearby_objects()
├── process_synthetic_photometry(bands=None) -> DataFrame
└── compute_events_chi2() -> DataFrame

LSSTData(...)
├── load_catalog(catalog, columns=None, filters=None, region=None, ...)
│      -> DataFrame
├── calexp_catalog(bands=None, n_max=None, load_mjd=True) -> None
└── load_calexp(visit=None, detector=None, dataset_ref=None) -> Calexp

ImageProduct(...)
├── to_dict() -> dict
├── to_record() -> dict
├── to_json() -> str
└── from_dict(value) -> ImageProduct

make_image_access(config=None, *, data_preview="dp0", ..., cache_dir=None)
    -> ButlerImageAccess | TapImageAccess

TapImageAccess(...)
├── find_exposures(*, region=None, bands=None, obs_id=None,
│                   visit=None, detector=None, limit=100)
│      -> list[ImageProduct]
└── load_exposure(product, *, output_path=None, overwrite=False)
       -> Any  # returns ExposureF at runtime

ButlerImageAccess(...)
├── find_exposures(*, region, bands, limit=None) -> list[ImageProduct]
└── load_exposure(product) -> Any  # returns ExposureF at runtime

Calexp(data_id, ..., metadata=None)
├── from_exposure(exposure, metadata=None, **kwargs) -> Calexp
├── from_fits(path, metadata=None, **kwargs) -> Calexp
├── cutout(roi) -> Calexp
├── pix_to_sky(x, y) -> (ra_deg, dec_deg)
├── sky_to_pix(ra, dec) -> (x, y)
├── plot(...) -> Axes | None
└── save_fits(path) -> None

LSSTTools(calexp, injection_config=None, measurement_config=None)
├── create_injection_catalog(...) -> Table
├── inject_sources(injection_catalog) -> dict
├── measure_photometry(exposure, catalog) -> DataFrame
└── process_calexp(injection_catalog) -> dict
```

Relevant signatures:

```python
SimPipeline(config_path: str = None, from_folder: str = None,
            setup_dir: bool = True, new: bool = True)

LSSTData(ra=62.0, dec=-34.0, radius=0.01, data_preview="dp1",
         bands=["u", "g", "r", "i", "z", "y"], name=None,
         main_path="./runs", image_access=None, image_cache_dir=None)

make_image_access(config=None, *, data_preview="dp0", butler=None,
                  service=None, session=None, cache_dir=None)

TapImageAccess(*, collection="LSST.DP1", product_subtype="lsst.visit_image",
               token_path=DEFAULT_TOKEN_PATH, tap_url=DEFAULT_TAP_URL,
               auth="rsp", service=None, session=None, cache_dir=None)

ButlerImageAccess(*, data_preview="dp0", collection=None,
                  butler_config=None, dataset_type=None, butler=None)

Calexp(data_id, butler=None, butler_config=None, data_preview=None, name=None,
       ra=0.0, dec=0.0, radius=0.0, bands=None, main_path=".", metadata=None)
```

The descriptor's dataclass fields are:

```python
ImageProduct(
    backend: str, stable_id: str, collection: str, product_type: str,
    product_subtype: Optional[str] = None, obs_id: Optional[str] = None,
    obs_publisher_did: Optional[str] = None, visit: Optional[int] = None,
    detector: Optional[int] = None, band: Optional[str] = None,
    expMidptMJD: Optional[float] = None, t_min: Optional[float] = None,
    t_max: Optional[float] = None, obscore_midpoint_mjd: Optional[float] = None,
    seeing: Optional[float] = None, psfSigma: Optional[float] = None,
    magLim: Optional[float] = None, datalink_url: Optional[str] = None,
    dataset_type: Optional[str] = None, data_id: Optional[dict[str, Any]] = None,
    local_path: Optional[str] = None,
    metadata: dict[str, Any] = field(default_factory=dict),
)
```

Lower-level image-access functions are available for explicit control of each
step:

```python
make_tap_service(token_path=DEFAULT_TOKEN_PATH, tap_url=DEFAULT_TAP_URL,
                 *, auth="rsp", session=None) -> tuple[Any, Any]
find_tap_images(service, *, obs_collection="LSST.DP1",
                product_subtype="lsst.visit_image", obs_id=None, visit=None,
                detector=None, region=None, bands=None, limit=100) -> list[dict[str, Any]]
resolve_datalink_product(session, datalink_url, *, semantics="#this") -> dict
download_tap_image(session, product, output_path, *, overwrite=False,
                   chunk_size=2 * 1024**2) -> Path
load_fits_exposure(path) -> Any             # ExposureF at runtime
find_butler_images(butler, *, dataset_type, where="", bind=None, limit=None) -> list[Any]
load_butler_exposure(butler, product, *, dataset_type=None) -> Any  # ExposureF
```

The backend objects and factory are the intended pipeline-facing interface.

Important plotting and science signatures:

```python
Calexp.plot(title=None, sources=None, show=False, close=False, ax=None,
            figsize=(6, 6), save_path=None, ra_space=10, vmin=None, vmax=None)

LSSTTools.create_injection_catalog(
    ra_values, dec_values, mag_values, magnification_values=None,
    event_ids=None, expMidptMJD=None, visit=None, detector=None,
    source_type="Star",
)
LSSTTools.inject_sources(injection_catalog)
LSSTTools.measure_photometry(injected_exposure, injected_catalog)
LSSTTools.process_calexp(injection_catalog)

plot_event_fov(event_id, sim_name, index_run=1, bands=None, save=True,
               show_nearby_objects=True)
plot_sky_map(sim_name=None, circle_region=None, n_calexps=5, zoom=0.1,
             plot_injected=True, file_name="sky_map.png", sources_alpha=0.8,
             sources_ms=0.8, data_preview="dp0", save=None)
event_gif(event_id, sim_name, band, output_dir=None, duration=5,
          cutout_size=200, point_radius=20)
```

In the pipeline, `ParallelProcessor` and `Catalog` are active implementation
components. `simulation_pipeline.py` uses `ParallelProcessor` for event,
photometry, and chi-squared work; `Catalog` manages Parquet outputs and combines
worker files. `Catalog` is also exported from the package, but most users should
interact with it through `SimPipeline`.

`processing_event_pipelines.py` holds worker functions, including
`LSST_synthetic_photometry(row, config)` and `compute_chi2(row, config)`.
`_load_worker_calexp(row, config)` is an internal helper that rebuilds an
`ImageProduct`, creates/reuses a worker-local backend, and returns a `Calexp`.

`light_curves.py` provides `Event`, including `simulate_lc(...)`, `plot(...)`,
`Event.from_parquet(event_id, photometry_path, events_path)`, and
`compute_chi2(...)`. `analysis.py` provides `plot_event_fov(event_id, sim_name,
index_run=1, bands=None, save=True, show_nearby_objects=True)`,
`plot_sky_map(...)`, and `event_gif(...)` for saved runs.

Other active/importable symbols:

- `processing_event_pipelines.py`: `process_cte_event(row, config)`,
  `process_ulens_event(row, config)`, `process_ulens_event_rubin_sim(row,
  config)`, `process_SNANA_ulens_event(row, config)`,
  `process_ulens_event_load_nearby_object(row, config)`,
  `LSST_synthetic_photometry(row, config)`, and `compute_chi2(row, config)`.
  These are worker callbacks, not user-level pipeline entry points.
- `region_sky.py`: `circ_sample(...)`, `spherical_circle_edge(...)`,
  `spherical_polygon_area(...)`, `spherical_circle_area(...)`; `SkyRegion`
  exposes `calculate_area(spherical=True)`, `edge(n_points=100)`,
  `contains(ra, dec)`, and `sample(n_points, seed=None)`. `LSSTDataLoader` has
  `load_calexps(n_max="all", bands=None)` and `load_sources(ra=None, dec=None,
  radius=None)`, but is a legacy Butler path.
- `utils.py`: `check_memory(...)`, `setup_logger(...)`,
  `get_nearby_objects(...)`, `count_objects_within_radius(...)`,
  `sky_catalog_query(...)`, `baseline_name()`, `get_dataSlice(...)`,
  `get_lsst_noisy_lc(...)`, and `get_lsst_mjds_per_band(...)`. They are helper
  functions used by pipeline/workers, not stable package-level exports.
- `ulens_utils.py`: `MicrolensingParams` exposes derived lens/source and orbital
  properties; active functions include `generate_event_params(...)`,
  `create_rubin_telescopes(...)`, `create_rubin_event(...)`,
  `simulate_lightcurve(...)`, `simulate_ideal_microlensing_event(...)`,
  `pylima_parameters(...)`, `pacz_parameters(...)`,
  `simulate_pacz_lightcurve(...)`, and `simulate_pylima_lightcurve(...)`.
  `Event` and event workers use these scientific helpers.
- `analysis.py`: also provides `parse_timestamp(...)`, `parse_task(...)`,
  `compute_durations(df)`, `generate_time_log(...)`,
  `generate_summary_plot(...)`, `get_folder_size(...)`, and `format_size(...)`.
- `catalogs_utils.py`: `Catalog.create(...)`, `add_rows(...)`,
  `combine_parquet_files(...)`, `get_max_value(...)`, and `get_schema()`.
- `parallel_utils.py`: `ParallelProcessor.process(...)`,
  `process_chunk(...)`, `save_results(...)`, and `get_results()`.
- `run_pipeline.py`: `main()`, the entry point declared by the `ulens_lsst`
  console script.

### Module responsibilities and use

| Module | Classification | Responsibility and pipeline users |
|---|---|---|
| `run_pipeline.py` | Active entry point | CLI creates `SimPipeline` and calls selected stages. |
| `simulation_pipeline.py` | Active orchestration | Coordinates events, exposure discovery, workers, and result files; uses `Catalog`, `ParallelProcessor`, `SkyRegion`, and worker functions. |
| `image_access.py` | Active access interface | Defines `ImageProduct`, TAP/DataLink and Butler operations, configuration validation, and the backend factory; used by `LSSTData` and workers. |
| `lsst_data.py` | Active data interface | `LSSTData` queries catalogs and delegates image access; `Calexp` wraps an exposure and offers WCS, cutout, and plot operations. Used by pipeline, workers, and analysis. |
| `lsst_tools.py` | Active scientific processing | Performs source injection and forced photometry on a `Calexp`; used by the synthetic-photometry worker and analysis plots. |
| `processing_event_pipelines.py` | Active internal | Implements per-event/per-exposure workers called by `SimPipeline` through `ParallelProcessor`. |
| `parallel_utils.py` | Active internal; API documented | Implements chunking, multiprocessing, checkpoints, and result handling; used by three pipeline stages. |
| `catalogs_utils.py` | Active internal and exported | Implements Parquet catalog creation, appends, and combination; used by pipeline, workers, and `Event`. |
| `light_curves.py` | Active user-facing module | Defines `Event`, light-curve simulation/plotting, serialization, and event-level chi-squared; used by pipeline and analysis. |
| `analysis.py` | Active user-facing module | Reloads run products and produces event/FOV, sky, GIF, and performance diagnostics. |
| `region_sky.py` | Mixed | `SkyRegion` samples the pipeline's sky region. `LSSTDataLoader` is a separate Butler-oriented legacy loader, not used by the current pipeline path. |
| `utils.py` | Active internal utilities | Provides catalog/query, nearby-object, cadence, and logging helpers used by pipeline/workers. |
| `ulens_utils.py` | Active scientific helpers | Generates microlensing parameters and model light curves; used by `Event`. |
| `__init__.py` | Package entry point | Exposes `SimPipeline` and `Catalog`, version, and LSST setup check. Most other interfaces are imported from their modules. |

`parallel_utils.py` and `catalogs_utils.py` are not unused: they are imported and
called from the modules identified above and are covered by tests. No module is
removed here. `region_sky.LSSTDataLoader` has no call site in the current
pipeline; keep it as legacy until external/notebook use is assessed.

## Data provenance

| Value | Origin and transport | Meaning in this implementation |
|---|---|---|
| Image, mask, variance | Loaded together into the LSST `ExposureF`, from a Butler dataset or a TAP FITS product. | Exposure planes used by downstream processing. This is image content, not cache metadata. |
| WCS | Exposure metadata read by `ExposureF`; accessed from `Calexp.wcs`. | Used for pixel/sky transforms, cutouts, overlays, and plots. |
| Visit, detector, band | Butler `DatasetRef.dataId` for Butler; ObsCore `lsst_visit`, `lsst_detector`, `lsst_band` for TAP. TAP discovery joins visit/detector to `CcdVisit`. | Stored in `ImageProduct`; flattened into `data_calexps.csv`; reconstructed in workers. |
| `expMidptMJD` | `CcdVisit.expMidptMJD`; in TAP discovery, matched by `(visitId, detector)`. In Butler mode, the pipeline obtains it from the `CcdVisit` catalog. | Exposure epoch used by the light-curve/photometry pipeline. |
| `t_min`, `t_max`, `obscore_midpoint_mjd` | ObsCore TAP row; midpoint is computed from the two bounds when both are present. | Kept distinct from `expMidptMJD`; the values are not silently substituted for each other. |
| `psfSigma`, `seeing`, `magLim` | Matching `CcdVisit` row where available. | Stored under separate names. There is no implicit conversion from `psfSigma` to `seeing`; either may be absent. |
| `ImageProduct` | Built by the access backend, serialized with `to_record()`, written with the exposure catalog, reconstructed by `image_product_from_record(...)` in the worker. | Carries backend, stable ID, collection/product fields, association metadata, DataLink origin or Butler data ID, local path, and other metadata. |
| Effective backend configuration | Run YAML: `lsst_images.data_preview` selects the data context; `image_access` selects image discovery/loading. `SimPipeline` passes these settings to `LSSTData` and worker configuration. | Workers reconstruct their own backend; live Butler/TAP service/session objects are not serialized. |

TAP keeps the original DataLink URL as provenance. The final `#this` URL can be
signed and temporary, so it is resolved near download time and is not the stable
product identifier. The local FITS cache is infrastructure: it is keyed from
the stable product ID, guarded by a per-path lock, and written by atomic
replacement. It does not alter the science fields in the exposure.

## Minimal usage examples

Examples that query Butler/TAP or load an `ExposureF` require an activated Rubin
Science Pipelines environment. TAP `auth: rsp` additionally requires an
authenticated RSP session. These snippets are examples of the current API; they
do not run a scientific pipeline here.

### 1. Load and edit run configuration

```python
from pathlib import Path
import yaml

config = yaml.safe_load(Path("config/config_file.yaml").read_text())
config["name"] = "demo"
Path("demo.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
```

### 2. Select Butler or TAP

Butler remains the default when `image_access` is absent. Explicit configuration:

```yaml
lsst_images:
  data_preview: dp0
image_access:
  backend: butler
```

TAP is currently validated for DP1 `lsst.visit_image`:

```yaml
lsst_images:
  data_preview: dp1
image_access:
  backend: tap
  auth: rsp
  collection: LSST.DP1
  product_subtype: lsst.visit_image
```

### 3. Search for an exposure

TAP and Butler backends accept region and band filters and return
`ImageProduct` descriptors:

```python
from ulens_lsst.image_access import make_image_access

access = make_image_access(data_preview="dp0")
products = access.find_exposures(
    region=(62.0, -36.0, 0.05), bands=["r"], limit=1
)
```

### 4. Load a TAP exposure

```python
from ulens_lsst.image_access import make_image_access

access = make_image_access(
    {"backend": "tap", "auth": "rsp", "collection": "LSST.DP1",
     "product_subtype": "lsst.visit_image"},
    data_preview="dp1", cache_dir="./image_cache",
)
product = access.find_exposures(
    region=(62.0, -36.0, 0.05), bands=["r"], limit=1
)[0]
exposure = access.load_exposure(product)
```

`load_exposure` resolves DataLink, downloads or reuses a cached FITS, and returns
an `ExposureF`. For direct use without `cache_dir`, pass `output_path` to
`load_exposure`.

### 5. Load a Butler exposure

```python
from ulens_lsst.image_access import make_image_access

access = make_image_access(data_preview="dp0")
product = access.find_exposures(
    region=(62.0, -36.0, 0.05), bands=["r"], limit=1
)[0]
exposure = access.load_exposure(product)
```

Butler initialization is lazy; this example's search requires the configured
Butler repository and collections.

### 6. Wrap an `ExposureF` or local FITS

```python
from ulens_lsst.lsst_data import Calexp

calexp = Calexp.from_exposure(exposure, metadata=product.to_dict())
local_calexp = Calexp.from_fits("image.fits")
```

These paths do not initialize Butler. The local FITS must be readable as an LSST
`ExposureF`.

### 7. Plot an exposure

```python
ax = calexp.plot(title="Exposure")
```

### 8. Cut out and transform coordinates

```python
cutout = calexp.cutout(roi=((62.0, -36.0), 200))
ra, dec = calexp.pix_to_sky(100, 200)
x, y = calexp.sky_to_pix(ra, dec)
```

`roi` is `((ra_deg, dec_deg), size_in_pixels)`; `cutout` returns another
`Calexp`.

### 9. Run the typical pipeline

The public Python workflow consists of explicit stages:

```python
from ulens_lsst.simulation_pipeline import SimPipeline

sim = SimPipeline("demo.yaml")
sim.simulate_lightcurves()
sim.load_nearby_objects()
sim.process_synthetic_photometry()
sim.compute_events_chi2()
```

The equivalent CLI orchestration is `ulens_lsst --config demo.yaml --steps all`.
There is no single Python `run()` method currently; callers invoke stages
directly or use the CLI.

### 10. Read results and analyze a saved run

```python
import pandas as pd
from ulens_lsst.analysis import plot_event_fov
from ulens_lsst.light_curves import Event

events = pd.read_parquet(sim.events_file)
event = Event.from_parquet(
    int(events.iloc[0]["event_id"]), sim.photometry_file, sim.events_file
)
event.plot()
plot_event_fov(int(event.event_id), sim.name, index_run=sim.new_file_index, save=False)
```

`plot_event_fov` reloads the run and its saved exposure descriptors to construct
the cutouts. The `Event` constructor and result filenames are tied to the
current run layout, so analysis of a run outside the default `runs/` layout may
require constructing the configured paths explicitly.

## Interface assessment

### Clear and already needed for Butler/TAP

- `ImageProduct` provides a serializable handoff from discovery to workers.
- `make_image_access` centralizes backend selection; both backends return
  `ExposureF` for the scientific processing layer.
- `Calexp.from_exposure` and `Calexp.from_fits` separate local/in-memory input
  from Butler access.
- Per-worker backend reconstruction avoids serializing live service sessions.

### Small, safe improvements before v1.0.3

- The Markdown reference is linked from the Sphinx index and usage page as a
  downloadable source file and from the README. It is not rendered inline by
  Sphinx, which currently has no Markdown parser configured.
- The API page exposes `image_access`, `LSSTData`, and `Calexp`; its worker
  module reference uses the implemented `processing_event_pipelines` name.
- The command examples use the installed `ulens_lsst` CLI name.

### Improvements that can wait

- `LSSTData` still combines catalog queries, run storage, and image discovery;
  splitting these responsibilities is a broader refactor.
- `Calexp.get_data()` performs a separate TAP catalog query and requires visit
  and detector metadata. Its name does not communicate that I/O clearly.
- `SimPipeline` has no single `run()` method, and `from_folder` is oriented to
  the conventional `runs/` layout; improving custom output-root ergonomics can
  be considered separately.
- `region_sky.LSSTDataLoader` is Butler-only and outside the integrated
  pipeline. Keep it for compatibility until external usage is known; consider
  deprecation only in a later release.
- Most lower-level helpers in `utils.py` and `ulens_utils.py` are importable but
  not presented as stable public APIs. Document them only if users need them.

These documentation and discoverability issues do not require changing the
scientific calculations or Butler/TAP loading contract for v1.0.3.
