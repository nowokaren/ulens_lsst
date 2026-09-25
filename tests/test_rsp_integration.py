"""Integration tests for an accessible Rubin Science Pipelines repository.

These tests are skipped unless the caller selects an accessible release
through ``ULENS_LSST_RSP_RELEASE``. Release-specific Butler settings come from
the installed ``DataReleaseConfig``.
"""

import os
import unittest
import pandas as pd

try:
    from data_release_config import get_data_release
    from ulens_lsst.image_context import ImageContext
    from ulens_lsst.image_processors import get_image_processor
    from ulens_lsst.lsst_data import LSSTData
except ImportError:  # pragma: no cover - depends on the RSP environment
    get_data_release = None


RSP_READY = all(
    os.environ.get(name)
    for name in ("ULENS_LSST_RSP_RELEASE",)
)


@unittest.skipUnless(
    RSP_READY and get_data_release is not None,
    "requires an accessible RSP Butler profile and collection",
)
class TestRspImageAccess(unittest.TestCase):
    """Exercise one real calibrated image through the generic access path."""

    @classmethod
    def setUpClass(cls):
        cls.release = get_data_release(os.environ["ULENS_LSST_RSP_RELEASE"])
        cls.data = LSSTData(
            release_config=cls.release,
            main_path=os.environ.get("ULENS_LSST_RSP_TEST_OUTPUT", "/tmp"),
        )
        dataset_type = cls.release.calexp_dataset_types[0]
        collection = cls.release.butler_collections
        # Query dimensions first. This avoids an unbounded DatasetRef scan in
        # large registries and uses only generic Butler concepts.
        visit = next(
            iter(
                cls.data.butler.registry.queryDataIds(
                    ["visit"],
                    datasets=dataset_type,
                    collections=collection,
                    where=os.environ.get("ULENS_LSST_RSP_WHERE", ""),
                )
            ),
            None,
        )
        cls.ref = None
        if visit is not None:
            visit_id = int(visit["visit"])
            data_id = next(
                iter(
                    cls.data.butler.registry.queryDataIds(
                        ["visit", "detector"],
                        datasets=dataset_type,
                        collections=collection,
                        where=f"visit={visit_id}",
                    )
                ),
                None,
            )
            if data_id is not None:
                cls.ref = next(
                    iter(
                        cls.data.butler.registry.queryDatasets(
                            dataset_type,
                            collections=collection,
                            dataId=data_id,
                            findFirst=True,
                        )
                    ),
                    None,
                )
        if cls.ref is None:
            raise unittest.SkipTest("configured RSP collection has no image DatasetRef")

    def test_load_real_image_context_and_exposure(self):
        data_id = {
            key: self.ref.dataId[key]
            for key in self.ref.datasetType.dimensions.names
        }
        context = ImageContext(
            image_id="rsp-integration-image",
            dataset_type=self.release.calexp_dataset_types[0],
            data_id=data_id,
            visit=int(data_id["visit"]),
            detector=int(data_id["detector"]),
            band=str(data_id["band"]),
            mjd=float(os.environ.get("ULENS_LSST_RSP_TEST_MJD", "0.0")),
        )
        exposure = self.data.load_image(context).expF
        self.assertIsNotNone(exposure.getWcs())
        self.assertIsNotNone(exposure.getPsf())
        self.assertIsNotNone(exposure.getPhotoCalib())
        self.assertIsNotNone(exposure.getMaskedImage().getMask())
        self.assertIsNotNone(exposure.getMaskedImage().getVariance())

    def test_injection_forced_photometry_on_real_image(self):
        data_id = {
            key: self.ref.dataId[key]
            for key in self.ref.datasetType.dimensions.names
        }
        context = ImageContext(
            image_id="rsp-integration-image",
            dataset_type=self.release.calexp_dataset_types[0],
            data_id=data_id,
            visit=int(data_id["visit"]),
            detector=int(data_id["detector"]),
            band=str(data_id["band"]),
            mjd=float(os.environ.get("ULENS_LSST_RSP_TEST_MJD", "0.0")),
        )
        image = self.data.load_image(context)
        sky = image.expF.getWcs().pixelToSky(image.expF.getBBox().getCenter())
        sources = pd.DataFrame(
            [
                {
                    "ra": sky.getRa().asDegrees(),
                    "dec": sky.getDec().asDegrees(),
                    "ideal_mag": 22.0,
                    "magnification": 1.0,
                    "event_id": 1,
                }
            ]
        )
        result = get_image_processor("injection_forced_photometry").process(
            image, context, sources
        )
        product = result.products["photometry"]
        self.assertEqual(result.status, "success")
        self.assertEqual(len(product), 1)
        self.assertIn("meas_flux", product.colnames)


if __name__ == "__main__":
    unittest.main()
