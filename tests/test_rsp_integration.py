"""Integration tests for an accessible Rubin Science Pipelines repository.

These tests are skipped unless the caller supplies a compatible Butler profile
through environment variables.  The values are intentionally not part of
ulens_lsst's release configuration.
"""

import os
import unittest
from dataclasses import replace

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
    for name in (
        "ULENS_LSST_RSP_RELEASE",
        "ULENS_LSST_RSP_COLLECTION",
        "ULENS_LSST_RSP_IMAGE_DATASET",
    )
)


@unittest.skipUnless(
    RSP_READY and get_data_release is not None,
    "requires an accessible RSP Butler profile and collection",
)
class TestRspImageAccess(unittest.TestCase):
    """Exercise one real calibrated image through the generic access path."""

    @classmethod
    def setUpClass(cls):
        release = get_data_release(os.environ["ULENS_LSST_RSP_RELEASE"])
        cls.release = replace(
            release,
            butler_collections=os.environ["ULENS_LSST_RSP_COLLECTION"],
            calexp_dataset_types=(os.environ["ULENS_LSST_RSP_IMAGE_DATASET"],),
        )
        cls.data = LSSTData(
            release_config=cls.release,
            main_path=os.environ.get("ULENS_LSST_RSP_TEST_OUTPUT", "/tmp"),
        )
        dataset_type = os.environ["ULENS_LSST_RSP_IMAGE_DATASET"]
        cls.ref = next(
            iter(
                cls.data.butler.registry.queryDatasets(
                    dataset_type,
                    collections=cls.release.butler_collections,
                    where=os.environ.get("ULENS_LSST_RSP_WHERE", ""),
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
