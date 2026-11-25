"""Regression tests for Calexp's input and data-access boundaries."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ulens_lsst.lsst_data import Calexp


class FakeExposure:
    def __init__(self, *args):
        self.args = args

    def getWcs(self):
        return Mock()

    def getFilter(self):
        return Mock(bandLabel="r")


class TestCalexpAccess(unittest.TestCase):
    def test_wrapping_exposure_does_not_construct_butler_and_keeps_metadata(self):
        exposure = FakeExposure()
        metadata = {"stable_id": "obs:123", "visit": 123, "detector": 4}

        with (
            patch("ulens_lsst.lsst_data.afwImage.ExposureF", FakeExposure),
            patch("ulens_lsst.lsst_data.Butler", side_effect=AssertionError("Butler should be lazy")),
        ):
            calexp = Calexp.from_exposure(exposure, metadata=metadata)

        self.assertIs(calexp.expF, exposure)
        self.assertIsNone(calexp.butler)
        self.assertEqual(calexp.metadata, metadata)
        self.assertEqual(calexp._default_title(), "obs:123 (Band: r)")

    def test_loading_fits_does_not_construct_butler(self):
        with tempfile.TemporaryDirectory() as tmp:
            fits_path = Path(tmp) / "visit_image.fits"
            fits_path.write_bytes(b"test image")
            with (
                patch("ulens_lsst.lsst_data.afwImage.ExposureF", FakeExposure),
                patch("ulens_lsst.lsst_data.Butler", side_effect=AssertionError("Butler should be lazy")),
            ):
                calexp = Calexp.from_fits(fits_path, metadata={"stable_id": "obs:123"})

        self.assertEqual(calexp.expF.args, (str(fits_path),))
        self.assertIsInstance(calexp.expF, FakeExposure)
        self.assertIsNone(calexp.butler)
        self.assertEqual(calexp.metadata["stable_id"], "obs:123")

    def test_legacy_data_id_still_loads_through_butler(self):
        exposure = FakeExposure()
        butler = Mock()
        butler.get.return_value = exposure
        with (
            patch("ulens_lsst.lsst_data.afwImage.ExposureF", FakeExposure),
            patch("ulens_lsst.lsst_data.Butler", return_value=butler) as butler_factory,
        ):
            calexp = Calexp(
                {"visit": 123, "detector": 4},
                data_preview="dp1",
            )

        butler_factory.assert_called_once_with("dp1", collections="LSSTComCam/DP1")
        butler.get.assert_called_once_with("visit_image", visit=123, detector=4)
        self.assertIs(calexp.expF, exposure)
        self.assertIs(calexp.butler, butler)

    def test_dataset_ref_is_supported_explicitly(self):
        class FakeDatasetRef:
            id = "dataset-uuid"
            dataId = {"visit": 123, "detector": 4}
            datasetType = Mock(name="visit_image")

        exposure = FakeExposure()
        butler = Mock()
        butler.get.return_value = exposure
        with (
            patch("ulens_lsst.lsst_data.DatasetRef", FakeDatasetRef),
            patch("ulens_lsst.lsst_data.afwImage.ExposureF", FakeExposure),
            patch("ulens_lsst.lsst_data.Butler", return_value=butler),
        ):
            calexp = Calexp(FakeDatasetRef(), data_preview="dp1")

        butler.get.assert_called_once()
        self.assertEqual(calexp.data_id, {"visit": 123, "detector": 4})

    def test_get_data_without_visit_detector_fails_clearly(self):
        calexp = object.__new__(Calexp)
        calexp.data_id = {}
        calexp.metadata = {}
        calexp.data_preview = "dp1"
        calexp.tap_schema = "dp1"

        with patch("ulens_lsst.lsst_data.get_tap_service") as tap_service:
            with self.assertRaisesRegex(ValueError, "visit and detector"):
                calexp.get_data()

        tap_service.assert_not_called()


if __name__ == "__main__":
    unittest.main()
