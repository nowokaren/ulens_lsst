"""Pipeline wiring tests for serializable image descriptors and worker loading."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from ulens_lsst.image_access import ImageProduct
from ulens_lsst.lsst_data import LSSTData
from ulens_lsst.processing_event_pipelines import _load_worker_calexp


class TestLSSTDataImageAccess(unittest.TestCase):
    def test_butler_initialization_is_lazy_and_legacy_is_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("ulens_lsst.lsst_data.Butler", side_effect=AssertionError("eager Butler")):
                data = LSSTData(main_path=tmp, name="lazy")

        self.assertEqual(data.image_access_config, {"backend": "butler"})
        self.assertIsNone(data._butler)
        self.assertIsNone(data._image_access)

    def test_tap_catalog_keeps_both_time_conventions_and_schema_columns(self):
        product = ImageProduct(
            backend="tap", stable_id="ivo:product", collection="LSST.DP1",
            product_type="image", product_subtype="lsst.visit_image",
            obs_id="obs:1", visit=10, detector=2, band="r", expMidptMJD=61000.5,
            t_min=61000.1, t_max=61000.2, obscore_midpoint_mjd=61000.15,
            datalink_url="https://example.org/datalink", seeing=0.8, psfSigma=2.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            data = LSSTData(main_path=tmp, name="tap", data_preview="dp1",
                            image_access={"backend": "tap", "collection": "LSST.DP1",
                                          "product_subtype": "lsst.visit_image"})
            data._image_access = Mock(find_exposures=Mock(return_value=[product]))
            data.calexp_catalog(bands=["r"])

        row = data.calexps_data.iloc[0]
        self.assertEqual(row["expMidptMJD"], 61000.5)
        self.assertEqual(row["obscore_midpoint_mjd"], 61000.15)
        self.assertEqual(row["t_min"], 61000.1)
        self.assertEqual(row["t_max"], 61000.2)
        self.assertEqual(row["seeing"], 0.8)
        self.assertEqual(row["psfSigma"], 2.0)
        self.assertEqual(data.epochs["r"], [61000.5])
        self.assertEqual(row["backend"], "tap")

    def test_butler_catalog_preserves_legacy_scientific_columns(self):
        product = ImageProduct(
            backend="butler", stable_id="uuid", collection="LSSTComCam/DP1",
            product_type="visit_image", visit=10, detector=2, band="r",
            dataset_type="visit_image", data_id={"visit": 10, "detector": 2, "band": "r"},
        )
        ccd = pd.DataFrame([{
            "visitId": 10, "detector": 2, "expMidptMJD": 61000.5,
            "psfSigma": 2.0, "seeing": 0.8, "magLim": 24.0,
        }])
        with tempfile.TemporaryDirectory() as tmp:
            data = LSSTData(main_path=tmp, name="butler", data_preview="dp1")
            data._image_access = Mock(find_exposures=Mock(return_value=[product]))
            data.load_catalog = Mock(return_value=ccd)
            data.calexp_catalog(bands=["r"])

        row = data.calexps_data.iloc[0]
        for column in ("visit", "detector", "band", "expMidptMJD", "psfSigma", "magLim"):
            self.assertIn(column, data.calexps_data.columns)
        self.assertEqual(row["expMidptMJD"], 61000.5)
        self.assertEqual(row["psfSigma"], 2.0)

    def test_catalog_descriptor_reloads_through_configured_backend(self):
        product = ImageProduct(
            backend="tap", stable_id="ivo:product", collection="LSST.DP1",
            product_type="image", product_subtype="lsst.visit_image", visit=10,
            detector=2, band="r", datalink_url="https://example.org/datalink",
        )
        exposure = object()
        access = Mock(load_exposure=Mock(return_value=exposure))
        with tempfile.TemporaryDirectory() as tmp:
            data = LSSTData(main_path=tmp, name="reload", data_preview="dp1",
                            image_access={"backend": "tap", "collection": "LSST.DP1",
                                          "product_subtype": "lsst.visit_image"})
            data._image_access = access
            with patch("ulens_lsst.lsst_data.Calexp.from_exposure", return_value="calexp") as wrap:
                result = data.load_calexp(dataset_ref=product.to_record())

        self.assertEqual(result, "calexp")
        access.load_exposure.assert_called_once()
        wrap.assert_called_once()


class TestImageWorker(unittest.TestCase):
    def test_worker_reconstructs_backend_once_and_passes_descriptor(self):
        product = ImageProduct(
            backend="tap", stable_id="ivo:product", collection="LSST.DP1",
            product_type="image", product_subtype="lsst.visit_image",
            visit=10, detector=2, band="r", expMidptMJD=61000.5,
            t_min=61000.1, t_max=61000.2, obscore_midpoint_mjd=61000.15,
            datalink_url="https://example.org/datalink",
        )
        row = pd.Series(product.to_record())
        config = {"data_preview": "dp1", "image_access": {"backend": "tap"}}
        access = Mock(load_exposure=Mock(return_value=object()))
        with (
            patch("ulens_lsst.processing_event_pipelines.make_image_access", return_value=access) as factory,
            patch("ulens_lsst.processing_event_pipelines.Calexp.from_exposure", return_value="calexp") as wrap,
        ):
            self.assertEqual(_load_worker_calexp(row, config), "calexp")
            self.assertEqual(_load_worker_calexp(row, config), "calexp")

        factory.assert_called_once_with(config["image_access"], data_preview="dp1", cache_dir=None)
        self.assertEqual(access.load_exposure.call_count, 2)
        self.assertEqual(access.load_exposure.call_args.args[0].stable_id, "ivo:product")
        self.assertEqual(wrap.call_args.kwargs["metadata"]["obscore_midpoint_mjd"], 61000.15)


if __name__ == "__main__":
    unittest.main()
