import logging
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from ulens_lsst.simulation_pipeline import SimPipeline
from ulens_lsst.utils import count_objects_within_radius


class TestNearbyObjectCounts(unittest.TestCase):
    def test_counts_unique_objects_within_radius(self):
        targets = np.array([[0.0, 0.0], [1.0, 0.0]])
        catalog = np.array([[0.0, 0.0], [0.0005, 0.0], [0.01, 0.0], [1.0, 0.0], [1.0005, 0.0]])
        object_ids = np.array([10, 11, 12, 20, 20])

        counts = count_objects_within_radius(targets, catalog, 2.0, object_ids)

        np.testing.assert_array_equal(counts, [2, 1])

    def test_empty_catalog_returns_zero_counts(self):
        counts = count_objects_within_radius(
            np.array([[0.0, 0.0]]), np.empty((0, 2)), 2.0, np.array([], dtype=int)
        )

        np.testing.assert_array_equal(counts, [0])

    def test_negative_radius_is_rejected(self):
        with self.assertRaises(ValueError):
            count_objects_within_radius(
                np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]), -1.0, np.array([1])
            )

    def test_load_nearby_objects_persists_nearest_and_count(self):
        class FakeLSSTData:
            def __init__(self, **kwargs):
                self.data_preview = kwargs["data_preview"]
                self.radius = kwargs["radius"]

            def load_catalog(self, catalog, columns):
                return pd.DataFrame({
                    "coord_ra": [0.0, 0.0005, 0.01],
                    "coord_dec": [0.0, 0.0, 0.0],
                    "ObjectId": [10, 11, 12],
                    "mag_g": [20.0, 21.0, 22.0],
                    "fwhm_g": [0.8, 0.9, 1.0],
                })

        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.parquet"
            pd.DataFrame({
                "event_id": [1, 2],
                "ra": [0.0, 0.009],
                "dec": [0.0, 0.0],
            }).to_parquet(events_path, index=False)

            sim = SimPipeline.__new__(SimPipeline)
            sim.logger = logging.getLogger("test-nearby-object-counts")
            sim.sim_type = "lsst_images"
            sim.events_file = str(events_path)
            sim.main_path = tmp
            sim.sky_center = {"coord": [0.0, 0.0], "frame": "icrs"}
            sim.sky_radius = 0.01
            sim.nearby_object_count_radius_arcsec = 2.0
            sim.data_preview = "dp0"
            sim.bands = ["g"]
            sim.name = "test"

            fake_module = SimpleNamespace(LSSTData=FakeLSSTData)
            with patch.dict(sys.modules, {"ulens_lsst.lsst_data": fake_module}):
                sim.load_nearby_objects()

            result = pd.read_parquet(events_path).sort_values("event_id")
            np.testing.assert_array_equal(result["nearby_object_n_within_radius"], [2, 0])
            np.testing.assert_array_equal(result["nearby_object_objId"], [10, 12])
            np.testing.assert_allclose(result["nearby_object_count_radius_arcsec"], [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
