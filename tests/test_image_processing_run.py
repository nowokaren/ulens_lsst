"""Offline tests for one deterministic PROCESS work unit."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ulens_lsst.image_processing_run import process_work_unit
from ulens_lsst.image_processors import ImageProcessResult
from ulens_lsst.work_units import build_work_unit_rows


class FakeImageData:
    def load_image(self, context):
        return object()


class FakeProcessor:
    def process(self, image, context, sources, config=None):
        return ImageProcessResult(
            products={
                "photometry": pd.DataFrame(
                    {
                        "event_id": [int(sources.iloc[0]["event_id"])],
                        "time": [float(sources.iloc[0]["time"])],
                        "band": [context.band],
                        "ideal_mag": [float(sources.iloc[0]["ideal_mag"])],
                        "magnification": [1.0],
                        "injection_flag": [False],
                        "measure_flag": [""],
                        "meas_flux": [100.0],
                        "meas_flux_err": [2.0],
                    }
                )
            }
        )


class TestProcessWorkUnit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name)
        (self.run_dir / "events").mkdir()
        config = {
            "name": "offline-run",
            "data_release": "test-release",
            "image_processor": "injection_forced_photometry",
            "bands": ["i"],
        }
        with (self.run_dir / "config_file.yaml").open("w") as stream:
            yaml.safe_dump(config, stream)

        image_id = "test-release:calexp:detector=2,visit=7"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "image_id": image_id,
                        "dataset_type": "calexp",
                        "data_id": json.dumps({"visit": 7, "detector": 2}),
                        "visit": 7,
                        "detector": 2,
                        "band": "i",
                        "mjd": 60000.0,
                    }
                ]
            ),
            self.run_dir / "images.parquet",
        )
        rows = build_work_unit_rows([image_id])
        pq.write_table(pa.Table.from_pylist(rows), self.run_dir / "work_units.parquet")
        pq.write_table(
            pa.Table.from_pylist(
                [{"event_id": 11, "time": 60000.0, "band": "i", "ideal_mag": 22.0, "magnification": 1.0}]
            ),
            self.run_dir / "photometry_1.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist([{"event_id": 11, "ra": 10.0, "dec": -2.0}]),
            self.run_dir / "events" / "part-000.parquet",
        )
        self.work_unit_id = rows[0]["work_unit_id"]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_process_writes_one_atomic_scientific_part_and_status(self):
        selection = SimpleNamespace(
            processor_name="injection_forced_photometry",
            release_config=SimpleNamespace(),
        )
        with patch("ulens_lsst.image_processing_run.resolve_configured_image_processing", return_value=selection), \
             patch("ulens_lsst.image_processing_run._create_image_data", return_value=FakeImageData()), \
             patch("ulens_lsst.image_processing_run.get_image_processor", return_value=FakeProcessor()):
            result = process_work_unit(self.run_dir, self.work_unit_id)

        self.assertEqual(result["state"], "completed")
        output = self.run_dir / "photometry" / f"part-{self.work_unit_id}.parquet"
        self.assertTrue(output.exists())
        table = pq.read_table(output)
        self.assertEqual(table.num_rows, 1)
        self.assertEqual(table.column("image_id")[0].as_py(), "test-release:calexp:detector=2,visit=7")
        self.assertTrue((self.run_dir / "status" / f"part-{self.work_unit_id}.json").exists())

    def test_completed_work_unit_is_skipped_without_force(self):
        selection = SimpleNamespace(
            processor_name="injection_forced_photometry",
            release_config=SimpleNamespace(),
        )
        with patch("ulens_lsst.image_processing_run.resolve_configured_image_processing", return_value=selection), \
             patch("ulens_lsst.image_processing_run._create_image_data", return_value=FakeImageData()), \
             patch("ulens_lsst.image_processing_run.get_image_processor", return_value=FakeProcessor()):
            first = process_work_unit(self.run_dir, self.work_unit_id)
            second = process_work_unit(self.run_dir, self.work_unit_id)

        self.assertEqual(first["state"], "completed")
        self.assertTrue(second["skipped"])


if __name__ == "__main__":
    unittest.main()
