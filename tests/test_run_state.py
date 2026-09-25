"""Offline tests for operational state, restart planning, and finalization."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from ulens_lsst.run_state import (
    classify_work_unit,
    finalize,
    pending_work_units,
    status_summary,
)
from ulens_lsst.work_units import build_work_unit_rows


class TestRunState(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name)
        (self.run_dir / "status").mkdir()
        (self.run_dir / "photometry").mkdir()
        rows = build_work_unit_rows(["image-a", "image-b"], 1)
        pq.write_table(pa.Table.from_pylist(rows), self.run_dir / "work_units.parquet")
        self.rows = rows

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_status(self, work_unit_id, state, **extra):
        payload = {"work_unit_id": work_unit_id, "state": state, **extra}
        (self.run_dir / "status" / f"part-{work_unit_id}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def write_part(self, work_unit_id, event_id=1, image_id="image-a"):
        pq.write_table(
            pa.table({"event_id": [event_id], "image_id": [image_id]}),
            self.run_dir / "photometry" / f"part-{work_unit_id}.parquet",
        )

    def test_pending_and_completed_states(self):
        first, second = [row["work_unit_id"] for row in self.rows]
        self.write_part(first, image_id="image-a")
        self.write_status(first, "completed", n_images=1, n_photometry_rows=1)
        self.assertEqual(classify_work_unit(self.run_dir, first), "completed")
        self.assertEqual(pending_work_units(self.run_dir), [second])

    def test_failed_empty_and_invalid_states_are_restartable(self):
        first, second = [row["work_unit_id"] for row in self.rows]
        self.write_status(first, "failed")
        pq.write_table(
            pa.table(
                {
                    "event_id": pa.array([], type=pa.int64()),
                    "image_id": pa.array([], type=pa.string()),
                }
            ),
            self.run_dir / "photometry" / f"part-{second}.parquet",
        )
        self.write_status(second, "empty_success", n_photometry_rows=0)
        self.assertEqual(classify_work_unit(self.run_dir, second), "empty_success")
        self.assertEqual(pending_work_units(self.run_dir), [first])

    def test_finalize_complete_run(self):
        for index, row in enumerate(self.rows):
            work_unit_id = row["work_unit_id"]
            image_id = f"image-{chr(ord('a') + index)}"
            self.write_part(work_unit_id, event_id=index, image_id=image_id)
            self.write_status(work_unit_id, "completed", n_images=1, n_photometry_rows=1)
        result = finalize(self.run_dir)
        self.assertTrue(result["complete"])
        self.assertEqual(result["duplicate_scientific_keys"], 0)

    def test_finalize_rejects_duplicate_keys(self):
        first, second = [row["work_unit_id"] for row in self.rows]
        self.write_part(first, event_id=1, image_id="image-a")
        self.write_part(second, event_id=1, image_id="image-a")
        self.write_status(first, "completed", n_images=1, n_photometry_rows=1)
        self.write_status(second, "completed", n_images=1, n_photometry_rows=1)
        result = finalize(self.run_dir)
        self.assertFalse(result["complete"])
        self.assertEqual(result["duplicate_scientific_keys"], 1)

    def test_missing_stale_and_corrupt_work_are_restartable(self):
        first, second = [row["work_unit_id"] for row in self.rows]
        old = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
        self.write_status(first, "running", started_at=old)
        self.write_status(second, "completed", n_images=1, n_photometry_rows=1)
        (self.run_dir / "photometry" / f"part-{second}.parquet").write_bytes(b"not parquet")
        self.assertEqual(classify_work_unit(self.run_dir, first), "stale")
        self.assertEqual(classify_work_unit(self.run_dir, second), "invalid")
        self.assertEqual(pending_work_units(self.run_dir), [first, second])

    def test_finalize_reports_incomplete_campaign(self):
        result = finalize(self.run_dir)
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["incomplete_work_units"]), 2)


if __name__ == "__main__":
    unittest.main()
