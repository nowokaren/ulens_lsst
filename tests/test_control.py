"""Offline tests for the run-control command interface."""

import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ulens_lsst.control import main


class TestControl(unittest.TestCase):
    def test_restart_plan_prints_work_unit_ids(self):
        output = io.StringIO()
        with redirect_stdout(output), patch(
            "ulens_lsst.control.pending_work_units", return_value=["wu-a", "wu-b"]
        ):
            result = main(["restart-plan", "/tmp/run"])
        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue().splitlines(), ["wu-a", "wu-b"])

    def test_status_is_human_readable(self):
        output = io.StringIO()
        summary = {
            "total_work_units": 2,
            "completed": 1,
            "running": 0,
            "pending": 1,
            "failed": 0,
            "stale": 0,
            "invalid": 0,
            "empty_success": 0,
            "images_completed": 1,
            "images_expected": 2,
            "photometry_rows": 4,
            "mean_work_unit_seconds": 2.5,
        }
        with redirect_stdout(output), patch(
            "ulens_lsst.control.status_summary", return_value=summary
        ):
            result = main(["status", "/tmp/run"])
        self.assertEqual(result, 0)
        self.assertIn("Work units", output.getvalue())
        self.assertIn("Photometry rows: 4", output.getvalue())


if __name__ == "__main__":
    unittest.main()
