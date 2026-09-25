"""Tests for deterministic image-processing identities and paths."""

import importlib.util
import unittest
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "ulens_lsst" / "work_units.py"
_SPEC = importlib.util.spec_from_file_location("work_units_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

make_image_id = _MODULE.make_image_id
make_injection_id = _MODULE.make_injection_id
make_work_unit_id = _MODULE.make_work_unit_id
work_unit_product_path = _MODULE.work_unit_product_path


class TestWorkUnitIdentities(unittest.TestCase):
    """Verify that identities do not depend on execution details."""

    def test_image_id_is_independent_of_data_id_insertion_order(self):
        first = make_image_id("DP0.2", "calexp", {"visit": 42, "detector": 7})
        second = make_image_id("DP0.2", "calexp", {"detector": 7, "visit": 42})
        self.assertEqual(first, second)

    def test_injection_id_contains_scientific_identity(self):
        image_id = make_image_id("DP0.2", "calexp", {"visit": 42, "detector": 7})
        injection_id = make_injection_id("run-1", 123, image_id)
        self.assertIn("run-1", injection_id)
        self.assertIn("123", injection_id)
        self.assertIn(image_id, injection_id)

    def test_work_unit_id_is_independent_of_image_order(self):
        first = make_work_unit_id(["image-b", "image-a"])
        second = make_work_unit_id(["image-a", "image-b"])
        self.assertEqual(first, second)

    def test_product_path_is_scoped_to_one_work_unit(self):
        path = work_unit_product_path("runs/example", "photometry", "wu-123")
        self.assertEqual(path.as_posix(), "runs/example/photometry/part-wu-123.parquet")

    def test_product_path_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            work_unit_product_path("runs/example", "photometry", "../other")


if __name__ == "__main__":
    unittest.main()
