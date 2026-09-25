"""Tests for the generic image context."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "ulens_lsst"
package = types.ModuleType("ulens_lsst")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("ulens_lsst", package)

for module_name in ("work_units", "image_context"):
    path = PACKAGE_ROOT / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"ulens_lsst.{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"ulens_lsst.{module_name}"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

ImageContext = sys.modules["ulens_lsst.image_context"].ImageContext


class FakeRelease:
    """Minimal stand-in exposing only the installed RSP API used here."""

    name = "test-release"
    calexp_dataset_types = ("image_product",)

    @staticmethod
    def calexp_data_id(visit, detector):
        return {"visit": int(visit), "detector": int(detector)}


class TestImageContext(unittest.TestCase):
    def test_context_uses_release_dataset_and_data_id(self):
        context = ImageContext.from_release(
            FakeRelease(),
            42,
            7,
            band="r",
            mjd=61234.5,
        )

        self.assertEqual(context.dataset_type, "image_product")
        self.assertEqual(context.data_id, {"visit": 42, "detector": 7})
        self.assertEqual(context.visit, 42)
        self.assertEqual(context.detector, 7)
        self.assertEqual(context.band, "r")
        self.assertIn("test-release:image_product", context.image_id)


if __name__ == "__main__":
    unittest.main()
