"""Tests for the RSP data-release boundary."""

import unittest
import importlib.util
from dataclasses import replace
from pathlib import Path
import sys


try:
    from data_release_config import DataReleaseConfig
except ImportError:  # pragma: no cover - depends on the RSP environment
    DataReleaseConfig = None

# Import this small boundary module directly.  The package initializer
# currently imports the full RSP-dependent simulation pipeline, which is not
# needed to test release resolution.
_MODULE_PATH = Path(__file__).parents[1] / "ulens_lsst" / "release_config.py"
_SPEC = importlib.util.spec_from_file_location("release_config_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["release_config_under_test"] = _MODULE
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
resolve_data_release = _MODULE.resolve_data_release
check_processor_compatibility = _MODULE.check_processor_compatibility
require_prepare_compatibility = _MODULE.require_prepare_compatibility
resolve_configured_image_processing = _MODULE.resolve_configured_image_processing


@unittest.skipIf(DataReleaseConfig is None, "RSP data_release_config is unavailable")
class TestReleaseConfig(unittest.TestCase):
    """Verify that ulens_lsst delegates release resolution to RSP."""

    def test_default_release_is_resolved_by_rsp(self):
        release = resolve_data_release()

        self.assertIsInstance(release, DataReleaseConfig)
        self.assertTrue(release.name)
        self.assertTrue(release.rsp_instance)
        self.assertTrue(release.calexp_dataset_types)

    def test_existing_release_configuration_is_preserved(self):
        release = resolve_data_release()
        self.assertIs(resolve_data_release(release), release)

    def test_individual_image_profile_requires_rsp_validation(self):
        release = replace(
            resolve_data_release(),
            name="test-image-release",
            photometry_method="calexp_forced",
        )
        result = check_processor_compatibility(
            "injection_forced_photometry", release
        )
        self.assertEqual(result.status, "requires_rsp_validation")
        self.assertIn("RSP", result.message)

    def test_non_image_photometry_profile_is_rejected(self):
        release = replace(resolve_data_release(), photometry_method="coadd_forced")
        with self.assertRaisesRegex(ValueError, "photometry_method"):
            require_prepare_compatibility(
                "injection_forced_photometry", release
            )

    def test_unknown_processor_is_rejected(self):
        release = resolve_data_release()
        with self.assertRaisesRegex(ValueError, "No release requirements"):
            require_prepare_compatibility("future_processor", release)

    def test_missing_data_release_keeps_legacy_path_explicit(self):
        selection = resolve_configured_image_processing(
            {"sim_type": "lsst_images"}
        )
        self.assertIsNone(selection)

    def test_configured_release_is_resolved_for_prepare(self):
        release = replace(
            resolve_data_release(),
            name="test-image-release",
            photometry_method="calexp_forced",
        )
        selection = resolve_configured_image_processing(
            {
                "sim_type": "lsst_images",
                "data_release": release,
                "image_processor": "injection_forced_photometry",
            }
        )
        self.assertIs(selection.release_config, release)
        self.assertEqual(
            selection.processor_name, "injection_forced_photometry"
        )


if __name__ == "__main__":
    unittest.main()
