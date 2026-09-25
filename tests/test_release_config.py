"""Tests for the RSP data-release boundary."""

import unittest
import importlib.util
from pathlib import Path


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
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
resolve_data_release = _MODULE.resolve_data_release


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


if __name__ == "__main__":
    unittest.main()
