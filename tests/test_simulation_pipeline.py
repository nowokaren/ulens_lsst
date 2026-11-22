"""Local, data-independent tests for SimPipeline initialization."""

import tempfile
import unittest
from pathlib import Path

import yaml

from ulens_lsst.simulation_pipeline import SimPipeline


class TestSimPipeline(unittest.TestCase):
    def test_initialization_from_minimal_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "main_path": str(Path(tmp) / "runs"),
                        "name": "test_run",
                        "n_events": 2,
                        "bands": ["g"],
                        "model": "PSPL",
                        "system_type": "BH",
                        "sim_type": "ideal",
                        "n_processes": 1,
                    }
                )
            )
            sim = SimPipeline(str(config_path), setup_dir=False)

        self.assertEqual(sim.sim_type, "ideal")
        self.assertEqual(sim.n_processes, 1)
        self.assertEqual(sim.main_path, str(Path(tmp) / "runs"))
        self.assertEqual(sim.bands, ["g"])


if __name__ == "__main__":
    unittest.main()
