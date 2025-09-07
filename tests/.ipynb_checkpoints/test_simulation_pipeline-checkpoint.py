"""
Unit tests for the ulens_lsst simulation pipeline.
"""
import unittest
import pandas as pd
from ulens_lsst.simulation_pipeline import SimPipeline

class TestSimPipeline(unittest.TestCase):
    """Test suite for SimPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_path = 'config.yaml'
        self.sim = SimPipeline(self.config_path)

    def test_initialization(self):
        """Test SimPipeline initialization."""
        self.assertEqual(self.sim.sim_type, 'lsst_images')
        self.assertIsInstance(self.sim.bands, list)

    def test_simulate_lightcurves(self):
        """Test simulate_lightcurves method."""
        results = self.sim.simulate_lightcurves(event_processor='ulens')
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn('event_id', results.columns)
        self.assertIn('status', results.columns)

if __name__ == '__main__':
    unittest.main()