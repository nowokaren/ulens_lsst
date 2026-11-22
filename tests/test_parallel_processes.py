import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ulens_lsst import __version__
from ulens_lsst.parallel_utils import ParallelProcessor
from ulens_lsst.simulation_pipeline import SimPipeline


def _record_worker_pid(row, config):
    return {
        "worker_pid": os.getpid(),
        "status": "success",
        "error": "",
    }


class FakeCatalog:
    def combine_parquet_files(self, **kwargs):
        pass

    def get_schema(self):
        return pa.schema([])


class TestPipelineProcessConfiguration(unittest.TestCase):
    def _write_config(self, folder, **overrides):
        config = {
            "main_path": str(Path(folder) / "runs"),
            "name": "test_run",
            "n_events": 2,
            "bands": ["g"],
            "model": "PSPL",
            "system_type": "BH",
            "sim_type": "ideal",
            "event_processor": "ulens",
            "photometry_processor": "synthetic",
            "logging": {"log_name": "test_pipeline_processes"},
        }
        config.update(overrides)
        path = Path(folder) / f"{config['name']}.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    def test_version_is_bumped_to_102(self):
        self.assertEqual(__version__, "1.0.2")

    def test_explicit_worker_count_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_config(tmp, n_processes=1)
            sim = SimPipeline(str(config_path), setup_dir=False)

        self.assertEqual(sim.n_processes, 1)

    def test_missing_or_null_worker_count_preserves_historical_default(self):
        for configured_value in ("missing", None):
            with self.subTest(configured_value=configured_value):
                with tempfile.TemporaryDirectory() as tmp:
                    values = {}
                    if configured_value is None:
                        values["n_processes"] = None
                    config_path = self._write_config(tmp, **values)
                    with patch("ulens_lsst.simulation_pipeline.os.cpu_count", return_value=16):
                        sim = SimPipeline(str(config_path), setup_dir=False)

                self.assertEqual(sim.n_processes, 8)

    def test_old_nested_output_setting_does_not_override_pipeline_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_config(
                tmp, format_output_params={"n_processes": 1}
            )
            with patch("ulens_lsst.simulation_pipeline.os.cpu_count", return_value=12):
                sim = SimPipeline(str(config_path), setup_dir=False)

        self.assertEqual(sim.n_processes, 8)

    def test_invalid_worker_counts_are_rejected(self):
        for value in (0, -1, True, 1.5, "2"):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    config_path = self._write_config(tmp, n_processes=value)
                    with self.assertRaisesRegex(ValueError, "n_processes must be a positive integer"):
                        SimPipeline(str(config_path), setup_dir=False)

    def test_distinct_names_create_disjoint_run_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_a = SimPipeline(
                str(self._write_config(tmp, name="run_a")), setup_dir=True
            )
            run_b = SimPipeline(
                str(self._write_config(tmp, name="run_b")), setup_dir=True
            )

            self.assertNotEqual(run_a.output_dir, run_b.output_dir)
            self.assertNotEqual(run_a.temp_dir, run_b.temp_dir)
            for attr in ("events_file", "photometry_file"):
                self.assertNotEqual(getattr(run_a, attr), getattr(run_b, attr))
            for filename in ("temp_input_events.parquet", "results_process_ulens_event.parquet"):
                self.assertNotEqual(
                    Path(run_a.output_dir) / filename,
                    Path(run_b.output_dir) / filename,
                )

    def test_simulate_lightcurves_passes_configured_workers_to_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = SimPipeline(
                str(self._write_config(tmp, n_processes=1)), setup_dir=False
            )
            sim.output_dir = tmp
            sim.temp_dir = str(Path(tmp) / "temp")
            Path(sim.temp_dir).mkdir()
            sim.input_path = str(Path(tmp) / "temp_input_events.parquet")
            sim.n_events = 2
            sim.event_processor_name = "ulens"
            sim._initialize_catalogs = Mock()
            sim._setup_event_ids = Mock(
                side_effect=lambda: setattr(sim, "event_ids", [1, 2])
            )
            sim._sample_sky_positions = Mock(
                return_value=(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
            )
            sim.load_event_sources_catalog = Mock()
            sim.survey_dates = (60000, 60010)
            sim.peak_range = (60000, 60010)
            sim.bands = ["g"]
            sim.sources_catalog = "TRILEGAL"
            sim.blend_distance = 0.001
            sim.mag_sat = {}
            sim.mag_5sigma = {}
            sim.pylima_blend = False
            sim.main_path = tmp
            sim.photometry_schema = {}
            sim.events_schema = {}
            sim.log_config = {"log_name": "test"}
            sim.max_temp_files = None
            sim.photometry_catalog = FakeCatalog()
            sim.events_catalog = FakeCatalog()
            sim.logger = logging.getLogger("test-simulate-lightcurves-processes")

            processor = Mock()
            processor.get_results.return_value = pd.DataFrame(
                {"status": ["success"]}
            )
            with patch(
                "ulens_lsst.simulation_pipeline.ParallelProcessor",
                return_value=processor,
            ) as processor_factory:
                sim.simulate_lightcurves()

        processor.process.assert_called_once_with(num_pools=1)
        processor_config = processor_factory.call_args.kwargs["config"]
        self.assertEqual(processor_config["main_path"], tmp)
        self.assertEqual(processor_config["name"], sim.name)

    def test_process_synthetic_photometry_passes_configured_workers_to_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = SimPipeline(
                str(self._write_config(tmp, n_processes=1)), setup_dir=False
            )
            sim.output_dir = tmp
            sim.temp_dir = str(Path(tmp) / "temp")
            Path(sim.temp_dir).mkdir()
            sim.new_file_index = 1
            sim.events_file = str(Path(tmp) / "events.parquet")
            sim.photometry_file = str(Path(tmp) / "photometry.parquet")
            pq.write_table(
                pa.table({"event_id": [1], "ra": [1.0], "dec": [2.0]}),
                sim.events_file,
            )
            pd.DataFrame({"calexp_id": [10, 11]}).to_csv(
                Path(tmp) / "data_calexps.csv", index=False
            )
            sim.survey_dates = (60000, 60010)
            sim.peak_range = (60000, 60010)
            sim.main_path = tmp
            sim.events_file = str(Path(tmp) / "events.parquet")
            sim.data_preview = "dp0"
            sim.log_config = {"log_name": "test"}
            sim.max_temp_files = None
            sim.logger = logging.getLogger("test-photometry-processes")
            sim.bands = ["g"]
            sim.photometry_processor_name = "synthetic"

            processor = Mock()
            processor.get_results.return_value = pd.DataFrame()
            with (
                patch("ulens_lsst.simulation_pipeline.Catalog"),
                patch(
                    "ulens_lsst.simulation_pipeline.ParallelProcessor",
                    return_value=processor,
                ),
            ):
                sim.process_synthetic_photometry()

        processor.process.assert_called_once_with(num_pools=1)

    def test_compute_events_chi2_passes_configured_workers_to_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = SimPipeline(
                str(self._write_config(tmp, n_processes=1)), setup_dir=False
            )
            sim.output_dir = tmp
            sim.temp_dir = str(Path(tmp) / "temp")
            Path(sim.temp_dir).mkdir()
            sim.events_file = str(Path(tmp) / "events.parquet")
            sim.input_path = str(Path(tmp) / "temp_input_events.parquet")
            pq.write_table(pa.table({"event_id": [1, 2]}), sim.events_file)
            sim.bands = ["g"]
            sim.chi2_statistic = "median"
            sim.calexps_photometry_file = str(Path(tmp) / "calexps.parquet")
            sim.max_temp_files = None
            sim.log_config = {"log_name": "test"}
            sim.logger = logging.getLogger("test-chi2-processes")

            processor = Mock()
            with patch(
                "ulens_lsst.simulation_pipeline.ParallelProcessor",
                return_value=processor,
            ):
                sim.compute_events_chi2()

        processor.process.assert_called_once_with(num_pools=1)


class TestParallelProcessorWorkerCount(unittest.TestCase):
    def test_one_process_creates_one_pool_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.csv"
            pd.DataFrame({"event_id": range(4)}).to_csv(input_path, index=False)
            processor = ParallelProcessor(
                input_path=str(input_path),
                output_dir=tmp,
                process_fn=_record_worker_pid,
                chunk_size=1,
                checkpoint_interval=1,
                debug=False,
            )

            from multiprocessing.pool import Pool as RealPool
            with patch("ulens_lsst.parallel_utils.Pool", wraps=RealPool) as pool_factory:
                processor.process(num_pools=1)

            pool_factory.assert_called_once_with(processes=1)
            results = processor.get_results()
            self.assertEqual(results["worker_pid"].nunique(), 1)
            self.assertNotEqual(results["worker_pid"].iloc[0], os.getpid())

    def test_non_positive_or_non_integer_pool_sizes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            processor = ParallelProcessor(
                input_path=str(Path(tmp) / "missing.csv"),
                output_dir=tmp,
                process_fn=_record_worker_pid,
            )
            for value in (0, -1, True, 1.5):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "num_pools must be a positive integer"):
                        processor.process(num_pools=value)


if __name__ == "__main__":
    unittest.main()
