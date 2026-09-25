import math

import pyarrow.parquet as pq

from ulens_lsst.scientific_validation import (
    ControlledValidationSource,
    recovery_quantities,
    summarize_validation,
    validation_schema,
    write_validation_rows,
)


def test_recovery_quantities_use_baseline_subtraction():
    result = recovery_quantities(
        expected_flux=100.0,
        pre_flux=20.0,
        post_flux=118.0,
        pre_flux_err=3.0,
        post_flux_err=4.0,
    )

    assert result["recovered_injected_flux"] == 98.0
    assert result["flux_residual"] == -2.0
    assert result["fractional_flux_residual"] == -0.02
    assert result["difference_flux_err"] == 5.0
    assert result["pull"] == -0.4


def test_validation_summary_is_descriptive_not_threshold_based():
    summary = summarize_validation(
        [
            {"pull": 1.0, "fractional_flux_residual": 0.1},
            {"pull": -1.0, "fractional_flux_residual": -0.1},
            {"pull": None, "fractional_flux_residual": None},
        ]
    )

    assert summary["n_rows"] == 3
    assert summary["n_finite_pulls"] == 2
    assert summary["pull_mean"] == 0.0
    assert math.isclose(summary["pull_std"], math.sqrt(2.0))
    assert summary["fractional_residual_mean"] == 0.0


def test_validation_product_is_partition_readable(tmp_path):
    source = ControlledValidationSource("source-1", 10.0, -2.0, 22.0, seed=7)
    path = tmp_path / "validation" / "part-000001.parquet"
    write_validation_rows(
        [
            {
                "validation_id": "image-1:source-1",
                "source_id": source.source_id,
                "image_id": "image-1",
                "requested_mag": source.magnitude,
                "expected_inst_flux": 123.0,
                "injection_status": "success",
                "measurement_status": "success",
            }
        ],
        path,
    )

    table = pq.read_table(path)
    assert table.schema == validation_schema()
    assert table.num_rows == 1
    assert table["requested_mag"][0].as_py() == 22.0
